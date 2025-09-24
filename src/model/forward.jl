# =============================================================================
# MATRIX NORMALIZATION AND PWM PREPROCESSING UTILITIES
# =============================================================================

"""
    normalize_matrix_through_squaring(matrix; ϵ=1e-5, reverse_comp=false)

Normalize a matrix by squaring elements and normalizing by column sums.
- Squares matrix elements and adds epsilon for numerical stability
- Normalizes by column sums (creates probability-like distribution)
- Optionally creates reverse complement and concatenates along dimension 4

# Arguments
- `matrix`: Input matrix to normalize
- `ϵ`: Small value added for numerical stability (prevents division by zero)
- `reverse_comp`: If true, creates reverse complement and concatenates

# Returns
- Normalized matrix with optional reverse complement concatenated along dimension 4

# Notes
This creates a probability-like distribution from squared values, useful for 
converting frequency matrices to pseudo-probability matrices in PWM construction.
"""
function normalize_matrix_through_squaring(
        matrix; 
        ϵ = DEFAULT_FLOAT_TYPE(1e-5), 
        reverse_comp = false
    )
    # More efficient: fuse operations where possible
    squared_with_eps = @. matrix^2 + ϵ  # Fused broadcast operation
    
    # Use more stable normalization with explicit type handling
    # @ignore_derivatives prevents gradient flow through normalization (treats as fixed preprocessing)
    column_sums = @ignore_derivatives sum(squared_with_eps; dims=1)
    normalized = @. squared_with_eps / column_sums  # Fused broadcast division
    
    # Early return if no reverse complement needed
    reverse_comp || return normalized
    
    # Create reverse complement and concatenate
    reverse_complement = reverse(normalized; dims=(1, 2))
    return cat(normalized, reverse_complement; dims=4)
end

# Apply ReLU activation followed by upper bound clamping
clamp_positive(x; upper_bound = 25) = min.(upper_bound, max.(0, x))

# Square input values and clamp to positive range with upper bound
square_and_clamp(x) = clamp_positive(x .^ 2; upper_bound = DEFAULT_FLOAT_TYPE(0.5))


# For nucleotides (4) and amino acids (20)
const NUCLEOTIDE_BACKGROUND = DEFAULT_FLOAT_TYPE(0.25)
const AMINO_ACID_BACKGROUND = DEFAULT_FLOAT_TYPE(0.05)
const BACKGROUND_LOOKUP = Dict(4 => NUCLEOTIDE_BACKGROUND, 20 => AMINO_ACID_BACKGROUND)

"""
    create_position_weight_matrix(frequencies; background=bg, reverse_comp=false)

Create a Position Weight Matrix (PWM) from frequency matrix.
Converts frequency counts to log2 odds ratios relative to background distribution.

# Arguments
- `frequencies`: Matrix of nucleotide/amino acid frequencies
- `background`: Background probability distribution (defaults to uniform)
- `reverse_comp`: If true, include reverse complement in the PWM

# Returns
- Position Weight Matrix with log2 odds ratios

# Notes
This is standard PWM construction: PWM[i,j] = log2(freq[i,j] / background[i])
"""
function create_position_weight_matrix(frequencies; reverse_comp = false)
    background = @ignore_derivatives get(BACKGROUND_LOOKUP, size(frequencies, 1)) do
        DEFAULT_FLOAT_TYPE(1.0 / size(frequencies, 1))
    end
    # Normalize frequencies to probabilities, then compute log2 odds ratio
    normalized_freq = normalize_matrix_through_squaring(frequencies; reverse_comp = reverse_comp)
    return @. log2(normalized_freq / background)
end

"""
    normalize_conv_filters_l2(filters; softmax_strength=SOFTMAX_ALPHA, use_sparsity=false)

Normalize CNN filters using L2 normalization, with optional sparsity-inducing softmax weighting.

# Arguments
- `filters`: Convolutional filter weights to normalize
- `softmax_strength`: Strength parameter for softmax sparsity (higher = more sparse)
- `use_sparsity`: If true, applies softmax weighting to encourage sparsity

# Returns
- L2-normalized filters, optionally with sparsity-inducing weights

# Notes
- L2 normalization helps with gradient stability and convergence
- Sparsity option uses softmax to emphasize larger weights and suppress smaller ones
"""
function normalize_conv_filters_l2(filters; softmax_strength = SOFTMAX_ALPHA, use_sparsity = false)
    if use_sparsity
        # Apply sparsity-inducing softmax weighting
        abs_filters = @ignore_derivatives abs.(filters)
        sparsity_weights = softmax(softmax_strength .* abs_filters; dims = 2)
        weighted_filters = filters .* sparsity_weights
        
        # L2 normalize the weighted filters
        filter_norm = @ignore_derivatives sqrt.(sum(weighted_filters .^ 2; dims = (1, 2)))
        return @. weighted_filters / filter_norm
    else
        # Standard L2 normalization
        filter_norm = @ignore_derivatives sqrt.(sum(filters .^ 2; dims = (1, 2)))
        return @. filters / filter_norm
    end
end

function img_pass(model, code; make_sparse=false)
    # Base layer pooling
    code_img = apply_pooling_to_code(
        code;
        poolsize = get_pooling_tuple(model.hp, 0),
        stride = get_stride_tuple(model.hp, 0),
        is_base_layer = true,
    )
    # Process all convolutional layers
    for layer_idx = 1:model.num_img_layers
        # Convolution
        code = model.img_filters[layer_idx](code_img, model.hp; make_sparse = make_sparse)

        # Pooling (identity pooling for layers beyond pool_lvl_top)
        code_img = apply_pooling_to_code(
            code;
            poolsize = get_pooling_tuple(model.hp, layer_idx),
            stride = get_stride_tuple(model.hp, layer_idx),
            identity = layer_idx > model.hp.pool_lvl_top,
        )
    end
    # Flatten to feature embedding vector
    embedding_length = size(code_img, 1) * size(code_img, 2)
    batch_size = size(code_img, 4)
    feature_embedding = reshape(code_img, (embedding_length, 1, batch_size))

    return feature_embedding
end

function cnn_feature_extraction(model::SeqCNN, sequences; 
        make_sparse = false, 
        get_first_layer_code = false
        )
    # Base layer: PWM convolution
    code = model.pwms(sequences)
    feature_embedding = img_pass(model, code; make_sparse=make_sparse)
    
    if get_first_layer_code
        return feature_embedding, code
    end
    return feature_embedding
end

"""
    get the output weights for a given position or all positions
"""
function determine_output_weights(model::SeqCNN; inference_position=nothing)
    if isnothing(inference_position)
        output_weights = model.output_weights
    else
        @assert 1 ≤ inference_position ≤ size(model.output_weights, 1) "inference_position out of bounds"
        output_weights = @view model.output_weights[inference_position:inference_position, :, :]
    end
    return output_weights
end

function get_predictions(linear_output)
    output_dim, batch_size = size(linear_output, 1), size(linear_output, 3)
    # Return 1D for single output, 2D for multi-output
    predictions = output_dim == 1 ? 
        reshape(linear_output, (batch_size,)) : 
        reshape(linear_output, (output_dim, batch_size))
    return predictions
end

"""
    predict_from_sequences(hyperparams, model, sequences; make_sparse=false)

Complete forward pass: CNN feature extraction → linear transformation → final predictions.

# Arguments
- `hyperparams`: CNN hyperparameters
- `model`: Trained CNN model instance  
- `sequences`: Input biological sequences
- `make_sparse`: Apply sparsity-inducing normalization

# Returns
- Model predictions (output_dim, batch_size)
"""
function predict_from_sequences(model::SeqCNN, sequences; 
        make_sparse = false, 
        inference_position=nothing)
    # Extract CNN features from sequences
    feature_embedding = cnn_feature_extraction(
        model, sequences; make_sparse = make_sparse) # (final_embedding_length, 1, batch_size)

    output_weights = determine_output_weights(model; inference_position=inference_position)

    # Apply linear transformation (final dense layer)
    linear_output = batched_mul(output_weights, feature_embedding)
    # output_weights is (output_dimension, final_embedding_dim, 1)
    # so the batched_mul gives (output_dim, 1, batch_size)

    return get_predictions(linear_output)
end

"""
    predict_from_code(model, code; make_sparse=false, inference_position=nothing)
"""
function predict_from_code(model, code; make_sparse=false, inference_position=nothing)
    feature_embedding = img_pass(model, code; make_sparse=make_sparse)
    output_weights = determine_output_weights(model; inference_position=inference_position)
    linear_output = batched_mul(output_weights, feature_embedding)
    return get_predictions(linear_output)
end


# Backward compatibility alias
const model_forward_pass = predict_from_sequences # TODO change in inference.jl

"""
    (m::SeqCNN)(seq; make_sparse=false, linear_sum=false, inference_position=nothing)

Call overload for `SeqCNN` models to perform a forward pass and generate predictions.

# Arguments
- `seq`: Input biological sequences (one-hot encoded or compatible format)
- `make_sparse`: If true, applies sparsity-inducing normalization to convolutional filters (default: false)
- `linear_sum`: (Unused in this model; included for compatibility)
- `inference_position`: If provided, returns predictions for a specific output position (default: `nothing` for all outputs)

# Returns
- Model predictions as an array of shape `(output_dim, batch_size)` 
- or `(batch_size,)` for single-output models

# Notes
This enables the model to be called as a function, e.g., `preds = model(sequences)`, for convenient ML-style usage.
"""
function (m::SeqCNN)(seq; 
    make_sparse = false, 
    linear_sum=false, 
    inference_position=nothing
    )
    # no if else on linear_sum, because in this model outputs a linear sum
    return predict_from_sequences(m, seq; 
        make_sparse = make_sparse, 
        inference_position = inference_position
        )
end

"""
    Base.getproperty(m::SeqCNN, sym::Symbol)

Allow dot-access to `batch_size` as a virtual field for a `SeqCNN` model.

- `model.batch_size` returns the batch size from the model's hyperparameters (`model.hp.batch_size`).
- All other fields are accessed as usual.

This enables convenient and familiar ML-style access to the batch size without storing redundant fields.
"""
function Base.getproperty(m::SeqCNN, sym::Symbol)
    if sym === :batch_size
        return m.hp.batch_size
    elseif sym === :num_img_layers
        return length(m.hp.num_img_filters)
    elseif sym === :prediction_from_code_then_sum
        # return a function that takes the code (first layer) as an input
        return (x; kwargs...) -> sum(predict_from_code(m, x; kwargs...))
    elseif sym === :first_layer_code
        # return a function that outputs the code (first layer)
        # haven't done reshaping -- do it later if needed
        return x->m.pwms(x)
    else
        # need this line otherwise all other fields won't be accessible
        return getfield(m, sym)
    end
end

