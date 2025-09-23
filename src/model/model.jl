"""
    LearnedPWMs

Learnable Position Weight Matrices for the base CNN layer.

# Fields
- `filters`: 4D array of PWM filters (height, width, channels, num_filters)
- `activation_thresholds`: Activation threshold parameters for each filter

# Constructor Arguments
- `filter_width`: Width of PWM filters (motif length)
- `num_filters`: Number of PWM filters to create
- `init_scale`: Initialization scale factor (pseudo count)
- `use_cuda`: Whether to move filters to GPU
"""
struct LearnedPWMs
    filters::AbstractArray{DEFAULT_FLOAT_TYPE,4}
    activation_scaler::AbstractArray{DEFAULT_FLOAT_TYPE,1}
    
    function LearnedPWMs(;
        filter_width::Int = DEFAULT_FILTER_WIDTH,
        filter_height::Int = 4,  # Default for nucleotides (A, C, G, T)
        num_filters::Int = 48,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1e-1),
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG,
    )
        # Initialize PWM filters with small random values
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (filter_height, filter_width, 1, num_filters))

        # Initialize activation thresholds
        activation_scaler = init_scale .* rand(rng, DEFAULT_FLOAT_TYPE, 1)

        if use_cuda
            filters = cu(filters)
        end

        return new(filters, activation_scaler)
    end
    
    # Direct constructor for pre-existing arrays
    function LearnedPWMs(filters, activation_scaler)
        return new(filters, activation_scaler)
    end
end

Flux.@layer LearnedPWMs


function first_lvl_prep_param(lpwms::LearnedPWMs; reverse_comp = false)
    return create_position_weight_matrix(lpwms.filters; reverse_comp = reverse_comp), 
           square_and_clamp(lpwms.activation_scaler)
end

# Init the sparse code according to the learned PWM objective
init_code_learned_pwm(grad, etas) = Flux.NNlib.relu(etas[1] .* grad)

function first_lvl_forward_pass(lpwms, S; return_pwms = false, reverse_comp = false)
    pwms, etas = first_lvl_prep_param(lpwms; reverse_comp = reverse_comp)
    grad = Flux_convolve(S, pwms; pad = 0, flipped = true)
    code = init_code_learned_pwm(grad, etas)
    return_pwms && (return code, pwms)
    return code
end

function (lpwms::LearnedPWMs)(S; return_pwms = false)
    return first_lvl_forward_pass(lpwms, S; return_pwms = return_pwms)
end

"""
    LearnedCodeImgFilters

Learnable convolutional filters for intermediate CNN layers.

# Fields
- `filters`: 4D array of convolutional filters (height, width, channels, num_filters)
- `activation_scaler`: Scalar activation parameter for the layer

# Constructor Arguments
- `input_channels`: Number of input channels (width of previous layer's output)
- `filter_height`: Height of convolutional filters
- `num_filters`: Number of filters in this layer
- `init_scale`: Initialization scale factor
- `use_cuda`: Whether to move filters to GPU
"""
struct LearnedCodeImgFilters
    filters::AbstractArray{DEFAULT_FLOAT_TYPE,4}
    
    function LearnedCodeImgFilters(;
        input_channels::Int,
        filter_height::Int = 6,
        num_filters::Int = 24,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1e-3),
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG,
    )
        # Initialize convolutional filters
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (filter_height, input_channels, 1, num_filters))

        if use_cuda
            filters = cu(filters)
        end
        
        return new(filters)
    end
    
    # Direct constructor for pre-existing arrays
    function LearnedCodeImgFilters(filters)
        return new(filters)
    end
end

# Backward compatibility alias
const learned_codeimg_filters = LearnedCodeImgFilters

Flux.@layer LearnedCodeImgFilters


"""
    prepare_codeimg_layer_params(conv_filters; alpha=SOFTMAX_ALPHA, make_sparse=false)

Prepare normalized filters and activation scalers for code image convolution layer.

# Arguments
- `conv_filters`: LearnedCodeImgFilters instance containing raw filters and scalers
- `alpha`: Softmax strength for sparsity-inducing normalization
- `make_sparse`: Whether to apply sparsity-inducing normalization

# Returns
- `normalized_filters`: L2-normalized convolutional filters, optionally with sparsity
- `activation_scaler`: Processed activation scaler (squared and clamped)
"""
function prepare_codeimg_layer_params(
    conv_filters::LearnedCodeImgFilters; 
    alpha = SOFTMAX_ALPHA, 
    make_sparse = false
    )
    normalized_filters =
        normalize_conv_filters_l2(conv_filters.filters; 
            softmax_strength = alpha, use_sparsity = make_sparse)
    return normalized_filters
end


"""
    initialize_codeimg_activations(gradient)

Initialize code image activations using ReLU activation.

# Arguments
- `gradient`: Raw convolution output gradients

# Returns
- ReLU-activated code image
"""
initialize_codeimg_activations(gradient) = Flux.NNlib.relu(gradient)

# Backward compatibility alias
const init_code_img_fils = initialize_codeimg_activations

"""
    codeimg_layer_forward_pass(conv_filters, prev_code_img, hp; return_filters=false, make_sparse=false)

Forward pass through a code image convolutional layer.

# Arguments
- `conv_filters`: LearnedCodeImgFilters instance
- `prev_code_img`: Previous layer's code image output
- `hp`: HyperParameters containing layer configuration
- `return_filters`: Whether to return normalized filters along with output
- `make_sparse`: Whether to apply sparsity-inducing filter normalization

# Returns
- `code`: Activated code image output
- `normalized_filters`: (Optional) Normalized filters if return_filters=true
"""
function codeimg_layer_forward_pass(
    conv_filters::LearnedCodeImgFilters,
    prev_code_img,
    hp::HyperParameters;
    return_filters = false,
    make_sparse = false,
)
    normalized_filters = prepare_codeimg_layer_params(
        conv_filters;
        alpha = hp.softmax_strength_img_fil,
        make_sparse = make_sparse,
    )
    gradient = Flux_convolve(prev_code_img, normalized_filters; pad = 0, flipped = true)
    code = initialize_codeimg_activations(gradient)
    return_filters && (return code, normalized_filters)
    return code
end


function (conv_filters::LearnedCodeImgFilters)(
    prev_code_img,
    hp::HyperParameters;
    return_filters = false,
    make_sparse = false,
)
    return codeimg_layer_forward_pass(
        conv_filters,
        prev_code_img,
        hp;
        return_filters = return_filters,
        make_sparse = make_sparse,
    )
end

const FILTER_HEIGHT_MAP = Dict(:Nucleotide => 4, :AminoAcid => 20)

"""
    SeqCNN

Convolutional neural network for biological sequence analysis and regression.

This model implements a multi-layer CNN architecture specifically designed for 
biological sequence data (DNA, RNA, proteins) with learnable Position Weight 
Matrices (PWMs) as the base layer followed by convolutional layers.

# Architecture Overview
1. **Base Layer**: Learnable PWMs for motif detection
2. **Convolutional Layers**: Code image filters with pooling
3. **Final Layer**: Linear transformation to target outputs

# Fields
- `position_weight_matrices::LearnedPWMs`: Base layer PWM filters for motif detection
- `convolutional_filters::Vector{LearnedCodeImgFilters}`: Multi-layer conv filters
- `output_weights::AbstractArray{Float,3}`: Final linear layer weights (output_dim × embed_dim × 1)
- `output_scalers::AbstractArray{Float,1}`: Output scaling parameters

# Constructor Arguments
- `hyperparams`: HyperParameters containing architecture specification
- `sequence_length::Integer`: Length of input biological sequences
- `output_dimension=1`: Number of output targets (e.g., binding affinities)
- `initialization_scale=0.5`: Weight initialization scale factor
- `use_cuda=true`: Whether to place model on GPU

# Example
```julia
# Create model for 41-length DNA sequences with 244 output targets
hp = generate_random_hyperparameters()
model = SeqCNN(hp, 41; output_dimension=244)

# Forward pass
predictions = model(sequences)  # Returns (244, batch_size)
```

# Notes
- Automatically calculates embedding dimensions based on architecture
- Supports both single and multi-output regression tasks
- Optimized for biological sequence lengths (typically 20-200 nucleotides)
"""
struct SeqCNN
    hp::HyperParameters
    pwms::LearnedPWMs
    img_filters::Vector{LearnedCodeImgFilters}
    output_weights::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    output_scalers::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    
    function SeqCNN(
        hp::HyperParameters,
        Xdim::Tuple{T, T},
        Ydim::T;
        initialization_scale = DEFAULT_FLOAT_TYPE(5e-1), 
        use_cuda = true,
        rng = Random.GLOBAL_RNG
    ) where T <: Integer

        # check input dimensions
        @assert length(Xdim) == 2 "Xdim must be a tuple of (height, sequence_length)"

        # obtain input and output dimensions        
        filter_height = Xdim[1]
        sequence_length = Xdim[2]
        output_dimension = Ydim

        # Initialize base layer PWM filters for motif detection
        pwms = LearnedPWMs(;
            filter_width = hp.pfm_len,
            filter_height = filter_height,
            num_filters = hp.num_pfms,
            init_scale = initialization_scale,
            use_cuda = use_cuda,
            rng = rng,
        )

        # Initialize convolutional layers for hierarchical feature extraction
        img_filters = [
            LearnedCodeImgFilters(; 
                input_channels = input_channels, 
                filter_height = filter_height, 
                num_filters = num_filters, 
                init_scale = initialization_scale, 
                use_cuda = use_cuda,
                rng = rng,
            ) for (input_channels, filter_height, num_filters) in zip(
                hp.img_fil_widths, 
                hp.img_fil_heights, 
                hp.num_img_filters
            )
        ]

        # Initialize output scaling parameters  
        output_scalers = randn(rng, DEFAULT_FLOAT_TYPE, (output_dimension, 1, 1))

        if use_cuda && output_dimension > 1
            output_scalers = output_scalers |> cu
        end

        # Calculate embedding dimension after final convolutional layer
        final_conv_length = calculate_final_conv_embedding_length(hp, sequence_length)
        final_embedding_dim = final_conv_length * hp.num_img_filters[end]

        # Initialize final linear layer weights
        output_weights = randn(rng, DEFAULT_FLOAT_TYPE, 
            (output_dimension, final_embedding_dim, 1))
        if use_cuda && (output_dimension*final_embedding_dim > 1)
            output_weights = output_weights |> cu
        end

        return new(hp, pwms, img_filters, output_weights, output_scalers)
    end
    
    # Direct constructor for pre-existing components (for model loading/conversion)
    function SeqCNN(hp, pwms, img_filters, output_weights, output_scalers)
        return new(hp, pwms, img_filters, output_weights, output_scalers)
    end

end

Flux.@layer SeqCNN

Flux.trainable(m::SeqCNN) = (
    pwms = m.pwms, 
    img_filters = m.img_filters, 
    output_weights = m.output_weights, 
    output_scalers = m.output_scalers
    )

"""
    model_init(hp::HyperParameters, X_dim::Tuple{T, T}, Y_dim::T; kwargs...) where T <: Integer
Create a SeqCNN model instance given hyperparameters and data dimensions.
"""
function model_init(hp::HyperParameters, X_dim::Tuple{T, T}, Y_dim::T; kwargs...) where T <: Integer
    return SeqCNN(hp, X_dim, Y_dim; kwargs...)
end

"""
    create_model(X_dim, Y_dim, batch_size::Int; rng=Random.GLOBAL_RNG, use_cuda::Bool=true)

Construct a new `SeqCNN` model with randomly generated hyperparameters and specified data dimensions.

# Arguments
- `X_dim::Tuple{Int, Int}`: Input data dimensions as (height, sequence_length).
- `Y_dim::Int`: Number of output targets (e.g., regression outputs).
- `batch_size::Int`: Batch size for hyperparameter generation.
- `rng`: (Optional) Random number generator to use for hyperparameter sampling (default: `Random.GLOBAL_RNG`).
- `use_cuda::Bool`: (Optional) Whether to place model parameters on GPU (default: `true`).

# Returns
- A new `SeqCNN` model instance if the final embedding length is valid (≥ 1), otherwise `nothing`.

# Notes
- Hyperparameters are generated randomly for each call unless a specific RNG is provided.
- Returns `nothing` if the model architecture is invalid for the given input dimensions.
- The `use_cuda` flag controls whether model parameters are moved to GPU.
"""
function create_model(X_dim, Y_dim, batch_size::Int; 
        rng=Random.GLOBAL_RNG, 
        use_cuda::Bool = true)
    if isnothing(rng)
        hp = HyperParameters()
    else
        hp = generate_random_hyperparameters(; 
            batch_size=batch_size, rng=rng)
    end
    
    if final_embedding_length(hp, X_dim) ≥ 1
        return model_init(hp, X_dim, Y_dim; use_cuda=use_cuda, rng=rng)
    else
        @warn "Invalid model architecture due to bad final embedding length. Returning `nothing`."
        return nothing
    end
end


"""
    get_output_weights(model::SeqCNN)

Extract the output layer weights from a SeqCNN model.

# Arguments
- `model`: SeqCNN instance

# Returns  
- Output weights tensor (output_dim, embedding_dim, 1)
"""
get_output_weights(m::SeqCNN) = m.output_weights

# ==============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ==============================================================================

# Backward compatibility aliases for the model struct and functions
const model = SeqCNN
const model_weights = get_output_weights # TODO change in inference.jl
