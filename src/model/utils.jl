# =============================================================================
# CNN Layer Parameter Access Functions
# =============================================================================

"""
    get_pooling_size(hyperparams, layer_level)

Get the pooling size for a specific CNN layer.
- Layer 0 (base): uses pool_base
- Other layers: uses poolsize[layer_level]
"""
get_pooling_size(hp, layer_level) = layer_level == 0 ? hp.pool_base : hp.poolsize[layer_level]

"""
    get_pooling_tuple(hyperparams, layer_level)

Get the pooling size as a tuple (height, width) for convolution operations.
Returns (pooling_size, 1) since these are 1D sequences treated as 2D.
"""
get_pooling_tuple(hp, layer_level) = layer_level == 0 ? (hp.pool_base, 1) : (hp.poolsize[layer_level], 1)

"""
    get_stride_size(hyperparams, layer_level)

Get the stride size for a specific CNN layer.
- Layer 0 (base): uses stride_base  
- Other layers: uses stride[layer_level]
"""
get_stride_size(hp, layer_level) = layer_level == 0 ? hp.stride_base : hp.stride[layer_level]

"""
    get_stride_tuple(hyperparams, layer_level)

Get the stride size as a tuple (height, width) for convolution operations.
Returns (stride_size, 1) since these are 1D sequences treated as 2D.
"""
get_stride_tuple(hp, layer_level) = layer_level == 0 ? (hp.stride_base, 1) : (hp.stride[layer_level], 1)



# =============================================================================
# CNN Layer Dimension Calculation Functions
# =============================================================================

"""
    calculate_conv_output_length(input_length, filter_length)

Calculate output length after convolution operation.
Formula: output_length = input_length - filter_length + 1
"""
calculate_conv_output_length(input_length, filter_length) = input_length - filter_length + 1

"""
    calculate_pooled_length(input_length, hyperparams, layer_level)

Calculate output length after pooling and striding operations.
Formula: output_length = (input_length - pool_size) ÷ stride + 1
"""
function calculate_pooled_length(input_length, hyperparams, layer_level)
    pool_size = get_pooling_size(hyperparams, layer_level)
    stride = get_stride_size(hyperparams, layer_level)
    return (input_length - pool_size) ÷ stride + 1
end

"""
    calculate_base_layer_output_length(hyperparams, sequence_length)

Calculate the output length after the base layer (first convolution with PFM filters).
Formula: output_length = sequence_length - pfm_length + 1

This represents the number of possible positions where a PFM of length pfm_len
can be placed on a sequence of length sequence_length.
"""
calculate_base_layer_output_length(hyperparams, sequence_length) = 
    sequence_length - hyperparams.pfm_len + 1

"""
    calculate_pooling_output_dimension(input_dimension, poolsize, stride)

Calculate the output dimension after pooling operation.
Standard pooling formula: output = (input - poolsize) ÷ stride + 1

# Arguments
- `input_dimension`: Size of input dimension
- `poolsize`: Size of pooling kernel
- `stride`: Stride of pooling operation

# Returns
- Output dimension after pooling
"""
calculate_pooling_output_dimension(input_dimension, poolsize, stride) = 
    (input_dimension - poolsize) ÷ stride + 1


"""
    calculate_conv_then_pooled_length(input_length, filter_length, pool_size, stride)

Calculate output length after convolution followed by pooling and stride operations.
Combines convolution and pooling calculations: conv → pool

# Arguments
- `input_length`: Length of input sequence/tensor
- `filter_length`: Length of convolutional filter
- `pool_size`: Size of pooling kernel
- `stride`: Stride of pooling operation

# Returns
- Final output length after convolution and pooling
"""
calculate_conv_then_pooled_length(input_length, filter_length, pool_size, stride) =
    calculate_pooling_output_dimension(
        calculate_conv_output_length(input_length, filter_length), 
        pool_size, stride
    )

"""
    calculate_final_conv_embedding_length(hyperparams, sequence_length)

Calculate the embedding length after the final convolutional layer.

This function simulates the forward pass through all CNN layers to determine
the final conv embedding dimension that will be fed to downstream layers (e.g., dense layers):
1. Base layer: convolution with PFM filters → pooling
2. For each subsequent layer up to pool_lvl_top: conv → pool  
3. For remaining layers: conv only (no pooling)

The result is the length of the convolutional feature vector produced by the last conv layer,
which represents the final conv embedding of the input biological sequence.

# Arguments
- `hyperparams`: CNN hyperparameters containing layer specifications
- `sequence_length`: Length of input biological sequence

# Returns  
- Final conv embedding length (dimension of feature vector from last conv layer)
"""
function calculate_final_conv_embedding_length(hp, sequence_length)
    # Start with base layer (PFM convolution + pooling)
    current_length = calculate_base_layer_output_length(hp, sequence_length)
    current_length = calculate_pooled_length(current_length, hp, 0)

    # Process each subsequent layer
    num_layers = hp.num_img_filters |> length
    for layer = 1:num_layers
        if layer ≤ hp.pool_lvl_top
            # Layers with both convolution and pooling
            current_length = calculate_conv_output_length(current_length, hp.img_fil_heights[layer])
            current_length = calculate_pooled_length(current_length, hp, layer)
        else
            # Layers with only convolution (no pooling)
            current_length = calculate_conv_output_length(current_length, hp.img_fil_heights[layer])
        end
    end
    
    return current_length
end

"""
    final_embedding_len(hp, X_dim::Tuple{T, T}) where T <: Integer
"""
function final_embedding_length(hp, X_dim::Tuple{T, T}) where T <: Integer
    return calculate_final_conv_embedding_length(hp, X_dim[2])
end


"""
    get_level_dimensions(tensor; is_base_layer=false)

Get relevant tensor dimensions for CNN layer operations.

For biological sequence CNNs, this function extracts the dimensions needed
for pooling and reshaping operations, handling the different tensor layouts
between base layer and subsequent layers.

# Arguments
- `tensor`: Input tensor (4D: height, width, channels, batch)
- `is_base_layer`: If true, treats as base layer with different dimension indexing

# Returns
- Tuple of (relevant_dim, channels, batch_size) for layer operations

# Notes
- Base layer: uses dimensions (2, 3, 4) → (width, channels, batch)  
- Other layers: uses dimensions (1, 3, 4) → (height, channels, batch)
"""
function get_level_dimensions(tensor; is_base_layer = false)
    if is_base_layer
        return size(tensor, 2), size(tensor, 3), size(tensor, 4)
    end
    return size(tensor, 1), size(tensor, 3), size(tensor, 4)
end


# ==============================================================================
# POOLING OPERATIONS
# ==============================================================================

"""
    apply_maxpool_to_code_image(code_image; poolsize=(2, 1), stride=(1, 1))

Apply max pooling to 4D code image and reshape to maintain tensor structure.

This function applies max pooling to code images from convolutional layers,
automatically calculating the correct output dimensions and reshaping the result
to maintain the expected 4D tensor format (height, width, channels, batch).

# Arguments
- `code_image`: 4D tensor of code image (height, width, channels, batch)
- `poolsize`: Pooling kernel size as (height, width) tuple
- `stride`: Stride of pooling operation as (height, width) tuple

# Returns
- Pooled 4D code image with updated dimensions
"""
function apply_maxpool_to_code_image(code_image; poolsize = (2, 1), stride = (1, 1))
    height, width, channels, batch_size = size(code_image)
    
    # Calculate output height after pooling (width stays same for 1D sequences)
    pooled_height = @ignore_derivatives calculate_pooling_output_dimension(height, poolsize[1], stride[1])
    
    # Apply max pooling and reshape to maintain 4D tensor structure
    pooled_code_image = reshape(
        Flux.NNlib.maxpool(code_image, poolsize; pad = 0, stride = stride),
        (pooled_height, width, channels, batch_size),
    )
    
    return pooled_code_image
end


"""
    apply_pooling_to_code(code; poolsize=(2,1), stride=(1,1), is_base_layer=false, identity=false, weights=nothing)

Apply pooling to CNN code with optional weighting and identity bypass.

# Arguments
- `code`: Input code to pool (3D or 4D)
- `poolsize`: Pooling kernel size, default (2,1) for 1D sequences
- `stride`: Pooling stride, default (1,1)
- `is_base_layer`: If true, handles base layer indexing differently
- `identity`: If true, bypasses pooling (for skip connections)
- `weights`: Optional weighting for attention-like mechanisms

# Returns
- Pooled 4D code (height, width, channels, batch)
"""
function apply_pooling_to_code(
    code;
    poolsize = (2, 1),
    stride = (1, 1),
    is_base_layer = false,
    identity = false,
    weights = nothing,
)
    # Extract tensor dimensions based on layer type
    relevant_dim, channels, batch_size = 
        get_level_dimensions(code; is_base_layer = is_base_layer)
    
    # Identity bypass: reshape to 4D format without pooling (for skip connections)
    identity && (return reshape(code, relevant_dim, channels, 1, batch_size))
    
    # Reshape to 4D image format for pooling operations
    code_img = reshape(code, relevant_dim, channels, 1, batch_size)
    
    # Apply optional weighting (for attention-like mechanisms)
    if !isnothing(weights)
        code_img = weights .* code_img
    end
    
    # Apply max pooling and return result
    pooled_code = apply_maxpool_to_code_image(code_img; poolsize = poolsize, stride = stride)
    return pooled_code
end


# Backward compatibility aliases
const get_pool = get_pooling_tuple  # TODO: change in src/model/inference.jl
const get_stride = get_stride_tuple  # TODO: change in src/model/inference.jl
const code_pool = apply_pooling_to_code   # TODO: change in src/model/inference.jl

