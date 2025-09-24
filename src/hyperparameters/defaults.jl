"""
    HyperParameters

CNN hyperparameters for biological sequence analysis.

# Fields
- `pfm_len`: Length of Position Frequency Matrix (motif length)
- `num_pfms`: Number of PFMs (base layer filters)
- `num_img_filters`: Number of filters per convolutional layer
- `img_fil_widths`: Input channels for each conv layer
- `img_fil_heights`: Filter heights for each conv layer
- `pool_base`: Pooling size for base layer
- `stride_base`: Stride for base layer
- `poolsize`: Pooling sizes for each conv layer
- `stride`: Strides for each conv layer
- `pool_lvl_top`: Highest layer that uses pooling
- `softmax_strength_img_fil`: Softmax strength for filter normalization
- `batch_size`: Training batch size
"""
Base.@kwdef struct HyperParameters
    pfm_len::Int = 10
    num_pfms::Int = 24
    num_img_filters::Vector{Int} = [65, 98, 128, 128, 76, 5]
    img_fil_widths::Vector{Int} = vcat([num_pfms], num_img_filters[1:(end-1)])
    img_fil_heights::Vector{Int} = [6, 6, 6, 6, 6, 5] # yeast
    pool_base::Int = 2
    stride_base::Int = 1
    poolsize::Vector{Int} = [2, 2, 2, 2, 2, 1] # yeast
    stride::Vector{Int} = [1, 1, 2, 2, 2, 1] # yeast
    pool_lvl_top::Int = 5
    softmax_strength_img_fil::DEFAULT_FLOAT_TYPE = 500.0
    batch_size::Int = 256
end


"""
Configuration struct for hyperparameter ranges with current defaults.
"""
@kwdef struct HyperParamRanges
    # Architecture ranges
    num_img_layers_range = 3:7
    pfm_length_range = 3:15
    num_base_filters_range = 72:12:512
    conv_filter_range = 128:32:512
    conv_filter_height_range = 1:5
    
    # Pooling ranges
    pool_size_range = 1:2
    stride_range = 1:2
    
    # Training ranges
    batch_size_options = [64, 128, 256]
    
    # Fixed parameters
    final_layer_filters = 48
    base_pool_size = 1
    base_stride = 1
    softmax_alpha = SOFTMAX_ALPHA
end

# Default instance with current ranges
const DEFAULT_RANGES = HyperParamRanges()

