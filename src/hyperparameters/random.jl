"""
    generate_random_hyperparameters(; batch_size=nothing, rng=Random.GLOBAL_RNG, ranges=DEFAULT_RANGES)

Generate random hyperparameters for CNN hyperparameter tuning experiments.

This function creates randomized but sensible hyperparameter configurations for
biological sequence CNN training, useful for hyperparameter search and ablation studies.

# Arguments
- `batch_size`: Fixed batch size (if nothing, randomly selected from available options in `ranges.batch_size_options`)
- `rng`: Random number generator to use (default: `Random.GLOBAL_RNG`)
- `ranges`: Struct containing parameter ranges and options (default: `DEFAULT_RANGES`)

# Returns
- `HyperParameters`: Randomized hyperparameter configuration

# Parameter Ranges (from `ranges`)
- `pfm_len`: motif length range (e.g., 6-15)
- `num_pfms`: base layer filter count (e.g., 256-512)
- `conv_filter_counts`: per-layer filter counts (e.g., 256-512)
- `filter_heights`: filter heights for each conv layer (e.g., 2-5)
- `pooling_sizes`: pooling sizes for each conv layer (e.g., 1-2)
- `strides`: stride for each conv layer (e.g., 1)

# Notes
- Last conv layer uses a fixed number of filters for compatibility with final embedding
- Pooling parameters are conservative to avoid over-downsampling
- All stride patterns end with 1 to preserve final resolution
"""
function generate_random_hyperparameters(; 
    batch_size = nothing, 
    rng = Random.GLOBAL_RNG,
    ranges = DEFAULT_RANGES
)
    num_img_layers = rand(rng, ranges.num_img_layers_range)

    @info "Number of image layers: $num_img_layers"
    
    # Core architecture parameters
    pfm_length = rand(rng, ranges.pfm_length_range)
    num_base_filters = rand(rng, ranges.num_base_filters_range)
    
    # Generate conv layer filter counts (preserve last layer size for compatibility)
    conv_filter_counts = [rand(rng, ranges.conv_filter_range) for _ = 1:(num_img_layers-1)]
    push!(conv_filter_counts, ranges.final_layer_filters)  # Fixed final layer size
    
    # Input channels for each conv layer (previous layer's output channels)
    conv_input_channels = vcat([num_base_filters], conv_filter_counts[1:(end-1)])
    
    # Filter dimensions for all layers
    conv_filter_heights = [rand(rng, ranges.conv_filter_height_range) for _ = 1:num_img_layers]
    
    # Pooling configuration (conservative to preserve sequence information)
    conv_pool_sizes = [rand(rng, ranges.pool_size_range) for _ = 1:(num_img_layers-1)]
    push!(conv_pool_sizes, 1)  # No pooling on final layer
    
    conv_strides = [rand(rng, ranges.stride_range) for _ = 1:(num_img_layers-1)]
    push!(conv_strides, 1)  # No stride on final layer
    
    # Training configuration
    max_pooling_layer = num_img_layers - 1  # Standard configuration
    selected_batch_size = isnothing(batch_size) ? rand(rng, ranges.batch_size_options) : batch_size
    
    return HyperParameters(
        pfm_len = pfm_length,
        num_pfms = num_base_filters,
        num_img_filters = conv_filter_counts,
        img_fil_widths = conv_input_channels,
        img_fil_heights = conv_filter_heights,
        pool_base = ranges.base_pool_size,
        stride_base = ranges.base_stride,
        poolsize = conv_pool_sizes,
        stride = conv_strides,
        pool_lvl_top = max_pooling_layer,
        softmax_strength_img_fil = ranges.softmax_alpha,
        batch_size = selected_batch_size,
    )
end


# Backward compatibility alias
const random_hyperparameters = generate_random_hyperparameters

