
"""
    get_num_conv_layers_above_base(hp)

Get the number of convolutional layers above the base layer.

# Arguments
- `hp`: HyperParameters instance

# Returns
- Number of conv layers above base (excludes PFM base layer)
"""
get_num_conv_layers_above_base(hp) = length(hp.num_img_filters)
# TODO: remove this?


"""
    get_input_sequence_length(data)

Extract the input sequence length from data structure.

# Arguments
- `data`: Data structure containing sequence matrices

# Returns
- Length of input sequences (second dimension of first matrix)
"""
function get_input_sequence_length(data)
    return size(data.data_matrices_full[1], 2)
end

# Backward compatibility alias
const get_num_lvl_abv_base = get_num_conv_layers_above_base # TODO change in inference.jl

