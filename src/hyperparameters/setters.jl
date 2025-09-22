"""
    change_batch_size(hp, new_batch_size)

Create a new HyperParameters instance with updated batch size.

# Arguments
- `hp`: Original HyperParameters instance
- `new_batch_size`: New batch size value

# Returns
- New HyperParameters instance with updated batch size
"""
function change_batch_size(hp::HyperParameters, new_batch_size::Int)
    return HyperParameters(
        pfm_len = hp.pfm_len,
        num_pfms = hp.num_pfms,
        num_img_filters = hp.num_img_filters,
        img_fil_widths = hp.img_fil_widths,
        img_fil_heights = hp.img_fil_heights,
        pool_base = hp.pool_base,
        stride_base = hp.stride_base,
        poolsize = hp.poolsize,
        stride = hp.stride,
        pool_lvl_top = hp.pool_lvl_top,
        softmax_strength_img_fil = hp.softmax_strength_img_fil,
        batch_size = new_batch_size
    )
end