
# function obtain_output_nuc(hp, m, seq; 
#         tau = fill(float_type(0.5), hp.batch_size) |> cu,
#             make_sparse = false)
#     q = cnn_feature_extraction(hp, m, seq; make_sparse = make_sparse) # code_img_reshape 
        
#     inner_1 = batched_mul(model_weights(m), q)
#     inner_2 = batched_mul(m.w_tau .* reshape(tau, 1, 1, hp.batch_size) .+ m.b_tau, q) # tau weights and biases

#     network_output =
#         m.beta[1] .* inner_1 .+ m.beta[2] .* inner_2
#     return network_output
# end

# function obtain_output_sum_nuc(hp, m::model, seq; make_sparse = false)
#     sum(obtain_output_nuc(hp, m, seq; make_sparse = make_sparse))
# end

# function get_grad_product_nuc(hp, m::model, seq; make_sparse = false)
#     grad = gradient(
#         x->obtain_output_sum_nuc(hp, m, x; make_sparse = make_sparse),
#         seq)
#     grad_nuc_prod = grad[1] .* seq;
#     return grad_nuc_prod
# end
