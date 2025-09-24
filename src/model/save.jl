function model2cpu(m::SeqCNN)
    img_filters_cpu = 
        [LearnedCodeImgFilters(
            m.img_filters[i].filters |> Array)
            for i in axes(m.img_filters, 1)]
    pwms_cpu = LearnedPWMs(
                    m.pwms.filters |> Array, 
                    m.pwms.activation_scaler)

    return SeqCNN(m.hp, 
                  pwms_cpu, 
                  img_filters_cpu, 
                  m.output_weights |> Array,
                  m.output_scalers |> Array)
end

function model2gpu(m::SeqCNN)
    img_filters_gpu = 
        [LearnedCodeImgFilters(
            m.img_filters[i].filters |> cu)
            for i in axes(m.img_filters, 1)]
    pwms_gpu = LearnedPWMs(
                    m.pwms.filters |> cu, 
                    m.pwms.activation_scaler )

    return SeqCNN(m.hp, 
                  pwms_gpu, 
                  img_filters_gpu, 
                  m.output_weights |> cu,
                  m.output_scalers |> cu)
end


# m_cpu = model2cpu(m);

# @save "cnn_code_sqr2/saved_models/model_1.jld2" m_cpu hp
