using Test
using ShaneDefaultCNN

@testset "hyperparameters/defaults.jl" begin
    hp = ShaneDefaultCNN.HyperParameters()
    @test hp.pfm_len > 0
    @test length(hp.num_img_filters) == length(hp.img_fil_widths) == length(hp.img_fil_heights)
    @test hp.batch_size > 0
end
