using Test
using ShaneDefaultCNN

@testset "hyperparameters/random.jl" begin
    hp = ShaneDefaultCNN.generate_random_hyperparameters()
    @test hp.pfm_len > 0
    @test hp.batch_size > 0
    @test length(hp.num_img_filters) == length(hp.img_fil_widths) == length(hp.img_fil_heights)
end
