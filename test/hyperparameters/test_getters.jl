using Test
using ShaneDefaultCNN

@testset "hyperparameters/getters.jl" begin
    hp = ShaneDefaultCNN.HyperParameters()
    # Example: test getter for pfm_len
    @test getfield(hp, :pfm_len) == hp.pfm_len
    # Add more getter tests as needed
end
