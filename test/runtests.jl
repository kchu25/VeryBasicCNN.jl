using VeryBasicCNN
using Test

@testset "VeryBasicCNN.jl" begin
    # Write your tests here.
end

@testset "create_model (CPU)" begin
    # Dummy data dimensions for a typical DNA sequence task
    X_dim = (4, 41)  # (height, sequence_length)
    Y_dim = 10       # e.g., 10 regression targets
    batch_size = 128

    model = ShaneDefaultCNN.create_model(X_dim, Y_dim, batch_size; use_cuda=false)
    if model === nothing
        @info "create_model returned nothing (invalid architecture for these random hyperparameters)"
    else
        @test typeof(model).name.wrapper === ShaneDefaultCNN.SeqCNN
        @test model.hp.batch_size == batch_size
        @test size(model.output_weights, 1) == Y_dim
    end
end

include("hyperparameters/test_defaults.jl")
include("hyperparameters/test_getters.jl")
include("hyperparameters/test_random.jl")

using Flux, ChainRulesCore

@testset "model: fake data, loss, and gradient (CPU)" begin
    # Model dimensions
    X_dim = (4, 300)  # (height, sequence_length)
    Y_dim = 3        # number of outputs
    batch_size = 2

    # Create model (CPU only)
    for _ = 1:2 # run multiple times with different random hyperparameters
        model = ShaneDefaultCNN.create_model(X_dim, Y_dim, batch_size; use_cuda=false)
        if !isnothing(model)
            # Fake input: random floats (not one-hot, but valid shape)
            X = rand(Float32, X_dim[1], X_dim[2], 1, batch_size)
            # Fake targets: random floats
            Y = rand(Float32, Y_dim, batch_size)
            # Mask: all true (no NaNs)
            mask = trues(size(Y))

            # Forward pass
            preds = model(X)
            @test size(preds) == size(Y)

            opt_state = Flux.setup(Flux.AdaBelief(), model)
            # Compute loss and gradients
            loss, gs = Flux.withgradient(model) do x
                ShaneDefaultCNN.masked_mse(x(X), Y, mask)
            end
            @test loss >= 0
            has_grad = any(x -> x !== nothing, gs)
            @test has_grad
        end
    end
end
