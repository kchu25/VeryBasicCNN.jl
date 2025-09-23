module VeryBasicCNN

# Write your package code here.
using Flux, CUDA
using Random 
using ChainRulesCore: @ignore_derivatives


const DEFAULT_FLOAT_TYPE = Float32  # Default float type for model parameters and computations
const first_layer_filter_width = 3
const SOFTMAX_ALPHA = DEFAULT_FLOAT_TYPE(500)
const mutation_scale = DEFAULT_FLOAT_TYPE(25)
const mutation_softmax_alpha = DEFAULT_FLOAT_TYPE(1000)
const Flux_convolve = Flux.NNlib.conv
const output_function = Flux.NNlib.tanh


# Hyperparameter utilities
include("hyperparameters/defaults.jl")
include("hyperparameters/random.jl")
include("hyperparameters/getters.jl")
include("hyperparameters/setters.jl")

# Model architecture - DEPENDENCY ORDER IS CRITICAL
# 1. First: Utility functions (no dependencies on model types)
include("model/utils.jl")         # All utility functions

# 2. Second: Main model struct and layers
include("model/model.jl")         # SeqCNN and layer structs

# 3. Third: Forward pass functions (depends on model structs)
include("model/forward.jl")       # Forward pass & prediction

# 4. Fourth: Loss functions (depends on forward pass)
include("model/loss.jl")          # Loss computation

# Save/load model parameters
include("model/save.jl")     # Save/load model parameters



end
