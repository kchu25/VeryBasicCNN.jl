"""
    huber_loss(predictions, targets; delta=0.85, quadratic_weight=0.5)

Compute Huber loss with NaN handling for robust regression training.

The Huber loss combines quadratic loss for small errors and linear loss for large errors,
making it robust to outliers while remaining smooth and differentiable.

# Arguments
- `predictions`: Model predictions (ŷ)
- `targets`: Ground truth target values (y)  
- `delta`: Threshold for switching between quadratic and linear loss
- `quadratic_weight`: Weight factor for quadratic loss component (default 0.5)

# Returns
- Mean Huber loss over valid (non-NaN) data points

# Notes
- NaN values in targets are automatically excluded from loss calculation
- For |error| < delta: loss = quadratic_weight * error²
- For |error| ≥ delta: loss = delta * (|error| - quadratic_weight * delta)
"""
function huber_loss(
    predictions, 
    targets; 
    delta = DEFAULT_FLOAT_TYPE(0.85), 
    quadratic_weight = DEFAULT_FLOAT_TYPE(0.5)
)
    # Pre-compute threshold for linear loss component
    linear_threshold = quadratic_weight * delta
    
    # Identify valid data points (exclude NaN targets)
    valid_mask = @ignore_derivatives .!isnan.(targets)
    num_valid_points = @ignore_derivatives sum(valid_mask) |> DEFAULT_FLOAT_TYPE
    
    # Compute absolute errors for valid points only
    absolute_errors = (abs.(predictions - targets))[valid_mask]
    
    # Determine which errors use quadratic vs linear loss
    use_quadratic_loss = @ignore_derivatives absolute_errors .< delta
    use_linear_loss = @ignore_derivatives .!use_quadratic_loss
    
    # Compute loss components
    quadratic_loss = @. quadratic_weight * (absolute_errors^2) * use_quadratic_loss
    linear_loss = @. delta * (absolute_errors - linear_threshold) * use_linear_loss
    
    # Return mean loss over valid points
    return sum(quadratic_loss + linear_loss) / num_valid_points
end


"""
    masked_mse(predictions, targets, mask)

Compute mean squared error only on valid (masked) entries.

# Arguments
- `predictions`: Model predictions
- `targets`: Ground truth targets  
- `mask`: Boolean mask indicating valid entries

# Returns
- MSE computed only on valid entries specified by mask
"""
function masked_mse(predictions, targets, mask)
    # Only compute loss on valid (non-NaN) entries
    valid_predictions = @view predictions[mask]
    valid_targets = @view targets[mask]
    
    # Return mean squared error only for valid entries
    return Flux.mse(valid_predictions, valid_targets) # default agg=mean
end


"""
    compute_training_loss(model, hyperparams, sequences, targets; make_sparse=false, verbose=true)

Compute the training objective (Huber loss) for the biological sequence CNN model.

This function performs a complete forward pass through the model and computes
the robust Huber loss for training the CNN on biological sequence data.

# Arguments
- `model`: Trained or training CNN model instance
- `sequences`: Input biological sequences (typically on GPU)
- `targets`: Ground truth target values for regression
- `make_sparse`: Whether to apply sparsity-inducing filter normalization
- `verbose`: Whether to print loss value during training

# Returns
- Scalar loss value for optimization

# Notes
- Uses Huber loss for robustness to outliers in biological data
- Handles NaN values in targets automatically
- Supports sparse filter training for interpretability
"""
function compute_training_loss(
    m::SeqCNN, 
    sequences, 
    targets; 
    make_sparse = false,
    verbose = true
)
    # Forward pass through the CNN model
    predictions = predict_from_sequences(m, sequences; make_sparse = make_sparse)
    
    # Compute robust Huber loss
    loss = huber_loss(predictions, targets)
    
    # Optional training progress output
    verbose && println("Training loss: $loss")
    
    return loss
end
