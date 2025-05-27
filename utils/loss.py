import torch


def bradley_terry_loss(rewards1, rewards2, preferences, cost=None, alpha=3.5, beta_max=8.0, eps = 1e-6):
    """
    Compute the Bradley-Terry preference learning loss (binary cross-entropy) with optional beta scaling.

    Args:
        rewards1: Predicted rewards for the first segments in each pair
                 Shape can be [batch_size] or [num_models, batch_size]
        rewards2: Predicted rewards for the second segments in each pair
                 Shape can be [batch_size] or [num_models, batch_size]
        preferences: Labels indicating which segment is preferred (1 = first segment preferred, 0 = second segment preferred, 0.5 = equal)
                    Shape is [batch_size]
        cost: DTW Cost used for beta scaling (optional)
                    
    Returns:
        Loss value if rewards are [batch_size]
        Loss tensor of shape [num_models] if rewards are [num_models, batch_size]
    """

    # Convert preferences to probabilities (1 = first segment preferred, 0 = second segment preferred, 0.5 = equal)
    prefs = preferences.float()

    # Optionally use cost to scale beta if provided
    if cost is not None:
        beta = beta_max * torch.exp(-alpha * cost)
    else:
        beta = 1

    # Compute probability that segment1 is preferred over segment2 using the Bradley-Terry model
    logits = torch.clamp(
        beta * (rewards1 - rewards2), min=-50.0, max=50.0
    )
    pred_probs = torch.sigmoid(logits)

    # For equal preferences (0.5), we want the model to predict 0.5
    # For preference 1, we want the model to predict close to 1
    # For preference 0, we want the model to predict close to 0
    bce = -(
        prefs * torch.log(pred_probs + eps)
        + (1 - prefs) * torch.log(1 - pred_probs + eps)
    )

    # Return mean over batch dimension
    return torch.mean(bce, dim=-1)  # Mean over last dimension (batch)
