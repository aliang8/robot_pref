import torch

def bradley_terry_loss(rewards1, rewards2, preferences):
    """
    Compute the Bradley-Terry preference learning loss (binary cross-entropy).
    
    Args:
        rewards1: Predicted rewards for the first segments in each pair
                 Shape can be [batch_size] or [num_models, batch_size]
        rewards2: Predicted rewards for the second segments in each pair
                 Shape can be [batch_size] or [num_models, batch_size]
        preferences: Labels indicating which segment is preferred (1 or 2)
                    Shape is [batch_size]

    Returns:
        Loss value if rewards are [batch_size]
        Loss tensor of shape [num_models] if rewards are [num_models, batch_size]
    """
    # Convert preferences to probabilities (1 = first segment preferred, 2 = second segment preferred)
    # Use detach and clone for tensor conversion if input is already a tensor
    if isinstance(preferences, torch.Tensor):
        prefs = (preferences == 1).float()
    else:
        prefs = (torch.tensor(preferences) == 1).float()

    # Compute probability that segment1 is preferred over segment2 using the Bradley-Terry model
    # Add a small epsilon for numerical stability
    eps = 1e-6
    logits = torch.clamp(
        rewards1 - rewards2, min=-50.0, max=50.0
    )  # Clip logits to prevent overflow
    pred_probs = torch.sigmoid(logits)
    
    # Standard binary cross-entropy loss: -(y*log(p) + (1-y)*log(1-p))
    # This is negative log likelihood (higher means worse fit)
    bce = -(prefs * torch.log(pred_probs + eps) + (1 - prefs) * torch.log(1 - pred_probs + eps))
    
    # Return mean over batch dimension
    return torch.mean(bce, dim=-1)  # Mean over last dimension (batch)
