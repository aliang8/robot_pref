import torch


def bradley_terry_loss(rewards1, rewards2, preferences, cost=None, alpha=1.0, beta_max=1.0, eps = 1e-6):
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


def distributional_bradley_terry_loss(
    reward_means1, reward_means2, 
    reward_vars1, reward_vars2,
    preferences, 
    lambda_weight=0.1,
    num_samples=5,
    beta=1.0,
):
    """
    Compute the distributional Bradley-Terry preference learning loss as described in Equation 7.
    
    This combines:
    1. Cross-entropy loss on reward means
    2. Cross-entropy loss on sampled rewards (weighted by lambda)
    
    Args:
        reward_means1: Predicted reward means for first segments [batch_size, seq_len]
        reward_means2: Predicted reward means for second segments [batch_size, seq_len]
        reward_vars1: Predicted reward variances for first segments [batch_size, seq_len]
        reward_vars2: Predicted reward variances for second segments [batch_size, seq_len]
        preferences: Labels indicating which segment is preferred (1 or 2) [batch_size]
        lambda_weight: Weight for the stochastic term (lambda in Equation 7)
        num_samples: Number of samples K for approximating the expectation
        beta: Scaling factor for the sigmoid function
    """
    # Convert preferences to probabilities (1 = first segment preferred, 2 = second segment preferred)
    if isinstance(preferences, torch.Tensor):
        prefs = (preferences == 1).float()
    else:
        prefs = (torch.tensor(preferences) == 1).float()

    eps = 1e-6
    
    # Sum over time dimension to get returns
    return_means1 = reward_means1.sum(dim=1)  # [batch_size]
    return_means2 = reward_means2.sum(dim=1)  # [batch_size]

    # First term: Cross-entropy loss on reward means
    logits_mean = torch.clamp(
        beta * (return_means1 - return_means2), min=-50.0, max=50.0
    )
    pred_probs_mean = torch.sigmoid(logits_mean).squeeze()
    
    bce_mean = -(
        prefs * torch.log(pred_probs_mean + eps)
        + (1 - prefs) * torch.log(1 - pred_probs_mean + eps)
    )
    loss_mean = torch.mean(bce_mean)
    
    # Second term: Cross-entropy loss on sampled rewards
    std1 = torch.sqrt(reward_vars1)
    std2 = torch.sqrt(reward_vars2)
    
    loss_samples = 0.0
    for _ in range(num_samples):
        # Sample from Gaussian distributions
        eps1 = torch.randn_like(reward_means1)
        eps2 = torch.randn_like(reward_means2)
        
        sampled_rewards1 = reward_means1 + std1 * eps1  # [batch_size, seq_len]
        sampled_rewards2 = reward_means2 + std2 * eps2  # [batch_size, seq_len]
        
        # Sum over time dimension to get returns
        sampled_returns1 = sampled_rewards1.sum(dim=1)  # [batch_size]
        sampled_returns2 = sampled_rewards2.sum(dim=1)  # [batch_size]
        
        # Compute Bradley-Terry loss on sampled returns
        logits_sample = torch.clamp(
            beta * (sampled_returns1 - sampled_returns2), min=-50.0, max=50.0
        )
        pred_probs_sample = torch.sigmoid(logits_sample).squeeze()
        
        bce_sample = -(
            prefs * torch.log(pred_probs_sample + eps)
            + (1 - prefs) * torch.log(1 - pred_probs_sample + eps)
        )
        loss_samples += torch.mean(bce_sample)
    
    # Average over samples
    loss_samples = loss_samples / num_samples
    
    # Combine losses according to Equation 7
    total_loss = loss_mean + lambda_weight * loss_samples
    
    return total_loss, loss_mean, loss_samples


def regularization_loss(reward_variances, eta=1.0):
    """
    Compute the regularization loss to prevent variance collapse as described in Equation 8.
    
    This forces the uncertainty level to maintain a minimum level eta by penalizing
    when the entropy of the Gaussian distribution falls below eta.
    
    Args:
        reward_variances: Predicted reward variances [batch_size, seq_len]
        eta: Minimum entropy level to maintain
        
    Returns:
        Regularization loss scalar
    """
    # Compute entropy of Gaussian distribution: h(N(μ, σ²)) = 0.5 * log(2πeσ²)
    # Since we have variances σ², the entropy is: 0.5 * log(2πe * σ²)
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * reward_variances + 1e-8)
    
    # Average entropy across time and batch dimensions
    avg_entropy = torch.mean(entropy)
    
    # Regularization loss: max(0, η - h(N(μ, σ²)))
    reg_loss = torch.clamp(eta - avg_entropy, min=0.0)
    
    return reg_loss


def distributional_reward_loss(
    reward_means1, reward_means2,
    reward_vars1, reward_vars2,
    preferences,
    lambda_weight=0.1,  # Reduced from 1.0 to give less weight to stochastic term
    alpha_reg=0.01,     # Reduced from 0.1 to give less weight to regularization
    eta=100.0,           # Reduced from 0.05 to allow lower entropy
    num_samples=5,
):
    """
    Complete loss function for distributional reward learning combining:
    1. Distributional Bradley-Terry loss (Equation 7)
    2. Regularization loss (Equation 8)
    
    Args:
        reward_means1, reward_means2: Reward means for both segments
        reward_vars1, reward_vars2: Reward variances for both segments
        preferences: Labels indicating which segment is preferred
        lambda_weight: Weight for the stochastic term
        alpha_reg: Weight for the regularization term
        eta: Minimum entropy level to maintain
        num_samples: Number of samples for stochastic term
        cost: DTW Cost for beta scaling
        alpha: Alpha parameter for beta scaling
        beta_max: Maximum beta value for scaling
    """
    # Get distributional BT loss + the deterministic and stochastic losses
    bt_loss, bt_loss_mean, bt_loss_samples = distributional_bradley_terry_loss(
        reward_means1, reward_means2,
        reward_vars1, reward_vars2,
        preferences,
        lambda_weight=lambda_weight,
        num_samples=num_samples
    )
    
    # Get regularization loss
    reg_loss = regularization_loss(
        torch.cat([reward_vars1, reward_vars2], dim=0),
        eta=eta
    )
    
    # Combine losses
    total_loss = bt_loss + alpha_reg * reg_loss
    
    return total_loss, bt_loss_mean, bt_loss_samples, reg_loss
