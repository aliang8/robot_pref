import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import optim
from tqdm import tqdm

import wandb
from utils.loss import bradley_terry_loss

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)  # camera-ready formatting + latex in plots


def evaluate_model_on_test_set(model, test_loader, device):
    """Evaluate model performance on the test set.

    Args:
        model: Trained reward model (single model or ensemble)
        test_loader: DataLoader for test data
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0
    logpdf_values = []

    # Check if model is an ensemble
    is_ensemble = hasattr(model, "num_models") and model.num_models > 1

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            
            batch = {k: v.to(device) for k, v in batch.items()}
            obs1, actions1, obs2, actions2, pref = batch["obs1"], batch["actions1"], batch["obs2"], batch["actions2"], batch["preference"]
            images1, images2 = batch["images1"], batch["images2"]
            if images1 is not None:
                images1 = images1.float().to(device)
            if images2 is not None:
                images2 = images2.float().to(device)

            # Get reward predictions
            reward1 = model(obs1, actions1, images1)
            reward2 = model(obs2, actions2, images2)

            # Handle ensemble and non-ensemble models consistently with training
            if is_ensemble:
                # [B, N, T] -> [B, N] for ensemble
                return1 = reward1.sum(dim=-1)  # Sum over time dimension
                return2 = reward2.sum(dim=-1)
                # Replicate preferences for ensemble models
                ensemble_pref = pref.unsqueeze(0).repeat(model.num_models, 1)

                import ipdb

                ipdb.set_trace()
            else:
                # Standard non-ensemble model
                return1 = reward1.sum(dim=1)  # Sum over time dimension
                return2 = reward2.sum(dim=1)

            # Compute standard loss
            loss = bradley_terry_loss(return1, return2, pref)
            test_loss += loss.item()

            # Get predictions
            pred_pref = torch.where(
                return1 > return2, torch.ones_like(pref), torch.ones_like(pref) * 2
            )

            # Compute logpdf
            logp = -loss

            # Compute accuracy (same for both cases)
            correct = (pred_pref == pref).sum().item()
            test_acc += correct
            test_total += pref.size(0)

            # Save logpdf values
            logpdf_values.append(logp.mean().item())

    avg_test_loss = (
        test_loss / len(test_loader) if len(test_loader) > 0 else float("nan")
    )
    test_accuracy = test_acc / test_total if test_total > 0 else 0
    avg_logpdf = np.mean(logpdf_values) if logpdf_values else float("nan")

    print(
        f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Avg LogPDF: {avg_logpdf:.4f}"
    )
    print(
        f"Correctly predicted {test_acc} out of {test_total} preference pairs ({test_accuracy:.2%})"
    )

    return {
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "avg_logpdf": avg_logpdf,
        "num_test_samples": test_total,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    lr=1e-4,
    wandb_run=None,
    is_ensemble=False,
    output_path=None,
):
    """Unified training function for both single reward models and ensembles.

    Args:
        model: Model to train (either single model or ensemble)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (can be None if no validation is needed)
        device: Device to run training on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        wandb: Optional wandb instance for logging
        is_ensemble: Whether the model is an ensemble (for logging purposes only)
        output_path: Path to save the training curve plot (if None, uses default filename)

    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    max_grad_norm = 1.0  # For gradient clipping

    # Check if we have validation data
    has_validation = val_loader is not None and len(val_loader) > 0

    print(
        f"Training {'ensemble' if is_ensemble else 'reward'} model with gradient clipping at {max_grad_norm}..."
    )
    print(f"Using learning rate: {lr}")
    if not has_validation:
        print("No validation data provided - will train without validation")

    # Move model to device
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        # Use a progress bar with ETA
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} (Train)",
            leave=False,
            dynamic_ncols=True,
        )

        for _, batch in enumerate(progress_bar):
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            obs1, actions1, obs2, actions2, pref = batch["obs1"], batch["actions1"], batch["obs2"], batch["actions2"], batch["preference"]
            images1, images2 = batch["images1"], batch["images2"]
            if images1 is not None:
                images1 = images1.float().to(device)
            if images2 is not None:
                images2 = images2.float().to(device)
            optimizer.zero_grad(set_to_none=True)

            # Compute rewards directly using the forward method
            reward1 = model(obs1, actions1, images1)
            reward2 = model(obs2, actions2, images2)

            if is_ensemble:
                # [B, N, T] -> [B, N]
                return1 = reward1.sum(dim=-1)
                return2 = reward2.sum(dim=-1)
                pref = pref.unsqueeze(0).repeat(model.num_models, 1)
                cost = cost.unsqueeze(0).repeat(model.num_models, 1) if cost is not None else None
            else:
                return1 = reward1.sum(dim=1)
                return2 = reward2.sum(dim=1)

            # Bradley-Terry loss already applies mean over batch dimension
            # For ensemble models, we get one loss per model
            loss = bradley_terry_loss(return1, return2, pref, cost=cost)

            # For ensemble, take mean across models
            batch_loss = loss.mean() if loss.dim() > 0 else loss
            batch_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            train_loss += batch_loss.item()
            progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

            if wandb_run:
                wandb_run.log({"train/loss": batch_loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0

        if len(val_loader) > 0:
            val_progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} (Val)",
                leave=False,
                dynamic_ncols=True,
            )

            with torch.no_grad():
                for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(
                    val_progress
                ):
                    obs1, actions1, obs2, actions2, pref = (
                        obs1.to(device),
                        actions1.to(device),
                        obs2.to(device),
                        actions2.to(device),
                        pref.to(device),
                    )

                    # Compute rewards directly
                    reward1 = model(obs1, actions1)
                    reward2 = model(obs2, actions2)

                    if is_ensemble:
                        # [B, N, T] -> [B, N]
                        return1 = reward1.sum(dim=-1)
                        return2 = reward2.sum(dim=-1)
                        pref = pref.unsqueeze(0).repeat(model.num_models, 1)
                    else:
                        return1 = reward1.sum(dim=1)
                        return2 = reward2.sum(dim=1)

                    # Bradley-Terry loss already applies mean over batch dimension
                    batch_loss = bradley_terry_loss(return1, return2, pref)

                    # For ensemble, take mean across models
                    batch_loss = (
                        batch_loss.mean() if batch_loss.dim() > 0 else batch_loss
                    )

                    val_loss += batch_loss.item()
                    val_progress.set_postfix({"loss": f"{batch_loss.item():.4f}"})

            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
        val_losses.append(avg_val_loss)

        if wandb_run:
            wandb_run.log({"val/loss": avg_val_loss})

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

    # Plot training curve
    if train_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        if val_losses:
            plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        # remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # remove border legend
        plt.legend(frameon=False, loc="upper right", handlelength=1, fontsize=12)

        # Use the provided output path or default
        plot_path = output_path if output_path else "reward_model_training.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Training curve saved to {plot_path}")

        # Log the plot to wandb
        if wandb_run:
            wandb_run.log({"training_curve": wandb.Image(plot_path)})

        plt.close()

    return model, train_losses, val_losses
