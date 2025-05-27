import matplotlib.pyplot as plt
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


def evaluate_model_on_test_set(model, test_loader, device, data=None, segment_start_end=None, num_vis=4):
    """Evaluate model performance on the test set.

    Args:
        model: Trained reward model (single model or ensemble)
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        num_vis: Number of top preferences to visualize as videos

    Returns:
        Dictionary containing evaluation metrics
    """
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0

    # For visualization: store the first num_vis examples
    vis_examples = []

    # Check if model is an ensemble
    is_ensemble = hasattr(model, "num_models") and model.num_models > 1

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
                # For visualization, just use the first model in the ensemble
                reward1_vis = reward1[0]
                reward2_vis = reward2[0]
            else:
                # Standard non-ensemble model
                return1 = reward1.sum(dim=1)  # Sum over time dimension
                return2 = reward2.sum(dim=1)
                reward1_vis = reward1
                reward2_vis = reward2

            # Compute standard loss
            loss = bradley_terry_loss(return1, return2, pref)
            test_loss += loss.item()

            # Get predictions
            pred_pref = torch.where(
                return1 > return2, 
                torch.ones_like(pref),  # First trajectory preferred
                torch.where(
                    return1 < return2,
                    torch.zeros_like(pref),  # Second trajectory preferred
                    torch.ones_like(pref) * 0.5  # Equal preference
                )
            )

            # Compute accuracy (same for both cases)
            # For equal preferences (0.5), we consider it correct if the model predicts either 0.5 or if the returns are very close
            correct = torch.where(
                pref == 0.5,
                (pred_pref == 0.5) | (torch.abs(return1 - return2) < 1e-3),  # Equal preference case
                pred_pref == pref  # Regular preference case
            ).sum().item()
            test_acc += correct
            test_total += pref.size(0)

            # Collect examples for visualization
            if len(vis_examples) < num_vis:
                # For each sample in the batch, up to num_vis
                for i in range(reward1_vis.shape[0]):
                    if len(vis_examples) >= num_vis:
                        break
                    # Get rewards over time as numpy arrays
                    r1 = reward1_vis[i].detach().cpu().numpy()
                    r2 = reward2_vis[i].detach().cpu().numpy()
                    # Get preference and prediction
                    gt_pref = float(pref[i].item())
                    pred = float(pred_pref[i].item())
                    # Get segment indices from the dataset
                    seg1 = test_loader.dataset.segment_pairs[batch_idx * test_loader.batch_size + i][0]
                    seg2 = test_loader.dataset.segment_pairs[batch_idx * test_loader.batch_size + i][1]
                    vis_examples.append({
                        "reward1": r1,
                        "reward2": r2,
                        "gt_pref": gt_pref,
                        "pred_pref": pred,
                        "seg1": seg1,
                        "seg2": seg2,
                    })

    avg_test_loss = (
        test_loss / len(test_loader) if len(test_loader) > 0 else float("nan")
    )
    test_accuracy = test_acc / test_total if test_total > 0 else 0

    print(
        f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}"
    )
    print(
        f"Correctly predicted {test_acc} out of {test_total} preference pairs ({test_accuracy:.2%})"
    )

    # --- Visualization and wandb logging ---
    if len(vis_examples) > 0 and wandb.run is not None and data is not None and segment_start_end is not None:
        videos = []
        for idx, ex in enumerate(vis_examples):
            r1 = ex["reward1"]
            r2 = ex["reward2"]
            gt_pref = ex["gt_pref"]
            pred_pref = ex["pred_pref"]
            seg1 = ex["seg1"]
            seg2 = ex["seg2"]

            # Create a matplotlib animation of the reward curves
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
            ax1.set_title(f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}")
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("Reward")
            ax1.grid(True)
            line1, = ax1.plot([], [], label="Traj 1", color="blue")
            line2, = ax1.plot([], [], label="Traj 2", color="red")
            ax1.legend(loc="upper right")

            # Set up image displays side by side
            ax2.set_title("Trajectory 1")
            ax2.axis('off')
            img1 = ax2.imshow(np.zeros((64, 64, 3)), animated=True)

            ax3.set_title("Trajectory 2")
            ax3.axis('off')
            img2 = ax3.imshow(np.zeros((64, 64, 3)), animated=True)

            max_len = max(len(r1), len(r2))
            ax1.set_xlim(0, max_len-1)
            min_y = min(np.min(r1), np.min(r2))
            max_y = max(np.max(r1), np.max(r2))
            ax1.set_ylim(min_y - 0.1 * abs(min_y), max_y + 0.1 * abs(max_y))

            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                img1.set_data(np.zeros((64, 64, 3)))
                img2.set_data(np.zeros((64, 64, 3)))
                return line1, line2, img1, img2

            def animate(t):
                x = np.arange(0, t+1)
                y1 = r1[:t+1] if t < len(r1) else r1
                y2 = r2[:t+1] if t < len(r2) else r2
                line1.set_data(x[:len(y1)], y1)
                line2.set_data(x[:len(y2)], y2)

                # Update images if available
                if "image" in data:
                    # Get the current frame indices for both trajectories
                    start1, end1 = segment_start_end[seg1]
                    start2, end2 = segment_start_end[seg2]
                    
                    # Get the current frame for each trajectory
                    if t < len(r1):
                        frame1 = data["image"][start1 + t].cpu().numpy()
                        img1.set_data(frame1)
                    if t < len(r2):
                        frame2 = data["image"][start2 + t].cpu().numpy()
                        img2.set_data(frame2)

                return line1, line2, img1, img2

            frames = max_len
            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=frames, interval=100, blit=True
            )

            # Add a text annotation at the end to indicate which trajectory is preferred
            def add_pref_annotation():
                if gt_pref == 1:
                    ax1.annotate("Preferred", xy=(len(r1)-1, r1[-1]), xytext=(len(r1)-1, r1[-1]+0.1),
                                color="blue", fontsize=12, arrowprops=dict(arrowstyle="->", color="blue"))
                elif gt_pref == 0:
                    ax1.annotate("Preferred", xy=(len(r2)-1, r2[-1]), xytext=(len(r2)-1, r2[-1]+0.1),
                                color="red", fontsize=12, arrowprops=dict(arrowstyle="->", color="red"))
                else:  # gt_pref == 0.5
                    ax1.annotate("Equal", xy=(len(r1)-1, (r1[-1] + r2[-1])/2), xytext=(len(r1)-1, (r1[-1] + r2[-1])/2 + 0.1),
                                color="green", fontsize=12, arrowprops=dict(arrowstyle="->", color="green"))
            # Save animation to buffer
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                ani.save(tmp.name, writer="ffmpeg", fps=10)
                # Read the temporary file into memory
                with open(tmp.name, 'rb') as f:
                    buf = io.BytesIO(f.read())
            # Clean up the temporary file
            import os
            os.unlink(tmp.name)
            
            plt.close(fig)
            buf.seek(0)
            # Add annotation as a final frame (not possible in animation, so just note in title)
            videos.append(wandb.Video(buf, caption=f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", format="mp4"))

        wandb.log({"test_set_reward_videos": videos})

    return {
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            obs1, actions1, obs2, actions2, pref, cost = batch["obs1"], batch["actions1"], batch["obs2"], batch["actions2"], batch["preference"], batch["cost"]
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
                for _, batch in enumerate(
                    val_progress
                ):
                    # Move data to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    obs1, actions1, obs2, actions2, pref, cost = batch["obs1"], batch["actions1"], batch["obs2"], batch["actions2"], batch["preference"], batch["cost"]
                    images1, images2 = batch["images1"], batch["images2"]
                    if images1 is not None:
                        images1 = images1.float().to(device)
                    if images2 is not None:
                        images2 = images2.float().to(device)

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
