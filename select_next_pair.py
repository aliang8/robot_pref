import torch
import numpy as np
import argparse
from models.reward_models import EnsembleRewardModel
from utils.data import load_tensordict, segment_episodes
from utils.dataset import PreferenceDataset
import os

def compute_disagreement(logits):
    """Compute disagreement between ensemble members."""
    # logits shape: [num_models, batch_size]
    # Compute variance across models for each data point
    return logits.var(dim=0)

def compute_entropy(logits):
    """Compute entropy of the averaged predictions."""
    # Average predictions across models
    mean_probs = torch.sigmoid(logits).mean(0)
    # Compute entropy: -p*log(p) - (1-p)*log(1-p)
    entropy = -mean_probs * torch.log(mean_probs + 1e-8) - (1-mean_probs) * torch.log(1-mean_probs + 1e-8)
    return entropy

def select_next_pair(model_path, data_path, segment_pairs_path, segment_indices_path, method='disagreement'):
    """Select next pair using active learning."""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load data
    data = load_tensordict(data_path)
    segment_pairs = np.load(segment_pairs_path)
    segment_indices = np.load(segment_indices_path).reshape(-1, 2)
    
    # Initialize ensemble model
    state_dim = data['obs'].shape[1]
    action_dim = data['action'].shape[1]
    model = EnsembleRewardModel(
        state_dim=state_dim,
        action_dim=action_dim,
        num_models=5,
        use_images=True
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Get predictions for all pairs
    all_scores = []
    with torch.no_grad():
        for pair_idx, (seg_a, seg_b) in enumerate(segment_pairs):
            # Get segments
            start_a, end_a = segment_indices[seg_a]
            start_b, end_b = segment_indices[seg_b]
            
            # Get observations and actions
            obs_a = data['obs'][start_a:end_a].unsqueeze(0)
            obs_b = data['obs'][start_b:end_b].unsqueeze(0)
            act_a = data['action'][start_a:end_a].unsqueeze(0)
            act_b = data['action'][start_b:end_b].unsqueeze(0)
            
            if 'image' in data:
                img_a = data['image'][start_a:end_a].unsqueeze(0)
                img_b = data['image'][start_b:end_b].unsqueeze(0)
            else:
                img_a = img_b = None
            
            # Get predictions from ensemble
            logits_a = model(obs_a.to(device), act_a.to(device), img_a.to(device) if img_a is not None else None)
            logits_b = model(obs_b.to(device), act_b.to(device), img_b.to(device) if img_b is not None else None)
            
            # Compute preference logits
            pref_logits = logits_a - logits_b
            
            # Compute acquisition score
            if method == 'disagreement':
                score = compute_disagreement(pref_logits)
            else:  # entropy
                score = compute_entropy(pref_logits)
            
            all_scores.append(score.item())
    
    # Convert to numpy array
    all_scores = np.array(all_scores)
    
    # Select pair with highest score
    next_pair_idx = np.argmax(all_scores)
    
    # Save acquisition scores for visualization
    scores_path = os.path.join(os.path.dirname(data_path), f'{method}_scores.pkl')
    torch.save(all_scores, scores_path)
    
    return next_pair_idx, all_scores[next_pair_idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--segment_pairs_path', type=str, required=True)
    parser.add_argument('--segment_indices_path', type=str, required=True)
    parser.add_argument('--method', type=str, default='disagreement', choices=['disagreement', 'entropy'])
    args = parser.parse_args()
    
    next_idx, score = select_next_pair(
        args.model_path,
        args.data_path,
        args.segment_pairs_path,
        args.segment_indices_path,
        args.method
    )
    
    print(f'Selected pair index: {next_idx}')
    print(f'Acquisition score: {score}') 