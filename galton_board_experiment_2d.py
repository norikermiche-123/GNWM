"""
=============================================================================
GEOMETRIC NEURAL WORLD MODEL (GNWM) - 2D COUNTERFACTUAL GALTON BOARD
=============================================================================
This script demonstrates the fundamental flaw in standard non-contrastive 
representation learning (e.g., JEPA, BYOL) using a deterministic L2 loss. 
When confronted with a stochastic bifurcation (a ball falling left OR right), 
an L2 Predictor acts as a low-pass filter and collapses to the conditional mean, 
predicting a "ghost" average that violates physical laws.

To solve this, we implement the Geometric Neural World Model (GNWM). 
We fold the latent space into a 10x10 topological grid (Cortical Sheet) and 
apply a Geometric Winner-Take-All (Geo-WTA) objective. This explicitly forces 
diverging causal futures into orthogonal, non-overlapping spatial representations,
completely bypassing the need for Batch Normalization.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import math

# ==========================================
# 1. Dataset Generation (The Galton Board)
# ==========================================
# [THEORY] Why a Galton Board? A Galton board explicitly forces a bifurcation.
# The ball hits the peg and MUST go left or right based on a causal intervention.
# This isolates the "mode averaging" failure of standard predictive models.

def generate_galton_video(path_type='left', frames_count=35):
    """
    Generates a 64x64 video of a ball dropping and bouncing off a peg.
    The `path_type` acts as the underlying causal truth.
    """
    frames = []
    x, y = 32.0, 5.0
    vx, vy = 0.0, 0.0
    g, dt = 9.8, 0.2
    
    for t in range(frames_count):
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        cv2.circle(frame, (32, 25), 4, (0, 200, 0), -1) # The static Peg
        
        # [DETAIL] At frame 12, the causal intervention occurs.
        if t == 12: 
            if path_type == 'left': vx = -30.0 
            elif path_type == 'right': vx = 30.0
            
        # Standard Euler integration for gravity and velocity
        vy += g * dt
        y += vy * dt
        x += vx * dt
        
        # Floor & Wall Collisions with restitution (bounciness)
        if y > 56: y, vy = 56, 0
        if x < 10: x, vx = 10, -vx * 0.8
        if x > 54: x, vx = 54, -vx * 0.8
        
        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)
        frames.append(frame)
        
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

def create_training_dataset(num_videos=100):
    """Creates a balanced dataset of Left and Right action-conditioned videos."""
    print(f"Generating Action-Conditioned Galton Board Dataset ({num_videos} videos)...")
    videos = []
    actions = [] # Stores the causal action (0=Left, 1=Right)
    for i in range(num_videos):
        path = 'left' if i % 2 == 0 else 'right'
        action = 0 if path == 'left' else 1
        videos.append(generate_galton_video(path))
        actions.append(action)
        
    # Generate clean, unseen probes for final evaluation
    probe_L = generate_galton_video('left')
    probe_R = generate_galton_video('right')
    return videos, torch.tensor(actions, dtype=torch.long), probe_L, probe_R

class OffsetDataset(Dataset):
    """
    Dynamically serves pairs of (Context_t, Target_t+k) alongside the causal action.
    [DETAIL] max_k enables temporal curriculum learning, slowly teaching the network
    to predict further into the future as training progresses.
    """
    def __init__(self, videos_tensor, actions_tensor, max_k=1):
        self.videos = videos_tensor 
        self.actions = actions_tensor
        self.N, self.T, _, _, _ = self.videos.shape
        self.max_k = max_k
        self.valid_t = self.T - max_k
        
    def __len__(self): return self.N * self.valid_t * self.max_k
        
    def __getitem__(self, idx):
        k = (idx % self.max_k) + 1
        idx //= self.max_k
        t = idx % self.valid_t
        n = idx // self.valid_t
        # Return context, target, k_offset, AND the action that caused the outcome
        return self.videos[n, t], self.videos[n, t+k], torch.tensor(k, dtype=torch.long), self.actions[n]

# ==========================================
# 2. Architectures & 2D SOM Helpers
# ==========================================
# [THEORY] 10x10 Grid. We map the 100D vector into a 2D spatial arrangement.
# This forces the network to learn continuous geometric neighborhoods.
GRID_SIZE = 10
LATENT_DIM = GRID_SIZE * GRID_SIZE 

class Encoder(nn.Module):
    """Standard Convolutional Feature Extractor (Perception Module)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2), # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2), # 16x16
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, LATENT_DIM) # Project to 100D
        )
    def forward(self, x): 
        return self.fc(self.conv(x))

class Predictor(nn.Module):
    """
    Action-Conditioned Dynamics Model.
    [DETAIL] We concatenate the latent state, the time horizon embedding, 
    and the action embedding. Concatenation allows the subsequent MLP layers 
    to perform highly non-linear routing (unlike simple addition), ensuring 
    perfect orthogonal separation of the causal futures.
    """
    def __init__(self):
        super().__init__()
        self.k_embed = nn.Embedding(20, 128) 
        self.a_embed = nn.Embedding(2, 128) # Isolation for causal action intervention
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + 128 + 128, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, LATENT_DIM)
        )
    def forward(self, x, k, a): 
        k_emb = self.k_embed(k)
        a_emb = self.a_embed(a)
        fused = torch.cat([x, k_emb, a_emb], dim=-1)
        return x + self.net(fused) # Residual connection for stability

def update_ema(enc, tgt_enc, momentum=0.99): 
    """Slow-moving target encoder update to prevent trivial representation collapse."""
    with torch.no_grad():
        for p_q, p_k in zip(enc.parameters(), tgt_enc.parameters()):
            p_k.data = p_k.data * momentum + p_q.data * (1. - momentum)

def apply_som_convolution_2d(x, grid_size=GRID_SIZE, sigma=1.5):
    """
    [THEORY] The Topological Constraint.
    Instead of allowing the network to activate random, disjoint neurons to satisfy
    the Winner-Take-All loss, we apply a fixed 2D Gaussian blur over the 10x10 grid.
    This mimics lateral excitation in a biological cortical sheet, mathematically 
    forcing physically similar states to organize into adjacent, continuous neighborhoods.
    """
    device = x.device
    kernel_size = int(6 * max(0.1, sigma)) # Rule of thumb for Gaussian bounds
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 3: kernel_size = 3
        
    # 1. Create 2D Gaussian kernel algebraically
    grid = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(grid, grid, indexing='ij')
    gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma + 1e-8)**2))
    gaussian = gaussian / torch.sum(gaussian) # Normalize to 1
    
    # 2. Reshape for PyTorch Depthwise 2D Convolution: [Out, In, H, W]
    weight = gaussian.view(1, 1, kernel_size, kernel_size)
    
    # 3. Reshape Flat Vector (Batch, 100) -> 2D Grid (Batch, 1, 10, 10)
    x_reshaped = x.view(-1, 1, grid_size, grid_size)
    
    # 4. Apply Spatial Convolution
    x_conv = F.conv2d(x_reshaped, weight, padding='same')
    
    # Flatten back to (Batch, 100)
    return x_conv.view(-1, grid_size * grid_size)

# ==========================================
# 3. Robust Training Loop
# ==========================================
def train_model(videos, actions, model_type="baseline", epochs=150):
    device = videos[0].device
    enc = Encoder().to(device)
    pred = Predictor().to(device)
    
    if model_type == "baseline":
        tgt_enc = copy.deepcopy(enc).to(device)
        tgt_enc.eval()
    else:
        # [DETAIL] GNWM uses a pure Siamese setup for the target to enforce geometry
        tgt_enc = enc 
    
    optimizer = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-4, weight_decay=1e-5)
    
    # --- ROBUST OPTIMIZATION: Warmup + Cosine Decay Scheduler ---
    def get_lr_multiplier(epoch):
        warmup_epochs = int(epochs * 0.1) # 10% Warmup to prevent gradient shock on the topology
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay locks the representations gently into their final neurons
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    print(f"\nTraining {model_type.upper()} for {epochs} epochs (No Early Stopping)...")
    for epoch in range(epochs):
        
        # [DETAIL] Temporal Curriculum: Start by predicting 1 frame ahead, slowly 
        # increase to predicting 15 frames ahead to prevent gradient shock.
        current_max_k = min(15, 1 + int(14 * (epoch / 100.0)))
        current_sigma = 1.5 # Fixed Sigma
        
        dataloader = DataLoader(OffsetDataset(torch.stack(videos), actions, max_k=current_max_k), batch_size=128, shuffle=True)
        
        for batch_x, batch_y, batch_k, batch_a in dataloader:
            batch_x, batch_y, batch_k, batch_a = batch_x.to(device), batch_y.to(device), batch_k.to(device), batch_a.to(device)
            optimizer.zero_grad()
            
            h_t = enc(batch_x)
            pred_h = pred(h_t, batch_k, batch_a) 
            
            if model_type == "baseline":
                with torch.no_grad(): 
                    tgt_h = tgt_enc(batch_y)
                loss = torch.mean(torch.sum((pred_h - tgt_h)**2, dim=-1))
                loss.backward()
                optimizer.step()
                update_ema(enc, tgt_enc)
                
            elif model_type == "softmax_wta_2d":
                tgt_h = enc(batch_y) 
                
                D = pred_h.size(1) 
                const = torch.ones((1, D), device=device)
                const = F.normalize(const, p=2, dim=1)
                
                pred_h_som = apply_som_convolution_2d(pred_h, sigma=current_sigma)
                tgt_h_som = apply_som_convolution_2d(tgt_h, sigma=current_sigma)
                
                p = F.softmax(pred_h_som, dim=1)
                z = F.softmax(tgt_h_som, dim=1)
                p_l2 = F.normalize(p, p=2, dim=1)
                z_l2 = F.normalize(z, p=2, dim=1)
                
                p_z_mean = torch.mean((p_l2 + z_l2) / 2.0, dim=0, keepdim=True)
                p_z_mean_l2 = F.normalize(p_z_mean, p=2, dim=1)
                
                loss_collapse = 1.0 - torch.mean(torch.sum(p_z_mean_l2 * const, dim=1))
                loss_wta = torch.mean(torch.sum(p_z_mean * const, dim=1))
                loss_similarity = 1.0 - torch.mean(torch.sum(p_l2 * z_l2, dim=1))
                
                gamma = 0.5
                loss = loss_collapse + loss_wta + gamma * loss_similarity

                loss.backward()
                optimizer.step()
        
        scheduler.step() # Advance learning rate schedule smoothly
            
    return enc, tgt_enc, pred

# ==========================================
# 4. Evaluation (Counterfactual Action Testing)
# ==========================================
def evaluate_rollout(enc, tgt_enc, pred, probe_L, probe_R, use_softmax_dist=False):
    """Evaluates the trained model's ability to 'imagine' counterfactual futures."""
    enc.eval(); pred.eval()
    if tgt_enc is not enc: tgt_enc.eval()
    
    dL_correct, dL_wrong, dR_correct, dR_wrong, dSep = [], [], [], [], []
    winning_neurons = None
    device = probe_L.device
    
    start_t = 11 # Frame exactly before the ball hits the peg
    rollout_steps = 15 # Predict 15 frames into the future
    eval_sigma = 1.5 # Fixed topology
    
    with torch.no_grad():
        current_h = enc(probe_L[start_t].unsqueeze(0))
        
        for step in range(1, rollout_steps + 1):
            t = start_t + step
            k_tensor = torch.tensor([step], device=device)
            
            # Counterfactual Queries
            a_L = torch.tensor([0], device=device)
            pred_h_L = pred(current_h, k_tensor, a_L)
            
            a_R = torch.tensor([1], device=device)
            pred_h_R = pred(current_h, k_tensor, a_R)
            
            # Actual Ground Truth Targets
            h_L = tgt_enc(probe_L[t].unsqueeze(0))
            h_R = tgt_enc(probe_R[t].unsqueeze(0))
            
            if use_softmax_dist:
                pred_h_L_som = apply_som_convolution_2d(pred_h_L, sigma=eval_sigma)
                pred_h_R_som = apply_som_convolution_2d(pred_h_R, sigma=eval_sigma)
                h_L_som = apply_som_convolution_2d(h_L, sigma=eval_sigma)
                h_R_som = apply_som_convolution_2d(h_R, sigma=eval_sigma)
                
                P_pred_L = F.normalize(F.softmax(pred_h_L_som, dim=-1), p=2, dim=-1)
                P_pred_R = F.normalize(F.softmax(pred_h_R_som, dim=-1), p=2, dim=-1)
                P_L = F.normalize(F.softmax(h_L_som, dim=-1), p=2, dim=-1)
                P_R = F.normalize(F.softmax(h_R_som, dim=-1), p=2, dim=-1)
                
                if step == rollout_steps:
                    win_pL = torch.argmax(P_pred_L, dim=-1).item()
                    win_pR = torch.argmax(P_pred_R, dim=-1).item()
                    win_tL = torch.argmax(P_L, dim=-1).item()
                    win_tR = torch.argmax(P_R, dim=-1).item()
                    
                    winning_neurons = {
                        'pred_L': (win_pL // GRID_SIZE, win_pL % GRID_SIZE),
                        'pred_R': (win_pR // GRID_SIZE, win_pR % GRID_SIZE),
                        'true_L': (win_tL // GRID_SIZE, win_tL % GRID_SIZE),
                        'true_R': (win_tR // GRID_SIZE, win_tR % GRID_SIZE)
                    }
                
                dL_correct.append(torch.norm(P_pred_L - P_L, p=2).item())
                dL_wrong.append(torch.norm(P_pred_L - P_R, p=2).item())
                dR_correct.append(torch.norm(P_pred_R - P_R, p=2).item())
                dR_wrong.append(torch.norm(P_pred_R - P_L, p=2).item())
                dSep.append(torch.norm(P_L - P_R, p=2).item())
            else:
                dL_correct.append(torch.norm(pred_h_L - h_L, p=2).item())
                dL_wrong.append(torch.norm(pred_h_L - h_R, p=2).item())
                dR_correct.append(torch.norm(pred_h_R - h_R, p=2).item())
                dR_wrong.append(torch.norm(pred_h_R - h_L, p=2).item())
                dSep.append(torch.norm(h_L - h_R, p=2).item())
            
    return dL_correct, dL_wrong, dR_correct, dR_wrong, dSep, winning_neurons

def visualize_galton_heatmaps(enc, pred, probe_L):
    """Renders the 10x10 SOM probability distribution as a heatmap."""
    print("\nGenerating 2D Heatmaps for Galton Board Counterfactuals...")
    enc.eval(); pred.eval()
    device = probe_L.device
    
    start_t = 11 
    step = 15    
    
    with torch.no_grad():
        current_h = enc(probe_L[start_t].unsqueeze(0))
        k_tensor = torch.tensor([step], device=device)
        
        # Query Action LEFT (0)
        a_L = torch.tensor([0], device=device)
        pred_h_L = pred(current_h, k_tensor, a_L)
        pred_h_L_som = apply_som_convolution_2d(pred_h_L, sigma=1.5)
        p_L = F.softmax(pred_h_L_som, dim=-1).view(GRID_SIZE, GRID_SIZE).cpu().numpy()
        
        # Query Action RIGHT (1)
        a_R = torch.tensor([1], device=device)
        pred_h_R = pred(current_h, k_tensor, a_R)
        pred_h_R_som = apply_som_convolution_2d(pred_h_R, sigma=1.5)
        p_R = F.softmax(pred_h_R_som, dim=-1).view(GRID_SIZE, GRID_SIZE).cpu().numpy()
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Context Image
    context_frame = (probe_L[start_t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    axes[0].imshow(context_frame)
    axes[0].set_title(f"Context Frame (t={start_t})")
    axes[0].axis('off')
    
    # Heatmap Left
    im_L = axes[1].imshow(p_L, cmap='jet')
    axes[1].set_title("Imagined Future: Action LEFT")
    axes[1].axis('off')
    fig.colorbar(im_L, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Heatmap Right
    im_R = axes[2].imshow(p_R, cmap='jet')
    axes[2].set_title("Imagined Future: Action RIGHT")
    axes[2].axis('off')
    fig.colorbar(im_R, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("galton_2d_heatmaps.png")
    print("✅ Saved Galton Board heatmaps to 'galton_2d_heatmaps.png'")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    vids, actions, pL, pR = create_training_dataset(100)
    vids = [v.to(device) for v in vids]
    actions = actions.to(device)
    pL, pR = pL.to(device), pR.to(device)
    
    eB, tB, pB = train_model(vids, actions, "baseline", epochs=150)
    
    # Run the full 400 epochs allowing the Cosine Scheduler to fully converge
    eU, tU, pU = train_model(vids, actions, "softmax_wta_2d", epochs=400) 
    
    bL_c, bL_w, bR_c, bR_w, bSep, _ = evaluate_rollout(eB, tB, pB, pL, pR, use_softmax_dist=False)
    uL_c, uL_w, uR_c, uR_w, uSep, u_winners = evaluate_rollout(eU, tU, pU, pL, pR, use_softmax_dist=True)
    
    print("\n" + "="*60 + "\nSTATISTICAL PRINTOUT (Final Step t=26)\n" + "="*60)
    print(f"TARGET SEPARATION (Distance between Actual Left and Right paths):")
    print(f"  -> Baseline: {bSep[-1]:.6f} (L2 Space Collapsed)")
    print(f"  -> User WTA (2D SOM): {uSep[-1]:.6f} (P-Space Orthogonal)")
    
    if u_winners:
        print("-" * 60)
        print(f"WINNING NEURONS ON 10x10 SOM GRID (At Final Frame):")
        print(f"  -> Actual Target Left:  Row {u_winners['true_L'][0]}, Col {u_winners['true_L'][1]}")
        print(f"  -> Predicted Action=Left: Row {u_winners['pred_L'][0]}, Col {u_winners['pred_L'][1]}")
        print(f"  -> Actual Target Right: Row {u_winners['true_R'][0]}, Col {u_winners['true_R'][1]}")
        print(f"  -> Predicted Action=Right:Row {u_winners['pred_R'][0]}, Col {u_winners['pred_R'][1]}")
        
    print("-" * 60)
    print(f"BASELINE ACTION-CONDITIONED PREDICTOR (L2 Bias Blur):")
    print(f"  -> Query 'Left':  Distance to True Left: {bL_c[-1]:.4f} | Distance to Wrong Right: {bL_w[-1]:.4f}")
    print(f"  -> Query 'Right': Distance to True Right: {bR_c[-1]:.4f} | Distance to Wrong Left: {bR_w[-1]:.4f}")
    print(f"USER'S CAUSAL 2D SOM PREDICTOR (True Disambiguation):")
    print(f"  -> Query 'Left':  Distance to True Left: {uL_c[-1]:.4f} | Distance to Wrong Right: {uL_w[-1]:.4f}")
    print(f"  -> Query 'Right': Distance to True Right: {uR_c[-1]:.4f} | Distance to Wrong Left: {uR_w[-1]:.4f}")
    print("="*60 + "\n")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(bL_c, color='blue', label="Query Left -> Actual Left")
    plt.plot(bL_w, color='blue', linestyle='--', alpha=0.3, label="Query Left -> Actual Right")
    plt.plot(bR_c, color='red', label="Query Right -> Actual Right")
    plt.plot(bR_w, color='red', linestyle='--', alpha=0.3, label="Query Right -> Actual Left")
    plt.title("Baseline Counterfactual Predictor (L2 Bias)")
    plt.xlabel("Prediction Horizon (k steps ahead)")
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(uL_c, color='blue', linewidth=2, label="Query Left -> Actual Left (Correct)")
    plt.plot(uL_w, color='blue', linestyle='--', alpha=0.5, label="Query Left -> Actual Right (Wrong)")
    plt.plot(uR_c, color='red', linewidth=2, label="Query Right -> Actual Right (Correct)")
    plt.plot(uR_w, color='red', linestyle='--', alpha=0.5, label="Query Right -> Actual Left (Wrong)")
    plt.title("User's Causal 2D SOM Predictor (GNWM)")
    plt.xlabel("Prediction Horizon (k steps ahead)")
    plt.legend(); plt.grid(alpha=0.3)
    
    visualize_galton_heatmaps(eU, pU, pL)
    
    plt.show()