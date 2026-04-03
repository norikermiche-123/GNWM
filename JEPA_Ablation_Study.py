import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

# ==========================================
# 0. On-the-Fly Dataset Generation
# ==========================================
def create_toy_dataset():
    print("Generating Toy Physics Dataset (64x64)...")
    width, height, fps = 64, 64, 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    def make_video(filename, reverse_g_step=None):
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        y, vy, dt, g = 10.0, 0.0, 0.2, 9.8
        for t in range(80):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            current_g = -g if (reverse_g_step and t >= reverse_g_step) else g
            vy += current_g * dt
            y += vy * dt
            if y > 54: y, vy = 54, -vy * 0.8
            if y < 10: y, vy = 10, -vy * 0.8
            cv2.circle(frame, (32, int(y)), 6, (255, 0, 0), -1)
            out.write(frame)
        out.release()
        
    make_video("normal.mp4")
    make_video("gravity_reversal.mp4", reverse_g_step=30)
    print("✅ Dataset generated.\n")

def load_video(video_path, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    return torch.stack(frames).to(device)

# ==========================================
# 1. Architectural Variants
# ==========================================
class Encoder(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x): return self.net(x)

def update_ema(encoder, target_encoder, momentum=0.99):
    with torch.no_grad():
        for p_q, p_k in zip(encoder.parameters(), target_encoder.parameters()):
            p_k.data = p_k.data * momentum + p_q.data * (1. - momentum)

# ==========================================
# 2. Training Logic (The 3 Paradigms)
# ==========================================
def train_model(video_tensor, model_type="baseline", epochs=100):
    device = video_tensor.device
    use_bn = (model_type == "batch_norm")
    
    enc = Encoder(use_bn).to(device)
    tgt_enc = copy.deepcopy(enc).to(device)
    tgt_enc.eval()
    pred = Predictor(use_bn).to(device)
    
    optimizer = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    
    std_history = [] 
    
    for epoch in range(epochs):
        batch_size = 16
        epoch_std = 0
        valid_batches = 0
        
        for i in range(0, len(video_tensor) - 1, batch_size):
            x_t = video_tensor[i:i+batch_size]
            x_next = video_tensor[i+1:i+batch_size+1]
            
            # --- FIX: Ensure matched lengths for the final batch ---
            min_len = min(len(x_t), len(x_next))
            if min_len < 2: continue 
            
            x_t = x_t[:min_len]
            x_next = x_next[:min_len]
            
            optimizer.zero_grad()
            h_t = enc(x_t)
            pred_h = pred(h_t)
            
            with torch.no_grad():
                tgt_h = tgt_enc(x_next)
                
            # Track Latent Variance (Collapse metric)
            epoch_std += tgt_h.std(dim=0).mean().item()
            valid_batches += 1

            if model_type in ["baseline", "batch_norm"]:
                # Standard L2 Loss
                loss = torch.mean((pred_h - tgt_h)**2)
                
            elif model_type == "variance_penalty":
                # Explicitly enforces variance (Simulating the BN effect mathematically)
                # This tests your exact theory: pushing the representations apart by variance
                loss_pull = torch.mean((pred_h - tgt_h)**2)
                std_pred = torch.sqrt(pred_h.var(dim=0) + 1e-4)
                loss_var = torch.mean(torch.relu(1.0 - std_pred)) # Push std to 1.0
                loss = loss_pull + loss_var
            
            elif model_type == "diffusion_margin":
                # Local Latent Diffusion Loss (Energy Margin)
                loss_pull = torch.mean((pred_h - tgt_h)**2)
                
                # Push away from diffused negative
                noise = torch.randn_like(tgt_h) * 0.5 
                diffused_negative = tgt_h + noise
                
                dist_to_neg = torch.mean((pred_h - diffused_negative)**2, dim=-1)
                loss_push = torch.mean(torch.relu(1.0 - dist_to_neg))
                
                loss = loss_pull + 0.5 * loss_push

            loss.backward()
            optimizer.step()
            update_ema(enc, tgt_enc)
            
        std_history.append(epoch_std / max(1, valid_batches))
        
    return enc, tgt_enc, pred, std_history

# ==========================================
# 3. Execution & Plotting
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Ablation on: {device}")
    
    create_toy_dataset()
    tensor_normal = load_video("normal.mp4", device)
    tensor_gravity = load_video("gravity_reversal.mp4", device)

    print("\nTraining Model A: Baseline (No BN, No Negative Sampling)...")
    enc_A, tgt_A, pred_A, std_A = train_model(tensor_normal, "baseline")
    
    print("Training Model B: The Hack (Batch Normalization)...")
    enc_B, tgt_B, pred_B, std_B = train_model(tensor_normal, "batch_norm")
    
    print("Training Model C: Variance Penalty (Math equivalent of BN)...")
    enc_C, tgt_C, pred_C, std_C = train_model(tensor_normal, "variance_penalty")
    
    print("Training Model D: The Fix (Local Latent Diffusion)...")
    enc_D, tgt_D, pred_D, std_D = train_model(tensor_normal, "diffusion_margin")

    # --- Plotting 1: Dimensional Collapse ---
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(std_A, label="Model A (Baseline - Collapses)", color='red', linewidth=2)
    plt.plot(std_B, label="Model B (Batch Norm Hack)", color='gray', linestyle='--')
    plt.plot(std_C, label="Model C (Variance Penalty)", color='green', linestyle='-.')
    plt.plot(std_D, label="Model D (Latent Diffusion Fix)", color='blue', linewidth=2)
    plt.title("Proof of Dimensional Collapse (Latent Std Dev)")
    plt.xlabel("Epoch")
    plt.ylabel("Latent Standard Deviation")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- Evaluation: Probe A (Gravity Reversal) ---
    def evaluate(enc, pred, tgt_enc, video):
        enc.eval(); pred.eval()
        errors = []
        with torch.no_grad():
            for t in range(len(video) - 1):
                h_t = enc(video[t].unsqueeze(0))
                pred_h = pred(h_t)
                tgt_h = tgt_enc(video[t+1].unsqueeze(0))
                errors.append(torch.norm(pred_h - tgt_h, p=2).item())
        return errors

    err_B = evaluate(enc_B, pred_B, tgt_B, tensor_gravity)
    err_C = evaluate(enc_C, pred_C, tgt_C, tensor_gravity)
    err_D = evaluate(enc_D, pred_D, tgt_D, tensor_gravity)

    plt.subplot(1, 2, 2)
    plt.plot(err_B, label="Model B (Batch Norm) Error", color='gray', linestyle='--')
    plt.plot(err_C, label="Model C (Variance) Error", color='green', linestyle='-.')
    plt.plot(err_D, label="Model D (Diffusion Fix) Error", color='blue', linewidth=2)
    plt.axvline(x=30, color='red', linestyle=':', label="Gravity Reverses")
    plt.title("Probe A: Gravity Reversal Detection")
    plt.xlabel("Frame")
    plt.ylabel("Latent $L_2$ Error")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ablation_results.png")
    print("\n✅ Success! Check 'ablation_results.png'.")
    plt.show()