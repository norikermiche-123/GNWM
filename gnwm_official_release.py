"""
=============================================================================
THE GLOBAL NEURAL WORLD MODEL (GNWM) - OFFICIAL RELEASE
=============================================================================
This repository contains the official implementation of the GNWM framework, 
capable of Drift-Free Topological Quantization and Action-Conditioned Planning.

Usage:
  python gnwm_official_release.py --paradigm passive
  python gnwm_official_release.py --paradigm active
  python gnwm_official_release.py --paradigm compositional
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import warnings

warnings.filterwarnings("ignore")

# Set global seeds for reliable CPU initialization
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. HYPERPARAMETERS & CONSTANTS
# ==========================================
GRID_SIZE = 15
LATENT_DIM = GRID_SIZE * GRID_SIZE  # 225
ACTION_DIM = 4 # Up, Right, Down, Left

# ==========================================
# 2. ENVIRONMENTS (Simulated 2D Physics)
# ==========================================
def generate_passive_video(frames_count=1000):
    frames = []
    x, y = 32.0, 32.0
    vx, vy = 2.5, 3.5 
    for _ in range(frames_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x += vx; y += vy
        if x > 56 or x < 8: vx *= -1; x += vx
        if y > 56 or y < 8: vy *= -1; y += vy
        cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 0), -1) 
        frames.append(frame)
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

def generate_active_video(frames_count=1000):
    frames, actions = [], []
    x, y = 32.0, 32.0
    for _ in range(frames_count):
        a = np.random.randint(0, 4)
        if a == 0: vx, vy = 0.0, -3.0   
        elif a == 1: vx, vy = 3.0, 0.0  
        elif a == 2: vx, vy = 0.0, 3.0  
        elif a == 3: vx, vy = -3.0, 0.0 
        x += vx; y += vy
        
        x = max(8.0, min(56.0, x)); y = max(8.0, min(56.0, y))
        
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 0), -1)
        frames.append(frame); actions.append(a)
        
    vids = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    acts = torch.tensor(actions, dtype=torch.long)
    return vids, acts

def generate_compositional_video(frames_count=1000):
    frames = []
    x1, y1, vx1, vy1 = 20.0, 20.0, 3.0, 2.0
    x2, y2, vx2, vy2 = 40.0, 40.0, -2.0, -3.0
    for _ in range(frames_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x1 += vx1; y1 += vy1
        if x1 > 56 or x1 < 8: vx1 *= -1; x1 += vx1
        if y1 > 56 or y1 < 8: vy1 *= -1; y1 += vy1
        
        x2 += vx2; y2 += vy2
        if x2 > 56 or x2 < 8: vx2 *= -1; x2 += vx2
        if y2 > 56 or y2 < 8: vy2 *= -1; y2 += vy2
        
        cv2.circle(frame, (int(x1), int(y1)), 6, (255, 0, 0), -1) 
        cv2.circle(frame, (int(x2), int(y2)), 6, (0, 0, 255), -1) 
        frames.append(frame)
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

class GNWMDataset(Dataset):
    def __init__(self, frames, actions=None):
        self.frames = frames
        self.actions = actions
    def __len__(self): return len(self.frames) - 1
    def __getitem__(self, idx): 
        if self.actions is not None:
            # CRITICAL FIX: Aligning causal arrow of time. Predicting frame[idx+1] requires action[idx+1]
            return self.frames[idx], self.actions[idx+1], self.frames[idx+1]
        return self.frames[idx], self.frames[idx+1]

# ==========================================
# 3. CORE GNWM MODULES
# ==========================================
class RetinotopicEncoder(nn.Module):
    def __init__(self, channels_out=1):
        super().__init__()
        self.channels_out = channels_out
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, channels_out, 2, 1, 0)
        )
        self.spatial_bias = nn.Parameter(torch.zeros(1, channels_out, LATENT_DIM))
        
    def forward(self, x): 
        out = self.net(x).view(-1, self.channels_out, LATENT_DIM)
        return out + self.spatial_bias

class SpatialPredictor(nn.Module):
    def __init__(self, action_conditioned=False, channels_in=1):
        super().__init__()
        self.action_conditioned = action_conditioned
        self.channels_in = channels_in
        
        in_dim = channels_in + (16 if action_conditioned else 0)
        if action_conditioned:
            self.action_emb = nn.Embedding(ACTION_DIM, 16) 
            
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, channels_in, 3, padding=1)
        )
        
    def forward(self, z_logits_t, a_t=None):
        B = z_logits_t.shape[0]
        x = z_logits_t.view(B, self.channels_in, GRID_SIZE, GRID_SIZE)
        
        if self.action_conditioned and a_t is not None:
            a_emb = self.action_emb(a_t).view(B, 16, 1, 1).expand(-1, -1, GRID_SIZE, GRID_SIZE)
            x = torch.cat([x, a_emb], dim=1)
            
        out = self.net(x)
        return out.view(B, self.channels_in, LATENT_DIM)

class JointSpatialPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
    def forward(self, z_logits_A, z_logits_B):
        B = z_logits_A.shape[0]
        z_spatial_A = z_logits_A.view(B, 1, GRID_SIZE, GRID_SIZE)
        z_spatial_B = z_logits_B.view(B, 1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([z_spatial_A, z_spatial_B], dim=1)
        out = self.net(x)
        return out.view(B, 2, LATENT_DIM)

def apply_som_convolution(logits, sigma=1.5):
    device = logits.device
    k = max(3, int(6 * max(0.1, sigma)) | 1)
    grid = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(grid, grid, indexing='ij')
    gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma + 1e-8)**2))
    gaussian = gaussian / torch.sum(gaussian)
    w = gaussian.view(1, 1, k, k).detach()
    x_conv = F.conv2d(logits.view(-1, 1, GRID_SIZE, GRID_SIZE), w, padding='same')
    return x_conv.view(-1, LATENT_DIM)

# ==========================================
# 4. UNIFIED GNWM LOSS
# ==========================================
def create_grid_coordinates(device):
    grid_1d = torch.linspace(-1, 1, GRID_SIZE, device=device)
    y_grid, x_grid = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    return torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

def compute_gnwm_thermodynamics(z_logits_t, z_logits_next, p_logits_next, grid_coords, epoch, warmup_epochs=20):
    device = z_logits_t.device
    const = F.normalize(torch.ones((1, LATENT_DIM), device=device), p=2, dim=1)
    
    # SIMULATED ANNEALING
    progress = min(1.0, epoch / warmup_epochs)
    temp = 1.0 + (progress * 9.0)        
    noise_scale = 0.5 * (1.0 - progress) 
    
    z_t = z_logits_t + torch.randn_like(z_logits_t) * noise_scale
    z_next = z_logits_next + torch.randn_like(z_logits_next) * noise_scale
    
    z_prob_t = F.softmax(apply_som_convolution(z_t * temp), dim=1)
    z_prob_next = F.softmax(apply_som_convolution(z_next * temp), dim=1)
    p_prob_next = F.softmax(apply_som_convolution(p_logits_next * temp), dim=1)
    
    z_l2_t = F.normalize(z_prob_t, p=2, dim=1)
    z_l2_next = F.normalize(z_prob_next, p=2, dim=1)
    p_l2_next = F.normalize(p_prob_next, p=2, dim=1)
    
    mean_vec = torch.mean((z_l2_t + z_l2_next) / 2.0, dim=0, keepdim=True)
    mean_l2 = F.normalize(mean_vec, p=2, dim=1)
    L_collapse = 1.0 - torch.sum(mean_l2 * const) 
    L_wta = torch.sum(mean_vec * const)          
    
    expected_pos_t = torch.matmul(z_prob_t, grid_coords)
    std_pos = torch.sqrt(expected_pos_t.var(dim=0) + 1e-4)
    L_var = torch.mean(F.relu(0.433 - std_pos)) 
    
    pos_centered = expected_pos_t - expected_pos_t.mean(dim=0)
    cov_pos = (pos_centered.T @ pos_centered) / (len(expected_pos_t) - 1)
    L_cov = cov_pos[0, 1].pow(2)
    
    L_sim = 1.0 - torch.mean(torch.sum(p_l2_next * z_l2_next.detach(), dim=1))
    
    loss = 1.0 * (L_collapse + L_wta) + 0.5 * L_sim
    
    return loss, L_var, z_prob_t

# ==========================================
# 5. PARADIGM TRAINING LOOPS
# ==========================================
def train_model(paradigm, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Paradigm: {paradigm.upper()} on {device}")
    
    acts_tensor = None
    if paradigm == 'passive':
        vids = generate_passive_video(1000).to(device)
        enc = RetinotopicEncoder(channels_out=1).to(device)
        pred = SpatialPredictor(action_conditioned=False, channels_in=1).to(device)
        loader = DataLoader(GNWMDataset(vids), batch_size=64, shuffle=True)
        
    elif paradigm == 'active':
        vids, acts = generate_active_video(1000)
        vids, acts_tensor = vids.to(device), acts.to(device)
        enc = RetinotopicEncoder(channels_out=1).to(device)
        pred = SpatialPredictor(action_conditioned=True, channels_in=1).to(device)
        loader = DataLoader(GNWMDataset(vids, acts_tensor), batch_size=64, shuffle=True)
        
    elif paradigm == 'compositional':
        vids = generate_compositional_video(1000).to(device)
        enc = RetinotopicEncoder(channels_out=2).to(device) 
        pred = JointSpatialPredictor().to(device)
        loader = DataLoader(GNWMDataset(vids), batch_size=64, shuffle=True)
    
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(loader), 
        pct_start=0.15, anneal_strategy='cos'
    )
    grid_coords = create_grid_coordinates(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            
            if paradigm == 'active':
                x_t, a_t, x_next = batch
                z_logits_t = enc(x_t)[:, 0, :]
                z_logits_next = enc(x_next)[:, 0, :]
                p_logits_next = pred(z_logits_t, a_t)[:, 0, :]
                loss, var_loss, _ = compute_gnwm_thermodynamics(z_logits_t, z_logits_next, p_logits_next, grid_coords, epoch)
            
            elif paradigm == 'passive':
                x_t, x_next = batch
                z_logits_t = enc(x_t)[:, 0, :]
                z_logits_next = enc(x_next)[:, 0, :]
                p_logits_next = pred(z_logits_t)[:, 0, :]
                loss, var_loss, _ = compute_gnwm_thermodynamics(z_logits_t, z_logits_next, p_logits_next, grid_coords, epoch)
            
            elif paradigm == 'compositional':
                x_t, x_next = batch
                z_logits_t = enc(x_t) 
                z_logits_next = enc(x_next)
                
                z_logits_t_A = z_logits_t[:, 0, :]
                z_logits_t_B = z_logits_t[:, 1, :]
                p_logits_next = pred(z_logits_t_A, z_logits_t_B) 
                
                loss_A, var_A, _ = compute_gnwm_thermodynamics(z_logits_t_A, z_logits_next[:, 0, :], p_logits_next[:, 0, :], grid_coords, epoch)
                loss_B, var_B, _ = compute_gnwm_thermodynamics(z_logits_t_B, z_logits_next[:, 1, :], p_logits_next[:, 1, :], grid_coords, epoch)
                
                temp = 1.0 + (min(1.0, epoch / 20.0) * 9.0)
                z_prob_t_A = F.softmax(apply_som_convolution(z_logits_t_A * temp), dim=1)
                z_prob_t_B = F.softmax(apply_som_convolution(z_logits_t_B * temp), dim=1)
                pos_A = torch.matmul(z_prob_t_A, grid_coords)
                pos_B = torch.matmul(z_prob_t_B, grid_coords)
                L_repel = torch.mean(F.cosine_similarity(pos_A, pos_B)) * 0.5
                
                loss = loss_A + loss_B + L_repel
                var_loss = (var_A + var_B) / 2.0

            loss.backward()
            opt.step()
            scheduler.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(loader):.4f} | Var Penalty: {var_loss.item():.4f}")
            
    print("✅ Training Complete.")
    return enc, pred, vids, acts_tensor

# ==========================================
# 6. EVALUATION & MATPLOTLIB PLOTTING
# ==========================================
def run_evaluations(paradigm, enc, pred, vids, acts):
    enc.eval(); pred.eval()
    device = vids.device
    print("\nGenerating Plots...")
    eval_temp = 10.0 

    def convert_to_white_bg(avg_img):
        """Converts additive light (on black) to subtractive ink (on white)."""
        r, g, b = avg_img[:,:,0], avg_img[:,:,1], avg_img[:,:,2]
        new_r = np.clip(255 - g - b, 0, 255)
        new_g = np.clip(255 - r - b, 0, 255)
        new_b = np.clip(255 - r - g, 0, 255)
        return np.stack([new_r, new_g, new_b], axis=-1)

    with torch.no_grad():
        if paradigm == 'passive':
            logits_all = enc(vids)[:, 0, :]
            prob = F.softmax(apply_som_convolution(logits_all * eval_temp, 1.5), dim=1)
            winners = torch.argmax(prob, dim=1).cpu().numpy()
            
            cell = 64
            map_image = np.full((GRID_SIZE * cell, GRID_SIZE * cell, 3), 255, dtype=np.uint8)
            active = 0
            for idx in range(LATENT_DIM):
                frames = vids[winners == idx].cpu().numpy()
                if len(frames) > 0:
                    active += 1
                    avg = np.mean(frames, axis=0).transpose(1, 2, 0) * 255.0
                    
                    # Apply color inversion
                    avg = convert_to_white_bg(avg)
                    
                    r_pos, c_pos = idx // GRID_SIZE, idx % GRID_SIZE
                    map_image[r_pos*cell:(r_pos+1)*cell, c_pos*cell:(c_pos+1)*cell] = avg.astype(np.uint8)
                    cv2.rectangle(map_image, (c_pos*cell, r_pos*cell), ((c_pos+1)*cell, (r_pos+1)*cell), (200, 200, 200), 2)
            
            print(f"[{paradigm.upper()}] Active Neurons: {active} / {LATENT_DIM}")
            plt.figure(figsize=(8, 8))
            plt.imshow(map_image)
            plt.gca().add_patch(plt.Rectangle((0, 0), GRID_SIZE * cell - 1, GRID_SIZE * cell - 1, fill=False, edgecolor='black', lw=3))
            plt.axis('off')
            plt.title(f"Semantic Map ({active} Neurons Active)")
            plt.savefig("exp1_ontology.png", bbox_inches='tight', dpi=300)
            plt.close()

            z_logits_start = enc(vids[0:1])[:, 0, :]
            z_logits_c = z_logits_start.clone()
            z_logits_q = z_logits_start.clone()
            
            var_c, var_q = [], []
            for _ in range(100):
                z_logits_c = pred(z_logits_c)[:, 0, :]
                z_prob_c = F.softmax(apply_som_convolution(z_logits_c * eval_temp, 1.5), dim=1)
                var_c.append(z_prob_c.std().item())
                
                z_logits_q = pred(z_logits_q)[:, 0, :]
                z_prob_next_q = F.softmax(apply_som_convolution(z_logits_q * eval_temp, 1.5), dim=1)
                winner = torch.argmax(z_prob_next_q, dim=1)
                
                z_logits_q = torch.zeros_like(z_logits_q)
                z_logits_q[0, winner] = 10.0 
                var_q.append(z_prob_next_q.std().item())

            print("\n[Stability Metrics - Latent Sharpness (Std Dev)]")
            print("Step | Continuous | Quantized")
            print("---- | ---------- | ---------")
            for step in [0, 24, 49, 74, 99]:
                print(f"{step + 1:4d} | {var_c[step]:.8f} | {var_q[step]:.8f}")
            
            plt.figure(figsize=(8, 5))
            plt.plot(var_c, 'r', label='Continuous (Drifts to Blur)')
            plt.plot(var_q, 'b', label='Quantized (Stays Sharp)')
            plt.title("Imagination Stability (100 Steps)")
            plt.ylabel("Latent Sharpness (Std Dev)"); plt.xlabel("Time Step")
            plt.legend(); plt.grid(alpha=0.3)
            plt.savefig("exp4_stability.png", bbox_inches='tight', dpi=300)
            plt.close()

        elif paradigm == 'active':
            z_logits_start = enc(vids[0:1])[:, 0, :]
            z_prob_start = F.softmax(apply_som_convolution(z_logits_start * eval_temp, 1.5), dim=1)
            start_winner = torch.argmax(z_prob_start, dim=1).item()

            plt.figure(figsize=(8, 8))
            plt.title("Agent Imagination Tree (Branching Futures)")
            plt.xlim(-0.5, 14.5); plt.ylim(14.5, -0.5)
            plt.grid(True, color='lightgray', linestyle='--')
            
            colors = ['blue', 'red', 'green', 'orange']
            labels = ['Up', 'Right', 'Down', 'Left']
            
            plt.plot(start_winner%15, start_winner//15, marker='*', markersize=15, color='black', label="Start State")
            
            for a_val in range(4):
                z_logits_q = torch.zeros_like(z_logits_start)
                z_logits_q[0, start_winner] = 10.0
                action_vec = torch.tensor([a_val], device=device)
                path_x, path_y = [start_winner%15], [start_winner//15]
                
                for _ in range(8): 
                    z_logits_q = pred(z_logits_q, action_vec)[:, 0, :]
                    z_prob_q_soft = F.softmax(apply_som_convolution(z_logits_q * eval_temp, 1.5), dim=1)
                    winner = torch.argmax(z_prob_q_soft, dim=1).item()
                    
                    z_logits_q = torch.zeros_like(z_logits_q)
                    z_logits_q[0, winner] = 10.0 
                    path_x.append(winner % 15)
                    path_y.append(winner // 15)
                    
                plt.plot(path_x, path_y, marker='o', markersize=6, color=colors[a_val], label=f"Action '{labels[a_val]}'", linewidth=2)

            active = len(np.unique(torch.argmax(F.softmax(apply_som_convolution(enc(vids)[:,0,:] * eval_temp, 1.5), dim=1), dim=1).cpu().numpy()))
            print(f"[{paradigm.upper()}] Active Neurons: {active} / {LATENT_DIM}")

            plt.legend()
            plt.savefig("exp3_imagination_tree.png", bbox_inches='tight', dpi=300)
            plt.close()

        elif paradigm == 'compositional':
            logits_all = enc(vids)
            prob_A = F.softmax(apply_som_convolution(logits_all[:, 0, :] * eval_temp, 1.5), dim=1)
            prob_B = F.softmax(apply_som_convolution(logits_all[:, 1, :] * eval_temp, 1.5), dim=1)
            
            winners_A = torch.argmax(prob_A, dim=1).cpu().numpy()
            winners_B = torch.argmax(prob_B, dim=1).cpu().numpy()
            
            cell = 64
            map_A = np.full((GRID_SIZE * cell, GRID_SIZE * cell, 3), 255, dtype=np.uint8)
            map_B = np.full((GRID_SIZE * cell, GRID_SIZE * cell, 3), 255, dtype=np.uint8)
            
            active_A, active_B = 0, 0
            for idx in range(LATENT_DIM):
                r_pos, c_pos = idx // GRID_SIZE, idx % GRID_SIZE
                
                frames_A = vids[winners_A == idx].cpu().numpy()
                if len(frames_A) > 0:
                    active_A += 1
                    avg_A = np.mean(frames_A, axis=0).transpose(1, 2, 0) * 255.0
                    avg_A = convert_to_white_bg(avg_A)
                    map_A[r_pos*cell:(r_pos+1)*cell, c_pos*cell:(c_pos+1)*cell] = avg_A.astype(np.uint8)
                    cv2.rectangle(map_A, (c_pos*cell, r_pos*cell), ((c_pos+1)*cell, (r_pos+1)*cell), (200, 200, 200), 2)
                
                frames_B = vids[winners_B == idx].cpu().numpy()
                if len(frames_B) > 0:
                    active_B += 1
                    avg_B = np.mean(frames_B, axis=0).transpose(1, 2, 0) * 255.0
                    avg_B = convert_to_white_bg(avg_B)
                    map_B[r_pos*cell:(r_pos+1)*cell, c_pos*cell:(c_pos+1)*cell] = avg_B.astype(np.uint8)
                    cv2.rectangle(map_B, (c_pos*cell, r_pos*cell), ((c_pos+1)*cell, (r_pos+1)*cell), (200, 200, 200), 2)

            print(f"[COMPOSITIONAL] Grid A Active Neurons: {active_A} / {LATENT_DIM}")
            print(f"[COMPOSITIONAL] Grid B Active Neurons: {active_B} / {LATENT_DIM}")

            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            
            axs[0].imshow(map_A)
            axs[0].set_title(f"Grid A Ontology ({active_A} Active)")
            axs[0].axis('off')
            axs[0].add_patch(plt.Rectangle((0, 0), GRID_SIZE * cell - 1, GRID_SIZE * cell - 1, fill=False, edgecolor='black', lw=3))
            
            axs[1].imshow(map_B)
            axs[1].set_title(f"Grid B Ontology ({active_B} Active)")
            axs[1].axis('off')
            axs[1].add_patch(plt.Rectangle((0, 0), GRID_SIZE * cell - 1, GRID_SIZE * cell - 1, fill=False, edgecolor='black', lw=3))
            
            plt.tight_layout()
            plt.savefig("exp1_factorized_ontology.png", bbox_inches='tight', dpi=300)
            plt.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paradigm', type=str, choices=['passive', 'active', 'compositional'], default='passive')
    args, unknown = parser.parse_known_args()
    enc, pred, vids, acts = train_model(args.paradigm, epochs=100)
    run_evaluations(args.paradigm, enc, pred, vids, acts)               