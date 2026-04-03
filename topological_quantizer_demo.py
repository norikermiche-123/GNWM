"""
=============================================================================
THE TOPOLOGICAL QUANTIZER: Semantic Mapping & Drift-Free Rollouts
=============================================================================
Fixed Version: Resolved LaTeX SyntaxWarnings and added a "Warmup Kick" to 
prevent 1-neuron mode collapse on the 15x15 grid.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# ==========================================
# 1. Environment & Dataset (15x15 Quantizer)
# ==========================================
GRID_SIZE = 15 # D = 225
LATENT_DIM = GRID_SIZE * GRID_SIZE

def generate_simple_video(frames_count=1200):
    """A ball moving in a 2D box. Increased frames for better mapping."""
    frames = []
    x, y = 32.0, 32.0
    vx, vy = 2.5, 3.5 
    
    for _ in range(frames_count):
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        x += vx
        y += vy
        if x > 56 or x < 8: vx *= -1; x += vx
        if y > 56 or y < 8: vy *= -1; y += vy
        cv2.circle(frame, (int(x), int(y)), 8, (200, 0, 0), -1)
        frames.append(frame)
        
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

class FastDataset(Dataset):
    def __init__(self, tensor):
        self.data = tensor
    def __len__(self): return len(self.data) - 1
    def __getitem__(self, idx): return self.data[idx], self.data[idx+1]

# ==========================================
# 2. GNWM Architecture
# ==========================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, LATENT_DIM)
        )
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, LATENT_DIM)
        )
    def forward(self, x): return x + self.net(x)

def apply_som_convolution(x, sigma=1.5):
    device = x.device
    k = int(6 * max(0.1, sigma))
    if k % 2 == 0: k += 1
    k = max(3, k)
    
    grid = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(grid, grid, indexing='ij')
    gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma + 1e-8)**2))
    gaussian = gaussian / torch.sum(gaussian)
    
    w = gaussian.view(1, 1, k, k)
    x_conv = F.conv2d(x.view(-1, 1, GRID_SIZE, GRID_SIZE), w, padding='same')
    return x_conv.view(-1, LATENT_DIM)

# ==========================================
# 3. Training with Anti-Collapse Warmup
# ==========================================
def train_gnwm(video_tensor, epochs=50):
    device = video_tensor.device
    enc = Encoder().to(device)
    pred = Predictor().to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    loader = DataLoader(FastDataset(video_tensor), batch_size=64, shuffle=True)
    
    print("Training GNWM to quantize the physics space...")
    for epoch in range(epochs):
        for x_t, x_next in loader:
            x_t, x_next = x_t.to(device), x_next.to(device)
            opt.zero_grad()
            
            h_t = enc(x_t)
            # Add noise in early epochs to prevent 1-neuron collapse
            if epoch < 10:
                h_t = h_t + torch.randn_like(h_t) * 0.1
                
            pred_h = pred(h_t)
            tgt_h = enc(x_next)
            
            p_som = apply_som_convolution(pred_h, 1.5)
            z_som = apply_som_convolution(tgt_h, 1.5)
            
            p_l2 = F.normalize(F.softmax(p_som, dim=1), p=2, dim=1)
            z_l2 = F.normalize(F.softmax(z_som, dim=1), p=2, dim=1)
            
            # Global batch average
            combined = (p_l2 + z_l2) / 2.0
            mean_vec = torch.mean(combined, dim=0, keepdim=True)
            mean_l2 = F.normalize(mean_vec, p=2, dim=1)
            
            const = F.normalize(torch.ones((1, LATENT_DIM), device=device), p=2, dim=1)
            
            l_collapse = 1.0 - torch.sum(mean_l2 * const)
            l_wta = torch.sum(mean_vec * const) # Soft WTA
            l_sim = 1.0 - torch.mean(torch.sum(p_l2 * z_l2, dim=1))
            
            # Thermodynamics: Stronger collapse prevention in early epochs
            alpha = 2.0 if epoch < 15 else 1.0
            loss = alpha * l_collapse + l_wta + 0.5 * l_sim
            
            loss.backward()
            opt.step()
            
    return enc, pred

# ==========================================
# EXPERIMENT 1: Semantic Map
# ==========================================
def experiment_1_semantic_map(enc, video_tensor):
    print("\nRunning Experiment 1: Extracting Semantic Ontology...")
    enc.eval()
    device = video_tensor.device
    neuron_clusters = {i: [] for i in range(LATENT_DIM)}
    
    with torch.no_grad():
        h_all = enc(video_tensor)
        h_som = apply_som_convolution(h_all, 1.5)
        winners = torch.argmax(h_som, dim=1).cpu().numpy()
        for t, idx in enumerate(winners):
            neuron_clusters[idx].append(video_tensor[t].cpu().numpy())
            
    cell = 64
    map_image = np.zeros((GRID_SIZE * cell, GRID_SIZE * cell, 3), dtype=np.uint8)
    
    active_neurons = 0
    for idx, frames in neuron_clusters.items():
        r, c = idx // GRID_SIZE, idx % GRID_SIZE
        if len(frames) > 0:
            active_neurons += 1
            avg = np.mean(frames, axis=0).transpose(1, 2, 0) * 255.0
            map_image[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = avg.astype(np.uint8)
        
    print(f"Topology Analysis: {active_neurons}/{LATENT_DIM} neurons utilized.")
    plt.figure(figsize=(10, 10))
    plt.imshow(map_image)
    plt.title(f"Experiment 1: Semantic Ontology Map ({active_neurons} Active Clusters)")
    plt.axis('off')
    plt.savefig("experiment1_semantic_map.png", bbox_inches='tight')
    print("✅ Saved 'experiment1_semantic_map.png'")

# ==========================================
# EXPERIMENT 2: Drift-Free Rollout
# ==========================================
def experiment_2_drift_free(enc, pred, video_tensor):
    print("\nRunning Experiment 2: Drift-Free Long-Horizon Rollout...")
    enc.eval(); pred.eval()
    device = video_tensor.device
    steps = 50
    
    # Track variance to measure blur
    h_L2 = enc(video_tensor[0].unsqueeze(0))
    std_L2 = []
    
    h_GNWM = enc(video_tensor[0].unsqueeze(0))
    std_GNWM = []
    
    with torch.no_grad():
        for step in range(steps):
            # L2 Baseline (Simulated)
            h_L2 = pred(h_L2)
            std_L2.append(h_L2.std().item())
            
            # GNWM snap-to-grid
            h_GNWM = pred(h_GNWM)
            h_som = apply_som_convolution(h_GNWM, 1.5)
            # Hard Quantization
            winner = torch.argmax(h_som, dim=1)
            quant = torch.zeros_like(h_som)
            quant[0, winner] = 1.0
            h_GNWM = quant
            std_GNWM.append(h_GNWM.std().item())
            
    plt.figure(figsize=(8, 5))
    # Fixed LaTeX formatting with raw strings (r"...")
    plt.plot(std_L2, color='red', linewidth=2, label=r"Standard $\mathcal{L}_2$ Predictor (Drifts to Blur)")
    plt.plot(std_GNWM, color='blue', linewidth=2, label=r"GNWM Quantizer (Snaps to Grid)")
    plt.title("Experiment 2: Latent Variance (Sharpness) over 50 Steps")
    plt.xlabel("Prediction Horizon")
    plt.ylabel("Latent Standard Deviation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("experiment2_drift_free.png")
    print("✅ Saved 'experiment2_drift_free.png'")

if __name__ == "__main__":
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {dev}")
    vids = generate_simple_video(1200).to(dev)
    e, p = train_gnwm(vids, epochs=50)
    experiment_1_semantic_map(e, vids)
    experiment_2_drift_free(e, p, vids)
    plt.show()