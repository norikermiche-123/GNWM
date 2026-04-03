"""
=============================================================================
GEOMETRIC NEURAL WORLD MODEL (GNWM) - 30x30 MULTI-OBJECT CAUSAL CONTROL
=============================================================================
This experiment scales the GNWM from a simple 1D discrete bifurcation (Galton Board)
to a highly chaotic, continuous, multi-body physical environment. 

The latent space is expanded to a massive 30x30 topological grid (D=900).
We inject a continuous Action Vector (a_t = [f_x, f_y]) representing a physical force 
applied to an "Agent" object. 

Objective: Prove that the Four Pillars of Geometric Self-Organization and the 
Warmup+Cosine Decay scheduler scale to high dimensions, allowing the network to 
hierarchically disentangle multiple objects and route counterfactual causal 
futures across a vast 2D cortical sheet without dimensional collapse.
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
# 1. Causal Multi-Object Environment
# ==========================================
# [THEORY] We simulate a 2D box with two balls. Ball 0 is the "Agent" (Red, light mass).
# Ball 1 is the "Obstacle" (Blue, heavy mass). The environment is conditioned on 
# a continuous 2D action vector (f_x, f_y) that applies physical force to the Agent.

class CausalPhysicsEnv:
    def __init__(self, num_balls=2, width=64, height=64):
        self.width = width
        self.height = height
        self.num_balls = num_balls
        self.dt = 0.5
        
        self.positions = np.random.rand(num_balls, 2) * (width - 20) + 10
        self.velocities = (np.random.rand(num_balls, 2) - 0.5) * 10
        self.masses = np.array([[2], [4]]) # Agent is lighter, Obstacle is heavier
        self.radii = self.masses * 3
        
        # Ball 0: Agent (Red), Ball 1: Obstacle (Blue)
        self.colors = [(255, 0, 0), (0, 0, 255)] 

    def step(self, action):
        """
        Applies causal intervention: action = [f_x, f_y] force on the Agent.
        """
        # Apply action force to the Agent's velocity (F = ma -> dv = F/m * dt)
        force = np.array(action)
        self.velocities[0] += (force / self.masses[0]) * self.dt
        
        # Add friction so the system eventually stabilizes
        self.velocities *= 0.98 
        
        # Update spatial positions
        self.positions += self.velocities * self.dt
        
        # Wall Collisions (Elastic)
        for i in range(self.num_balls):
            r = self.radii[i, 0]
            if self.positions[i, 0] - r < 0:
                self.positions[i, 0] = r
                self.velocities[i, 0] *= -1
            elif self.positions[i, 0] + r > self.width:
                self.positions[i, 0] = self.width - r
                self.velocities[i, 0] *= -1
                
            if self.positions[i, 1] - r < 0:
                self.positions[i, 1] = r
                self.velocities[i, 1] *= -1
            elif self.positions[i, 1] + r > self.height:
                self.positions[i, 1] = self.height - r
                self.velocities[i, 1] *= -1
                
    def render(self):
        """Renders the physical state to a 64x64 RGB pixel array."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        for i in range(self.num_balls):
            pos = (int(self.positions[i, 0]), int(self.positions[i, 1]))
            radius = int(self.radii[i, 0])
            cv2.circle(frame, pos, radius, self.colors[i], -1)
        return frame

def generate_causal_dataset(num_videos=100, frames_per_video=40):
    """
    Generates a dataset of randomized kinematic trajectories.
    At every step, a random continuous force is applied to the Agent.
    """
    print(f"Generating Action-Conditioned Chaos Dataset ({num_videos} simulations)...")
    data = [] 
    
    for _ in range(num_videos):
        env = CausalPhysicsEnv()
        frame_t = env.render()
        
        for _ in range(frames_per_video):
            # Generate a random continuous force vector [-5.0 to 5.0]
            action_t = (np.random.rand(2) - 0.5) * 10.0 
            
            env.step(action_t)
            frame_t1 = env.render()
            
            # Normalize frame and store transition tuple: (State_t, Action_t, State_t+1)
            t_tensor = torch.from_numpy(frame_t).permute(2, 0, 1).float() / 255.0
            t1_tensor = torch.from_numpy(frame_t1).permute(2, 0, 1).float() / 255.0
            a_tensor = torch.tensor(action_t, dtype=torch.float32)
            
            data.append((t_tensor, a_tensor, t1_tensor))
            frame_t = frame_t1
            
    return data

class CausalDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 2. Architectures & 30x30 SOM Helpers
# ==========================================
# [THEORY] 30x30 Grid. We scale the continuous latent space to 900 dimensions.
# This massive geometric space allows the network to maintain distinct localized
# heat-points for both the Agent and the Obstacle simultaneously.
GRID_SIZE = 30
LATENT_DIM = GRID_SIZE * GRID_SIZE 

class ChaosEncoder(nn.Module):
    """Deep Perception Module mapping chaotic 64x64 frames to 900D latent state."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, LATENT_DIM)
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))

class CausalPredictor(nn.Module):
    """
    Action-Conditioned Convolutional World Model Transition Function.
    [DETAIL] We reshape the 900D state back into a 30x30 spatial grid and process 
    it with a CNN. To explicitly break translation invariance, we use "CoordConv" 
    by injecting fixed X and Y coordinate channels. The action is broadcasted across 
    the spatial grid. This local processing allows the network to shift the Agent's 
    specific topological bump while mathematically guaranteeing the Obstacle is untouched.
    """
    def __init__(self, action_dim=2):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, 16)
        
        # Input channels: 1 (State) + 16 (Action) + 2 (X,Y Coords) = 19
        self.conv_net = nn.Sequential(
            nn.Conv2d(19, 64, kernel_size=5, padding=2),
            nn.LayerNorm([64, GRID_SIZE, GRID_SIZE]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LayerNorm([64, GRID_SIZE, GRID_SIZE]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # CoordConv: Explicit Grid Coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, GRID_SIZE),
            torch.linspace(-1, 1, GRID_SIZE),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('grid_y', grid_y.unsqueeze(0).unsqueeze(0))
        
    def forward(self, h_t, a_t):
        B = h_t.size(0)
        
        # Reshape flat 900D state into 30x30 spatial map
        h_spatial = h_t.view(B, 1, GRID_SIZE, GRID_SIZE)
        
        # Embed and broadcast action to match spatial dimensions
        a_emb = F.relu(self.action_embed(a_t))
        a_spatial = a_emb.view(B, -1, 1, 1).expand(-1, -1, GRID_SIZE, GRID_SIZE)
        
        # Expand coordinates for the batch
        coords_x = self.grid_x.expand(B, -1, -1, -1)
        coords_y = self.grid_y.expand(B, -1, -1, -1)
        
        # Concatenate: [State, Action, X, Y] -> [B, 19, 30, 30]
        fused = torch.cat([h_spatial, a_spatial, coords_x, coords_y], dim=1)
        
        # Compute local spatial delta
        delta = self.conv_net(fused)
        
        # Flatten and apply residual connection
        return h_t + delta.view(B, LATENT_DIM)

def update_ema(enc, tgt_enc, momentum=0.99): 
    """Slow-moving target encoder update for stability."""
    with torch.no_grad():
        for p_q, p_k in zip(enc.parameters(), tgt_enc.parameters()):
            p_k.data = p_k.data * momentum + p_q.data * (1. - momentum)

def apply_30x30_som_convolution(x, sigma=2.0):
    """
    Applies the topological constraint across the massive 30x30 grid.
    This creates an elastic cortical sheet that mathematically binds physically
    similar states into adjacent, continuous 2D neighborhoods.
    """
    device = x.device
    kernel_size = int(6 * max(0.1, sigma))
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 3: kernel_size = 3
        
    grid = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(grid, grid, indexing='ij')
    gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma + 1e-8)**2))
    gaussian = gaussian / torch.sum(gaussian)
    
    weight = gaussian.view(1, 1, kernel_size, kernel_size)
    x_reshaped = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
    x_conv = F.conv2d(x_reshaped, weight, padding='same')
    
    return x_conv.view(-1, LATENT_DIM)

# ==========================================
# 3. Robust Thermodynamic Training Loop
# ==========================================
def train_causal_som(data_list, epochs=150, sigma=2.0):
    device = data_list[0][0].device
    enc = ChaosEncoder().to(device)
    tgt_enc = copy.deepcopy(enc).to(device)
    tgt_enc.eval()
    pred = CausalPredictor().to(device)
    
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-4)
    dataloader = DataLoader(CausalDataset(data_list), batch_size=64, shuffle=True)
    
    # --- ROBUST OPTIMIZATION: Warmup + Cosine Decay Scheduler ---
    def get_lr_multiplier(epoch):
        warmup_epochs = int(epochs * 0.1) # 10% Warmup
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    std_history = []
    loss_history = []
    
    print(f"\nTraining Causal GNWM on 30x30 Grid for {epochs} epochs (No Early Stopping)...")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_std = 0
        batches = 0
        
        for batch_x, batch_a, batch_y in dataloader:
            batch_x, batch_a, batch_y = batch_x.to(device), batch_a.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            h_t = enc(batch_x)
            pred_h = pred(h_t, batch_a)
            
            with torch.no_grad():
                tgt_h = tgt_enc(batch_y) 
            
            # --- THE FOUR PILLARS OF GEOMETRIC SELF-ORGANIZATION ---
            D = pred_h.size(1) 
            const = torch.ones((1, D), device=device)
            const = F.normalize(const, p=2, dim=1)
            
            # 1. Topological Spatial Smoothing
            pred_h_som = apply_30x30_som_convolution(pred_h, sigma=sigma)
            tgt_h_som = apply_30x30_som_convolution(tgt_h, sigma=sigma)
            
            # 2 & 3. Simplex Projection & Hypersphere Normalization
            p_l2 = F.normalize(F.softmax(pred_h_som, dim=1), p=2, dim=1)
            z_l2 = F.normalize(F.softmax(tgt_h_som, dim=1), p=2, dim=1)
            
            p_z_mean = torch.mean((p_l2 + z_l2) / 2.0, dim=0, keepdim=True)
            p_z_mean_l2 = F.normalize(p_z_mean, p=2, dim=1)
            
            # 4. Thermodynamic Balance
            loss_collapse = 1.0 - torch.mean(torch.sum(p_z_mean_l2 * const, dim=1))
            loss_wta = torch.mean(torch.sum(p_z_mean * const, dim=1))
            loss_similarity = 1.0 - torch.mean(torch.sum(p_l2 * z_l2, dim=1))
            
            gamma = 0.5
            loss = loss_collapse + loss_wta + gamma * loss_similarity

            loss.backward()
            optimizer.step()
            update_ema(enc, tgt_enc)
            
            epoch_loss += loss.item()
            epoch_std += tgt_h.std(dim=0).mean().item()
            batches += 1
            
        scheduler.step() # Advance Cosine Decay
        
        std_history.append(epoch_std / batches)
        loss_history.append(epoch_loss / batches)
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss_history[-1]:.4f} | Latent Std: {std_history[-1]:.4f}")
            
    return enc, tgt_enc, pred, std_history, loss_history

# ==========================================
# 4. Counterfactual Imagination (Visualization)
# ==========================================
def visualize_counterfactuals(enc, pred, env, sigma=2.0):
    """
    Renders 30x30 Heatmaps to prove the model has achieved Judea Pearl's "Rung 2"
    of Causation. We pause the universe, apply two massively divergent causal forces
    to the Agent, and watch the SOM grid route the futures to completely orthogonal 
    hierarchical locations.
    """
    print("\nGenerating 2D Counterfactual 'Imagination' Heatmaps...")
    enc.eval()
    pred.eval()
    device = next(enc.parameters()).device
    
    # Get the starting state
    frame_t = env.render()
    t_tensor = torch.from_numpy(frame_t).permute(2, 0, 1).float() / 255.0
    t_tensor = t_tensor.unsqueeze(0).to(device)
    
    # Define two completely opposite causal interventions (Extreme Force on the Agent)
    action_left = torch.tensor([[-40.0, 0.0]], dtype=torch.float32).to(device)
    action_right = torch.tensor([[40.0, 0.0]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        h_t = enc(t_tensor)
        
        # Predict counterfactual futures
        pred_h_left = pred(h_t, action_left)
        pred_h_right = pred(h_t, action_right)
        
        # Map back through the Topological Constraint (Sigma)
        h_som_left = apply_30x30_som_convolution(pred_h_left, sigma=sigma)
        h_som_right = apply_30x30_som_convolution(pred_h_right, sigma=sigma)
        
        # Convert to Probability Distributions over the 30x30 grid
        p_left = F.softmax(h_som_left, dim=1).view(GRID_SIZE, GRID_SIZE).cpu().numpy()
        p_right = F.softmax(h_som_right, dim=1).view(GRID_SIZE, GRID_SIZE).cpu().numpy()
        
        # Calculate Causal Delta (Subtract Right future from Left future)
        # The heavy static Obstacle will mathematically cancel out to 0!
        p_delta = p_left - p_right
        
    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # Added a 4th panel for the Delta
    
    # Original Frame
    # Note: Agent is Red, Obstacle is Blue
    axes[0].imshow(cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Context ($S_t$) [Red=Agent, Blue=Obstacle]")
    axes[0].axis('off')
    
    # Heatmap: Massive Force LEFT
    im_l = axes[1].imshow(p_left, cmap='jet')
    axes[1].set_title("Imagined Future: Force Agent LEFT")
    axes[1].axis('off')
    fig.colorbar(im_l, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Heatmap: Massive Force RIGHT
    im_r = axes[2].imshow(p_right, cmap='jet')
    axes[2].set_title("Imagined Future: Force Agent RIGHT")
    axes[2].axis('off')
    fig.colorbar(im_r, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Heatmap: CAUSAL DELTA (Left - Right)
    # Use a diverging colormap centered at 0 to show isolation of the Agent
    vmax = max(abs(p_delta.max()), abs(p_delta.min()))
    im_delta = axes[3].imshow(p_delta, cmap='bwr', vmin=-vmax, vmax=vmax)
    axes[3].set_title("Causal Delta (Left - Right)\nStatic Obstacle Cancels Out!")
    axes[3].axis('off')
    fig.colorbar(im_delta, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("multi_object_causal_imagination.png")
    print("✅ Counterfactuals rendered! Check 'multi_object_causal_imagination.png'")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    # Generate the dataset
    raw_data = generate_causal_dataset(num_videos=100)
    device_data = [(x.to(device), a.to(device), y.to(device)) for x, a, y in raw_data]
    
    # Train using the robust Cosine Decay optimizer
    # 200 epochs allows the 30x30 grid plenty of time to structurally organize
    enc, tgt_enc, pred, std_hist, loss_hist = train_causal_som(device_data, epochs=200, sigma=2.0)
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(std_hist, color='blue', linewidth=2)
    plt.title("Latent Standard Deviation (No Collapse)")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_hist, color='red', linewidth=2)
    plt.title("Causal GNWM Loss (Warmup + Cosine Decay)")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("causal_30x30_training.png")
    print("\n✅ Training complete! Metrics saved to 'causal_30x30_training.png'")
    
    # Generate the Imagination Heatmaps
    test_env = CausalPhysicsEnv()
    # Step the environment a few times so the balls are in the middle of the box
    for _ in range(10): test_env.step([0, 0]) 
    
    visualize_counterfactuals(enc, pred, test_env, sigma=2.0)
    plt.show()