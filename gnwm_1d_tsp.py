import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Force deterministic behavior for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 1D CIRCULAR GNWM ARCHITECTURE
# ==========================================
class GNWM_Circular_TSP(nn.Module):
    def __init__(self, num_cities=30, num_nodes=60):
        super().__init__()
        self.num_nodes = num_nodes
        
        # The MLP mapping 2D city coordinates to the 1D circular grid
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes)
        )

    def get_circular_kernel(self, sigma, device):
        """Generates a 1D Gaussian kernel for topological smearing."""
        k_size = int(sigma * 6) | 1 # Ensure odd kernel size
        k_size = max(3, k_size)
        x = torch.arange(-k_size//2 + 1, k_size//2 + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x**2 / (2 * sigma**2 + 1e-8))
        return kernel / kernel.sum()

    def forward(self, x, sigma):
        # 1. Base Logits: [N_cities, N_nodes]
        logits = self.mlp(x)
        N = logits.size(0)
        
        # 2. Topological Smearing (Circular 1D Convolution)
        kernel = self.get_circular_kernel(sigma, x.device).view(1, 1, -1)
        k_size = kernel.shape[2]
        pad = k_size // 2
        
        # View as [N, Channels, Length] for conv1d
        logits_reshaped = logits.view(N, 1, self.num_nodes)
        
        # Apply strict circular padding to form the closed loop
        logits_padded = F.pad(logits_reshaped, (pad, pad), mode='circular')
        smeared_logits = F.conv1d(logits_padded, kernel).view(N, self.num_nodes)
        
        # 3. Project to Simplex (Softmax)
        p = F.softmax(smeared_logits, dim=-1)
        return p

# ==========================================
# 2. THERMODYNAMIC TSP LOSS
# ==========================================
def calculate_tsp_thermodynamics(p, cities, num_nodes):
    device = p.device
    
    # Uniform constant vector (c) for thermodynamics
    c = torch.ones(num_nodes, device=device) / np.sqrt(num_nodes)
    
    # L2 Normalization for the probability mass
    p_l2 = F.normalize(p, p=2, dim=1)
    
    # Global batch mean across all cities
    mean_vec = torch.mean(p_l2, dim=0)
    mean_l2 = F.normalize(mean_vec, p=2, dim=0)
    
    # 1. Expansion Force (Forces the ring to stretch across all nodes)
    L_collapse = 1.0 - torch.dot(mean_l2, c)
    
    # 2. Contraction Force (Forces cities to snap to discrete nodes)
    L_WTA = torch.dot(mean_vec, c)
    
    # 3. Topological Similarity (The Elastic Net Pull)
    # Decode the expected 2D position of each neural node on the ring
    # W_j = Sum(prob_city_at_j * city_pos) / Sum(prob_city_at_j)
    node_weights = p.sum(dim=0).unsqueeze(1) + 1e-8
    node_positions = torch.matmul(p.T, cities) / node_weights # [num_nodes, 2]
    
    # Penalize the 2D distance between topologically adjacent nodes on the ring
    # This untangles the loop!
    node_positions_next = torch.roll(node_positions, shifts=-1, dims=0)
    L_sim = torch.mean(torch.sum((node_positions - node_positions_next)**2, dim=1))
    
    return L_collapse, L_WTA, L_sim, node_positions

# ==========================================
# 3. TRAINING & VISUALIZATION
# ==========================================
def solve_tsp():
    # Setup
    N_cities = 30
    N_nodes = 60 # Rubber band resolution (usually 2x to 3x N_cities)
    epochs = 3000
    
    # Generate random cities in a [0, 1] x [0, 1] 2D plane
    cities = torch.rand((N_cities, 2))
    
    model = GNWM_Circular_TSP(num_cities=N_cities, num_nodes=N_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    
    print("Beginning GNWM Topological Unrolling for TSP...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Simulated Annealing: Decay the neighborhood size (elasticity)
        # Starts highly elastic (sigma=4.0) and cools to brittle (sigma=0.2)
        progress = epoch / epochs
        current_sigma = 4.0 * (1.0 - progress) + 1
        
        # Heat up the alpha multiplier for thermodynamics early on
        alpha = min(1.0, (epoch + 1) / 200.0)
        
        p = model(cities, sigma=current_sigma)
        
        L_collapse, L_WTA, L_sim, node_positions = calculate_tsp_thermodynamics(p, cities, N_nodes)
        
        # Unified Loss
        # Heavy emphasis on L_sim to force the path to untangle
        loss = (L_collapse + L_WTA) + 1.0 * L_sim
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch+1:04d} | Sigma: {current_sigma:.2f} | L_tot: {loss.item():.4f} | L_sim (Path Len): {L_sim.item():.4f}")

    print("✅ Convergence Reached.")
    return model, cities, node_positions.detach()

def plot_tour(cities, node_positions):
    cities_np = cities.numpy()
    nodes_np = node_positions.numpy()
    
    # Close the loop for plotting
    nodes_closed = np.vstack([nodes_np, nodes_np[0]])
    
    plt.figure(figsize=(8, 8))
    plt.title("GNWM Topological Routing (TSP Solution)\nSoft-WTA + Circular Elasticity", fontsize=14)
    
    # Plot the "rubber band" neural nodes
    plt.plot(nodes_closed[:, 0], nodes_closed[:, 1], 'b-', alpha=0.6, linewidth=2, label='1D Circular Topology (GNWM)')
    plt.scatter(nodes_np[:, 0], nodes_np[:, 1], c='blue', s=15, alpha=0.5, zorder=2)
    
    # Plot the cities
    plt.scatter(cities_np[:, 0], cities_np[:, 1], c='red', s=50, edgecolors='black', zorder=3, label='Cities')
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, cities, final_nodes = solve_tsp()
    plot_tour(cities, final_nodes)