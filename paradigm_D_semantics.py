import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

GRID_SIZE = 15
LATENT_DIM = GRID_SIZE * GRID_SIZE
CHANNELS = 3

# ==========================================
# 1. Generate Causal MNIST Video
# ==========================================
def get_mnist_digits():
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    digits = {0: [], 1: [], 2: []}
    for img, label in dataset:
        if label in digits and len(digits[label]) < 10: 
            digits[label].append(torch.tensor(np.array(img)).float() / 255.0)
        if all(len(v) == 10 for v in digits.values()):
            break
    return digits

def generate_causal_mnist(frames_count=1000):
    print("Generating Causal MNIST Data...")
    digits = get_mnist_digits()
    frames = []
    labels = []
    
    x, y = 18.0, 18.0
    vx, vy = 1.5, 1.5
    current_digit = 0
    
    for t in range(frames_count):
        if t % 30 == 0:
            current_digit = np.random.randint(0, 3) # Shapeshift!
            
            # CAUSAL PHYSICS RULES: Identity dictates velocity
            sign_x = np.sign(vx) if vx != 0 else 1.0
            sign_y = np.sign(vy) if vy != 0 else 1.0
            
            if current_digit == 0:
                vx, vy = sign_x * 1.5, sign_y * 1.5 # 0 is Slow
            elif current_digit == 1:
                vx, vy = sign_x * 4.0, sign_y * 4.0 # 1 is Fast
            elif current_digit == 2:
                vx, vy = sign_x * 2.5, sign_y * 2.5 # 2 is Medium
            
        # 2 is erratic - randomly flips Y direction 10% of the time
        if current_digit == 2 and np.random.rand() < 0.10:
            vy *= -1
            
        frame = torch.zeros((64, 64))
        x += vx; y += vy
        
        # 28x28 digit bounds within 64x64 canvas
        if x > 35 or x < 1: vx *= -1; x += vx
        if y > 35 or y < 1: vy *= -1; y += vy
        
        digit_img = digits[current_digit][np.random.randint(0, 10)]
        ix, iy = int(x), int(y)
        frame[iy:iy+28, ix:ix+28] = digit_img
        
        frame_3c = frame.unsqueeze(0).repeat(3, 1, 1)
        frames.append(frame_3c)
        labels.append(current_digit)
        
    return torch.stack(frames), np.array(labels)

class FastDataset(Dataset):
    def __init__(self, tensor): self.data = tensor
    def __len__(self): return len(self.data) - 1
    def __getitem__(self, idx): return self.data[idx], self.data[idx+1]

# ==========================================
# 2. GNWM Architecture (3 Grids)
# ==========================================
class VisionEncoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, CHANNELS * LATENT_DIM)
        )
    def forward(self, x): 
        out = self.net(x)
        return out.view(-1, CHANNELS, LATENT_DIM)

class VisionPredictor3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CHANNELS * LATENT_DIM, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, CHANNELS * LATENT_DIM)
        )
    def forward(self, z_grids):
        B = z_grids.shape[0]
        flat_z = z_grids.view(B, -1)
        out = flat_z + self.net(flat_z)
        return out.view(B, CHANNELS, LATENT_DIM)

def apply_som_convolution(logits, sigma=1.5):
    device = logits.device
    k = max(3, int(6 * sigma) | 1)
    grid = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    w = (gaussian / gaussian.sum()).view(1, 1, k, k).to(device)
    # Apply SOM independently across all batch and channel dims
    x_conv = F.conv2d(logits.view(-1, 1, GRID_SIZE, GRID_SIZE), w, padding='same')
    return x_conv.view(-1, LATENT_DIM)

def get_grid_coords(device):
    grid_1d = torch.linspace(-1, 1, GRID_SIZE, device=device)
    y_grid, x_grid = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    return torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

# ==========================================
# 3. Training with Tri-Grid Thermodynamics
# ==========================================
def compute_single_grid_thermo(z_logits_t, z_logits_n, p_logits_n, epoch):
    device = z_logits_t.device
    const = F.normalize(torch.ones((1, LATENT_DIM), device=device), p=2, dim=1)
    
    # Annealing
    z_t = z_logits_t + torch.randn_like(z_logits_t) * 0.5 if epoch < 10 else z_logits_t
    z_n = z_logits_n + torch.randn_like(z_logits_n) * 0.5 if epoch < 10 else z_logits_n
    
    z_prob_t = F.softmax(apply_som_convolution(z_t, 1.5), dim=1)
    z_l2_t = F.normalize(z_prob_t, p=2, dim=1)
    z_l2_n = F.normalize(F.softmax(apply_som_convolution(z_n, 1.5), dim=1), p=2, dim=1)
    p_l2_n = F.normalize(F.softmax(apply_som_convolution(p_logits_n, 1.5), dim=1), p=2, dim=1)
    
    mean_l2 = F.normalize(torch.mean((z_l2_t + z_l2_n) / 2.0, dim=0, keepdim=True), p=2, dim=1)
    L_collapse = 1.0 - torch.sum(mean_l2 * const)
    L_WTA = torch.sum(torch.mean((z_l2_t + z_l2_n) / 2.0, dim=0, keepdim=True) * const)
    L_sim = 1.0 - torch.mean(torch.sum(p_l2_n * z_l2_n.detach(), dim=1))
    
    alpha = 2.0 if epoch < 10 else 1.0 
    loss = alpha * (L_collapse + L_WTA) + 0.5 * L_sim
    return loss, z_prob_t

def train_causal_gnwm(video_tensor, epochs=40):
    device = video_tensor.device
    enc = VisionEncoder3().to(device)
    pred = VisionPredictor3().to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    loader = DataLoader(FastDataset(video_tensor), batch_size=64, shuffle=True)
    coords = get_grid_coords(device)
    
    print("Training 3-Channel Causal GNWM...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_t, x_next in loader:
            opt.zero_grad()
            z_logits_t_all = enc(x_t)
            z_logits_n_all = enc(x_next)
            p_logits_n_all = pred(z_logits_t_all)
            
            total_loss = 0.0
            positions = []
            
            # Compute physics for each grid independently
            for c in range(CHANNELS):
                loss_c, z_prob_c = compute_single_grid_thermo(
                    z_logits_t_all[:, c, :], 
                    z_logits_n_all[:, c, :], 
                    p_logits_n_all[:, c, :], 
                    epoch
                )
                total_loss += loss_c
                positions.append(torch.matmul(z_prob_c, coords))
                
            # Repel grids from tracking the same physics trajectory
            L_repel = (torch.mean(F.cosine_similarity(positions[0], positions[1])) +
                       torch.mean(F.cosine_similarity(positions[1], positions[2])) +
                       torch.mean(F.cosine_similarity(positions[0], positions[2]))) * 0.5
            
            total_loss += L_repel
            total_loss.backward()
            opt.step()
            epoch_loss += total_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Total Loss: {epoch_loss/len(loader):.4f}")
            
    return enc

# ==========================================
# 4. Tri-Grid Evaluation
# ==========================================
def evaluate_3grids(enc, vids, labels):
    enc.eval()
    with torch.no_grad():
        logits_all = enc(vids)
        winners_A = torch.argmax(apply_som_convolution(logits_all[:, 0, :]), dim=1).cpu().numpy()
        winners_B = torch.argmax(apply_som_convolution(logits_all[:, 1, :]), dim=1).cpu().numpy()
        winners_C = torch.argmax(apply_som_convolution(logits_all[:, 2, :]), dim=1).cpu().numpy()
        
    def plot_grid(winners, title, ax):
        grid_colors = np.ones((GRID_SIZE, GRID_SIZE, 3)) 
        shape_colors = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]} # Red=0, Green=1, Blue=2
        
        neuron_labels = {i: [] for i in range(LATENT_DIM)}
        for t in range(len(winners)):
            neuron_labels[winners[t]].append(labels[t].item())
            
        active, primary_digit = 0, []
        for idx, labs in neuron_labels.items():
            if len(labs) > 0:
                active += 1
                most_freq = max(set(labs), key=labs.count)
                primary_digit.append(most_freq)
                r, c = idx // GRID_SIZE, idx % GRID_SIZE
                grid_colors[r, c] = shape_colors[most_freq]
                
        # Determine dominant digit for this grid
        dom = max(set(primary_digit), key=primary_digit.count) if primary_digit else -1
        dom_name = {0: "Zeros (Red)", 1: "Ones (Green)", 2: "Twos (Blue)", -1: "None"}[dom]
                
        ax.imshow(grid_colors)
        ax.set_title(f"{title}\nActive: {active} | Dominant: {dom_name}")
        ax.axis('off')

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_grid(winners_A, "Grid A", axs[0])
    plot_grid(winners_B, "Grid B", axs[1])
    plot_grid(winners_C, "Grid C", axs[2])
    
    plt.tight_layout()
    plt.savefig("expC_causal_mnist_3grids.png", bbox_inches='tight')
    print("\n✅ Saved 'expC_causal_mnist_3grids.png'. Did they separate?")

if __name__ == "__main__":
    device = torch.device('cpu') # Running on CPU per your setup
    vids, labels = generate_causal_mnist(1000)
    vids = vids.to(device)
    e = train_causal_gnwm(vids)
    evaluate_3grids(e, vids, labels)