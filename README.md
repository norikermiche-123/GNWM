# The Global Neural World Model (GNWM)

> **Official PyTorch Implementation of: "The Global Neural World Model: Spatially Grounded Discrete Topologies for Action-Conditioned Planning"**

The GNWM is a self-stabilizing framework that achieves topological quantization through balanced continuous entropy constraints. Operating as an action-conditioned Joint-Embedding Predictive Architecture (JEPA), it maps continuous environments onto discrete 2D grids, enforcing translational equivariance without pixel-level reconstruction.

This repository contains the core training loops and plotting evaluations to reproduce the primary experiments from the paper, demonstrating drift-free topological tracking and compositional multi-object factorization.

## 🚀 Key Features
* **Thermodynamic Equilibrium:** Replaces fragile anti-collapse heuristics (like EMA target networks or BYOL weight decay) with mathematically guaranteed expansion/contraction constraints.
* **Fully Differentiable SOMs:** Upgrades classic Self-Organizing Maps with continuous differentiable probabilities, decoupling topology from activation for exact gradient flow.
* **Drift-Free Imagination:** Employs "grid snapping" during inference to arrest manifold drift, acting as a native Error-Correction Code for long-horizon rollouts.

## 🛠️ Installation

Requirements: Python 3.8+ and PyTorch.

```bash
git clone [https://github.com/yourusername/GNWM.git](https://github.com/yourusername/GNWM.git)
cd GNWM
pip install -r requirements.txt
(Dependencies: torch, numpy, matplotlib, opencv-python)

🔬 Usage & Reproducing Paradigms
The core script gnwm_official_release.py contains simulated 2D physics environments and the unified GNWM architecture. You can execute three distinct experimental paradigms:

1. Paradigm A: Passive Observation
Evaluates the network's ability to map a continuous kinetic environment onto a discrete 15x15 topological grid without action labels or codebook collapse.

Bash
python gnwm_official_release.py --paradigm passive
Outputs: * exp1_ontology.png - The quantized semantic map displaying generalized conceptual manifolds.

exp4_stability.png - Evaluation of latent sharpness over 100-step autoregressive rollouts.

2. Paradigm B: Active Agent Control
Evaluates the Action-Conditioned Spatial Predictor, demonstrating the generation of deterministic, orthogonal branches across the latent grid.

Bash
python gnwm_official_release.py --paradigm active
Outputs: * exp3_imagination_tree.png - Visualizes the branching deterministic futures for the 4 available actions without hallucinating dead nodes.

3. Paradigm C: Compositional Multi-Object Factorization
Evaluates dimensionality reduction and dual-channel separation for a multi-entity scene.

Bash
python gnwm_official_release.py --paradigm compositional
Outputs: * exp1_factorized_ontology.png - Side-by-side semantic maps showing autonomous factorization of Object A and Object B.

4. Paradigm D: Abstract Semantic Topology
Evaluates the architecture's capacity to act as a generalized causal mapping engine for non-spatial sequences. The network processes 40 unique continuous word embeddings governed by a rigid grammatical rule (Noun → Verb → Adjective → Object) and autonomously clusters them into a structured, discrete 2D ontology.

bash
python paradigm_D_semantics.py
Outputs: * exp6_cognitive_topology.png - The clustered semantic grid demonstrating the geometric organization of abstract causal concepts.

5. Combinatorial Generalization: 1D Circular Topologies (TSP)Evaluates the GNWM's capacity to act as a differentiable elastic net, unrolling discrete combinatorial logic using only continuous thermodynamic gradients. This script solves a 30-city Traveling Salesman Problem (TSP) using a 1D circular topology and Dynamic Tension Decay.

Bash
python gnwm_1d_tsp.py
Outputs: * A pop-up matplotlib visualization displaying the discrete TSP tour mapped by the neural nodes, showcasing the final tension snap.Console logs detailing the simulated annealing, topological elasticity ($\sigma$), and loss metrics.

📜 Citation
If you find this code or the theoretical framework useful in your research, please cite:

Code snippet
@article{kermiche2026gnwm,
  title={The Global Neural World Model: Spatially Grounded Discrete Topologies for Action-Conditioned Planning},
  author={Kermiche, Noureddine},
  journal={arXiv preprint},
  year={2026}
}
