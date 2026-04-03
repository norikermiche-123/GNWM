Geometric Neural World Model (GNWM) 🌐

Official codebase for the paper: "Self-Supervised World Models: Breaking Dimensional Collapse via Geometric Self-Organization"

This repository contains the PyTorch implementations and empirical physics benchmarks demonstrating the failure modes of standard non-contrastive predictive architectures (like JEPA and BYOL) and introducing the Geometric Neural World Model (GNWM)—a mathematically principled approach to world modeling that utilizes topological self-organization to prevent dimensional collapse and achieve drift-free causal imagination.

🧠 The Core Problem: Predictive Perception $\neq$ World Models

Current flagship implementations of self-supervised predictive architectures rely on deterministic $\mathcal{L}_2$ objectives to predict latent futures. However, minimizing an $\mathcal{L}_2$ distance inherently forces the predictor to converge to the conditional expectation of the target.

In stochastic, multimodal physical environments, this causes the network to act as a spatial low-pass filter. It averages out diverging valid futures into a blurry, impossible superposition ("ghost mean"). To survive dimensional collapse, these models rely on unprincipled stabilization hacks like Batch Normalization, which inject severe gradient noise and fail to learn sharp causal boundaries.

🚀 The Solution: Geometric Self-Organization & The Topological Quantizer

The GNWM explicitly prevents collapse using strictly linear computational complexity ($\mathcal{O}(N \cdot D)$) via the Four Pillars of Geometric Self-Organization:

Fixed Topological Convolution: Forces adjacent latent dimensions to activate together, creating a continuous "cortical sheet."

Collapse Prevention: Pulls the batch-mean towards a uniform center $C$, explicitly ensuring all neurons are utilized.

Geometric Winner-Take-All (WTA): Pulls the unnormalized batch mean away from the uniform center, forcing individual sample representations to become sparse and orthogonal.

Temporal Similarity: Anchors the predictor to true physical dynamics.

By forcing continuous embeddings toward the orthogonal vertices of a probability simplex, GNWM acts as a Topological Quantizer, discretizing continuous physics into a finite vocabulary of causal states. This unlocks absolute interpretability and drift-free, infinite-horizon counterfactual planning.

📂 Repository Structure & Experiments

The codebase is split into four standalone, highly reproducible empirical proofs. Every script generates its own toy physics dataset on-the-fly and outputs publication-ready matplotlib figures.

1. The Pathology of $\mathcal{L}_2$ Collapse

JEPA_Ablation_Study.py

What it does: Trains multiple architectural variants (Baseline, Batch-Normalized, Variance Penalty, Latent Diffusion) on a 2D physics dataset.

Proves: Standard $\mathcal{L}_2$ architectures plummet to zero latent variance (Dimensional Collapse). Batch Normalization acts as an unprincipled hack that fails physics anomaly detection (Gravity Reversal).

2. The Causal Galton Board

galton_board_experiment_2d.py

What it does: Simulates a chaotic bifurcation conditioned on an action vector (Left or Right). Implements a 10x10 SOM cortical sheet.

Proves: Baseline $\mathcal{L}_2$ models collapse diverging futures into a "Ghost Mean." GNWM correctly routes counterfactual futures to completely orthogonal topological neighborhoods.

3. High-Dimensional Multi-Object Chaos

causal_multi_object_som.py

What it does: Scales GNWM to a massive 30x30 topological grid ($D=900$). Evaluates multi-body continuous physics with localized causal routing via Convolutional Predictors (CoordConv).

Proves: GNWM achieves Judea Pearl's "Rung 2" of causation. By computing the Causal Delta between two imagined interventions, heavy static obstacles mathematically cancel out to $0.0$, demonstrating perfect hierarchical causal disentanglement.

4. The Topological Quantizer (Semantic Map & Drift-Free Rollouts)

topological_quantizer_demo.py

What it does: Trains a 15x15 GNWM ($D=225$) with a thermodynamic warmup schedule.

Proves: * Semantic Ontology: Extracts a human-readable 2D map of the environment's physical states directly from the latent "hot neurons."

Drift-Free Rollouts: Performs a 50-step autoregressive rollout where standard models drift to blurriness by step 15, but GNWM maintains a perfectly flat latent standard deviation by actively purging floating-point noise via grid-snapping.

🛠️ Installation & Usage

This codebase is designed to be extremely lightweight and fast. It does not require massive GPU clusters to reproduce the core theoretical findings.

Requirements

pip install torch torchvision opencv-python numpy matplotlib


Running the Experiments

Simply run any of the standalone Python scripts. The dataset will be generated on-the-fly, the models will train (typically < 2 minutes on a standard GPU), and the resultant evaluation plots and heatmaps will be saved as .png files in your working directory.

python topological_quantizer_demo.py


📝 Citation

If you find this codebase or theory useful in your research, please cite our manuscript:

@article{gnwm2026,
  title={Self-Supervised World Models: Breaking Dimensional Collapse via Geometric Self-Organization},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
