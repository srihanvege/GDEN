# GDEN — GPCR Diffusion Enhanced Network

**GDEN** is a generative AI framework for **target-conditioned molecular design**, focused on creating **bivalent antagonists** for **Class A GPCRs** (G protein–coupled receptors).  
The project combines **attention-based protein encoding**, **diffusion-based molecular generation**, and **reinforcement-guided optimization** to design new drug-like compounds tailored to specific GPCR pockets — such as the **β₂-adrenergic receptor (B2AR)**.

---

## Overview

**Goal:**  
To generate *target-specific* drug candidates by conditioning a molecular diffusion model on deep protein embeddings derived from **GPCRdb** sequence and structure data.

**Architecture Summary:**

| Stage | Description |
|:------|:-------------|
| **Protein Encoder** | Uses transformer or GNN attention layers (ESM, GAT, or GVP) to create embeddings of Class A GPCR binding pockets. |
| **Molecule Generator** | A conditional diffusion model (plus optional autoregressive transformer) generates molecular graphs or SELFIES strings conditioned on the target pocket embedding. |
| **Bivalent Constraints** | Generates two pharmacophore “warheads” and a linker satisfying geometric constraints predicted from the receptor structure. |
| **Evaluation Loop** | Filters generated compounds using drug-likeness (Lipinski, QED, SA) and QSAR property predictors; validates with docking scores (GNINA/Vina). |
| **RL Fine-Tuning** | Reinforces models using PPO/RLHF with composite rewards (affinity, selectivity, drug-likeness). |

---

## Key Features

- **Attention-based GPCR encoder** — integrates sequence and 3D residue-graph information.
- **Conditional diffusion generation** — molecular denoising conditioned on receptor embeddings.
- **Bivalent ligand synthesis** — dual-warhead generation with linker length/orientation prediction.
- **Multi-objective scoring** — QSAR activity prediction, docking, synthetic accessibility.
- **Reinforcement optimization** — PPO loop for guided molecule improvement.

---

## Conceptual Pipeline

```text
   GPCRdb → Protein Encoder → Target Embedding
             ↓
        Conditional Diffusion Model
             ↓
   Generated Molecules (Warheads + Linker)
             ↓
  Scoring → Docking → Reinforcement Optimization
