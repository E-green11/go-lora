# Beyond Decoupled PEFT: Geometry-Aware Low-Rank Adaptation via Riemannian Reparameterization

This repository is an anonymous code release for the paper:

**Beyond Decoupled PEFT: Geometry-Aware Low-Rank Adaptation via Riemannian Reparameterization**

## Abstract

Low-Rank Adaptation (LoRA) and its weight-decomposed variants dominate parameter-efficient fine-tuning, yet their directional update mechanisms remain geometrically mismatched to the normalized parameterization they impose. In existing decoupled PEFT methods, low-rank directional updates are computed in unconstrained Euclidean space and only then projected back onto a spherical manifold, which can introduce capacity redundancy and gradient squashing. To address these limitations, we propose **Geodesic Orthogonal Low-Rank Adaptation (GO-LoRA)**, a geometry-aware reparameterization framework for decoupled PEFT. GO-LoRA first applies orthogonal tangent projection to remove components parallel to the pretrained base direction, ensuring that the low-rank directional budget is devoted entirely to tangent-space motion. It then uses a geodesic flow based on the Riemannian Exponential Map to construct manifold-consistent directional updates on the hypersphere, improving geometric fidelity relative to Euclidean normalization while retaining standard optimization over the underlying low-rank parameters. Extensive experiments across language, vision, code generation, and multimodal reasoning tasks show that GO-LoRA consistently outperforms strong LoRA baselines while introducing no additional inference-time overhead after offline merging.

## Submission Status

This code accompanies a paper currently under review for **NeurIPS 2026**.
