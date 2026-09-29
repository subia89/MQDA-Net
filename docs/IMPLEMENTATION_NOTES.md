# Implementation notes

This file lists every place where the paper leaves a detail open and the code has to choose. It also lists what the repository does not contain. Equation numbers count the paper's display equations in order of appearance: Eq. 1 is the model composition and Eq. 16 is L_txt / L_align.

## Encoder and decoder (§3.2–3.3)

- **Backbone.** The encoder is a MedNeXt-style network written from scratch in `blocks.py`, so no external MedNeXt package is needed. It uses five stages with base width 32 and kernel 3. The MedNeXt-L block counts are kept except at the bottleneck, where 2 blocks are used instead of 8, bringing the encoder to 28.6 M parameters (paper: 28.4 M). Both settings can be changed with `enc_blocks` and `enc_exp`.
- **MSASPP.** Follows Eqs. 3–4:
  - four dilated 3×3×3 branches (r = 1, 2, 4, 6) plus a GAP branch;
  - a Res2Net split-and-fuse over the five concatenated branches (y₁ = x₁, yᵢ = Kᵢ(xᵢ + yᵢ₋₁));
  - a 1×1×1 projection back to the bottleneck width.
- **Decoder.** Two MedNeXt-style U-Net branches with their own weights. Both use the same encoder skip connections. The block counts mirror MedNeXt-L at `[8, 8, 4, 3]`, for 31.4 M parameters (paper: 32.8 M).
- **DFCAM.**
  - Attached after decoder stages 3 and 4, the two highest-resolution stages.
  - Patch embedding is factorised as a 1×1 projection to d = 256 followed by p-sized average pooling. A dense p³-kernel convolution would cost about 52 M parameters at p = 16.
  - Patch sizes are 8 at stage 3 and 16 at stage 4, which gives 512 tokens at 128³ input.
  - The update MLP(LN(attention)) is projected back with a 1×1 convolution, upsampled, and added residually to the semantic features, as in Eq. 6.
- **Label convention.** The semantic branch predicts four mutually exclusive classes: background, NCR (necrotic / non-enhancing core), edema and ET. Evaluation uses the BraTS regions ET = {3}, TC = {1, 3} and WT = {1, 2, 3}.
  - BraTS 2021 stores ET as label 4; it is remapped to 3.
  - In BraTS 2024, label 4 (resection cavity) is mapped to background.
- **2D datasets.** FigShare and folder datasets have only binary tumour masks. These are written to label 1 (`mask_label`). Images without a mask (Br35H) contribute no segmentation loss.

## Segmentation loss (Eqs. 7–8)

- In Dice–CE, ε is also added to the numerator. A class missing from both the prediction and the target then scores Dice = 1 instead of 0; otherwise perfect predictions would still be penalised.
- **Boundary term.** ∂M is computed as a soft morphological gradient of the probability map (3×3(×3) max-pool dilation minus min-pool erosion), and the Dice-style overlap of Eq. 8 is taken between these soft boundaries.
- Weights are w_b = 0.8 and w_s = 1.0, as stated in the paper.

## Quantum classification head (§3.4)

- **Classifier input f_cls.** Masked average pooling of F5′ inside the soft NCR / ED / ET masks, plus radiomics per region:
  - log volume (cm³) and log surface area;
  - for each modality, the mean, standard deviation, skewness and log-kurtosis of intensity.

  The input has 3·512 + 3·18 = 1,590 dimensions for BraTS.
- **Projection.** An MLP maps 1,590 → 2,048 → 1,920 → 36, with LayerNorm, GELU and dropout 0.1. The outputs are squashed to angles with π·tanh. The whole head has 7.27 M parameters (paper: 7.3 M), of which 72 are circuit parameters.
- **Circuit.**
  - **Spatial register.** Qubits 0–2 are "position qubits" (Hadamard-prepared). Each gets H followed by RZ(π·c), where c is one coordinate of the soft tumour centroid, normalised to [−1, 1].
  - **Feature register.** Qubits 3–11 each get U3(θ, φ, λ) from consecutive triplets of the projected vector (27 angles).
  - **Controlled-U3 gates.** The remaining 9 angles drive three controlled-U3 gates, one from each spatial qubit to a feature qubit. Together, steps 2–3 use exactly 3·n_q = 36 angles, one triplet per qubit (Eq. 9).
  - **Controlled-Z ring.** A "recurrent" CZ ring (k, k+1 mod 12) then entangles the two registers.
  - **PQC layers.** Two layers, each made of 4 kernels on disjoint 3-qubit blocks: U3 on every qubit, then a CZ chain (Eq. 10). In the second layer the block partition is shifted by one qubit so information crosses kernel borders.
  - **Read-out.** The circuit returns ⟨X⟩ for every qubit, which is then passed through a linear layer and softmax (Eq. 11).
- **Gradients.**
  - `parameter-shift` implements the two-term shift rule (±π/2) for every U3 angle.
  - The angles feeding controlled-U3 gates use the four-term rule of Anselmetti et al. (2021), because the two-term rule is not exact for controlled rotations.
  - `backprop` differentiates the exact state vector directly. The test suite checks that both modes give the same gradients.
- **Back-ends.** The default is an exact batched PyTorch state-vector simulator with 4,096 amplitudes. `q_backend: pennylane` runs the same circuit on `default.qubit`; the tests check that the two agree to 1e-4.
- **Loss.** Focal cross-entropy with label smoothing, exactly as in Eq. 12, with γ = 2 and β = 0.1. Labels of −100 (a tumour whose type is unknown) are ignored.

## Knowledge graph (§3.5)

- **Nodes.**
  - one patient node;
  - one node per detected sub-region (NCR / ED / ET with ≥ `min_region_voxels` voxels), with features from the pooled region features and radiomics;
  - up to 8 atlas-location nodes, each carrying its overlap fraction and hemisphere;
  - all knowledge-base concept nodes, i.e. the "clinical attribute" nodes of Fig. 3;
  - symptom nodes found in the clinical notes by keyword matching (`kb.py`).
- **Edges.**
  - **Fixed edges** (weight 1): patient–region, region–location, atlas adjacency between locations, knowledge-base ontology edges (region→concept, location→concept, symptom→location) and patient–symptom.
  - **Learned edges.** Candidate pairs (patient / region / location × concept) are scored by an MLP on [h_u, h_v, W·z]. The sigmoid probability becomes a soft edge weight, `adj = max(fixed, learned)`, so the graph's structure depends on z for each case.
- **GAT.**
  - Three layers with four heads; the first two concatenate their heads and the last averages them.
  - Attention logits follow Eq. 13, e_gh = LeakyReLU(aᵀ[W h_g ‖ W h_h ‖ W_m z]), written as the equivalent sum of per-part projections, with log(edge weight) added.
  - Each layer has an ELU, a residual connection and LayerNorm for stability; these are not specified in the paper.
  - Read-out is g = mean ⊕ max over the valid nodes.
- **Atlas.**
  - `NiftiAtlas` intersects the predicted mask with any MNI-space label atlas. This requires the volume to be registered to MNI first (`scripts/preprocess.py --template`).
  - `CoarseLobeAtlas` is a dependency-free fallback that splits the volume box into hemisphere × lobe compartments by coordinate rules. It is only a rough approximation of anatomy. Use a real atlas for any reported localisation.
  - MNI coordinates of the centroid come from the atlas affine (NiftiAtlas) or from bounding-box scaling (coarse).

## Report generator (§3.6)

- **Q-Former.**
  - Uses the BLIP-2 Q-Former architecture from `transformers` (`Blip2QFormerModel`) with 4 layers, hidden size 768, cross-attention in every layer and encoder width 1408.
  - It has 44 learned queries: 32 visual, 8 graph and 4 quantum. The outputs are split into T_vis, T_graph and T_quant (Eq. 15) and projected to the LLM width.
  - Inputs are the F5′ tokens (512 at 128³), the GAT node tokens plus a pooled-graph token, and a z token, each with a stream embedding.
  - Weights with matching shapes, and the first 32 query tokens, are copied from `Salesforce/blip2-opt-2.7b`.
  - With these settings the projector has 46.0 M parameters (paper: 38.8 M). Setting `qformer_encoder_hidden: 768` brings it close to the paper, at the cost of fewer BLIP-2 tensors being reusable.
- **Language model.**
  - `MllamaForConditionalGeneration` (Llama 3.2 11B Vision-Instruct) is used through `inputs_embeds`. Its own vision tower is not used: the Q-Former tokens take its place, and the cross-attention layers are skipped when no image is given.
  - The prefix tokens are inserted right after BOS.
  - LoRA uses r = 16 and α = 32 on q/k/v/o of the 32 self-attention layers, about 13.6 M parameters (paper: 12.0 M; the exact target set is not stated).
- **Prompt.** A chain-of-thought instruction that enforces the five sections: Technique; Findings; Laterality and Lobe; Impression; BT-RADS. In addition to the token conditioning, the prompt text contains:
  - the measured volumes (ET, TC, WT, edema in cm³);
  - the enhancement ratio (ET / TC) and necrosis ratio (NCR / TC) — the paper does not define these ratios;
  - the atlas regions and MNI centroid;
  - the predicted tumour type with its confidence.
- **InfoNCE (Eq. 16).**
  - The visual representation is the mean of T_vis; the text representation is the mean of the LLM's last hidden state over the report tokens.
  - Both are projected to 256 dimensions and compared with cosine similarity at τ = 0.07, using in-batch negatives. As in the paper, the loss runs in one direction (visual → text).
- **DPO.**
  - `scripts/dpo_refine.py` uses the standard DPO loss with β = 0.1.
  - The reference policy is the adapter as loaded, with its log-probabilities cached.
  - Preference pairs can be given directly (chosen / rejected), or as candidates that RadGraph-F1 ranks against a reference report.
  - Only the LoRA weights are updated.

## Training (§3.7, Table 1)

- **Stages.** Table 1 gives separate epochs, learning rates and batch sizes for "seg + cls" and "LLM", so training runs in two stages with the same model code:
  1. `stage: vision` — L_seg + L_cls, 200 epochs, lr 1e-4, batch 4 (3D) / 32 (2D).
  2. `stage: joint` — all four losses, 50 epochs, batch 8, lr 1e-4 for vision / graph / Q-Former and 2e-5 for LoRA and the alignment heads. Everything is trained end to end in this stage.
- **Optimiser.** AdamW with weight decay 1e-5, gradient clipping at 1.0 and bf16. The cosine learning-rate schedule is not stated in the paper and can be turned off with `train.scheduler: none`.
- **Early stopping.** Patience is 20 epochs. By default the monitored value is mean Dice + accuracy − 0.1·val L_txt.
- **Cropping.** Training crops are centred on a random tumour voxel with probability 0.5, and on a random brain voxel otherwise. Evaluation uses a brain-centred crop, or sliding-window segmentation of the full volume (`--full-volume`, Gaussian weighting, 50 % overlap).
- **Augmentation.** Random affine transforms (±15°, scale 0.85–1.15), elastic deformation, intensity shift and Gaussian noise (σ = 0.05).
- **Graph construction.** By default, graphs during training use the predicted masks. `graph_from_ground_truth: true` uses the ground-truth masks instead.

## Not included

- Trained weights, and the 150 clinician-annotated reports or the 50 curated DPO pairs used in the paper.
- The exact Vox-MMSD BraTS 2024 split list. Place it under `splits/` if you have it.
- A skull-stripping tool. BraTS is already skull-stripped; for other data use e.g. HD-BET before `scripts/preprocess.py`.
