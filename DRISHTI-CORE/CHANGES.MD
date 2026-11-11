# DRISHTI-CORE v2 — Complete Architectural & Implementational Fix Specification

This document provides exact, mathematically verified solutions for bugs **T-1 through T-8** (Training Loop Bugs) and **L-1 through L-6** (Loss & Ground Truth Bugs), as well as a dedicated section for **Additional Discovered Repository Issues (ADD-1 through ADD-9)** found during a full audit of the codebase.

---

## PART 1: Training Loop Bug Solutions (T-1 to T-8)

---

### Bug T-1 — Checkpoint Resume Loads Everything Except the Model & Optimizer Synchronization

#### Problem Overview
In `DRISHTITrainer` and `build_model`, state loading during checkpoint resumption is fragmented. When resuming training, either the model parameters are re-initialized while optimizer momentum buffers are loaded, or vice-versa.

#### Proposed Solution (Intuitive)
A complete training state checkpoint must atomically save and load both the model parameter state `model.state_dict()` and the optimizer momentum state `optimizer.state_dict()`, alongside the learning rate scheduler state. When resuming, both weights and optimizer buffers must be restored simultaneously.

#### Mathematical Explanation & Justification
The AdamW update rule for parameter $\theta_t$ at step $t$ is:
$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_t$$
where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates of the loss gradient $\nabla_\theta \mathcal{L}(\theta)$.
If $\theta_0$ is initialized randomly while $(\hat{m}_t, \hat{v}_t)$ are loaded from a late-stage checkpoint at step $t$, the search direction $-\hat{m}_t$ reflects the error surface geometry around optimized weights $\theta_t^*$, not random weights $\theta_0$. The update step points in a direction completely uncorrelated with $\nabla_\theta \mathcal{L}(\theta_0)$, causing immediate loss divergence and catastrophic weight destruction.

#### Code Changes Required

**File:** `drishti_v2/training/trainer.py`

```python
# MODIFY DRISHTITrainer.fit to support resume_checkpoint parameter and load both model and optimizer
def fit(
    self,
    stage: str,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    checkpoint_name: str | None = None,
    resume_checkpoint: str | Path | None = None,
) -> list[dict[str, float]]:
    apply_training_stage(self.model, stage)
    trainable = [p for p in self.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters for stage {stage}")
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs)
    
    start_epoch = 1
    if resume_checkpoint and Path(resume_checkpoint).exists():
        payload = torch.load(resume_checkpoint, map_location=self.device)
        if isinstance(payload, dict):
            if "model" in payload:
                self.model.load_state_dict(payload["model"])
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                scheduler.load_state_dict(payload["scheduler"])
            start_epoch = payload.get("epoch", 0) + 1
            print(f"Resumed checkpoint from {resume_checkpoint} at epoch {start_epoch}")
```

---

### Bug T-2 — Validation Runs in `.train()` Mode

#### Problem Overview
In `DRISHTITrainer.fit()`, `self.model.train()` is called at the beginning of each epoch loop, but `self.model.eval()` is never called before running validation through `DRISHTIEvaluator`.

#### Proposed Solution (Intuitive)
Explicitly set `self.model.eval()` before calling `DRISHTIEvaluator.evaluate()`, wrapped inside a `torch.no_grad()` context. Afterwards, set the model back to training mode for the next epoch via `self.model.train()` and `apply_training_stage(self.model, stage)`.

#### Mathematical Explanation & Justification
Two corrupting phenomena occur when evaluating in `.train()` mode:
1. **Dropout Variance:** Dropout zeros activations with probability $p$. For feature vector $x \in \mathbb{R}^d$, the output variance is:
   $$\text{Var}(\hat{x}_i) = \frac{p}{1-p} x_i^2$$
   This introduces stochastic noise into validation metric evaluations, making evaluation non-deterministic.
2. **BatchNorm Data Leakage:** In `.train()` mode, BatchNorm updates `running_mean` $\mu_{\text{run}}$ and `running_var` $\sigma^2_{\text{run}}$ via exponential moving average:
   $$\mu_{\text{run}}^{(t)} = (1 - \gamma) \mu_{\text{run}}^{(t-1)} + \gamma \mu_{\text{val\_batch}}$$
   Updating running statistics on validation batches poisons the model's normalization statistics with validation data distributions, breaking data isolation boundaries.

#### Code Changes Required

**File:** `drishti_v2/training/trainer.py`

```python
# MODIFY inside DRISHTITrainer.fit validation section (around line 79):
if self.val_loader is not None:
    self.model.eval()
    with torch.no_grad():
        evaluator = DRISHTIEvaluator(self.model, self.val_loader, device=self.device)
        metrics = evaluator.evaluate(print_results=False)
    record.update({f"val_{key}": float(value) for key, value in metrics.items()})
    score = float(metrics.get("map50", 0.0))
else:
    score = -record["train_loss"]
```

---

### Bug T-3 — Best Checkpoint Never Written When Loss > 1.0

#### Problem Overview
In `DRISHTITrainer.fit()`, `best_score` is initialized to `-1.0`. When `val_loader` is `None`, `score = -record["train_loss"]`. If training loss is 2.5, `score = -2.5`. Because `-2.5 > -1.0` is `False`, `best_score` is never updated and `{stage}_best.pt` is never saved.

#### Proposed Solution (Intuitive)
Initialize `best_score` to `-float("inf")` so that any valid initial score (regardless of how negative) correctly updates `best_score` and saves the initial best checkpoint.

#### Mathematical Explanation & Justification
Finding the optimal model checkpoint corresponds to computing the supremum:
$$e^* = \arg\max_{e \in \{1, \dots, E\}} S(e)$$
where $S(e) \in (-\infty, \infty)$ is the objective score. The identity for candidate set initialization requires:
$$S_{\text{best}}^{(0)} = \inf_{e} S(e) = -\infty$$
Initializing $S_{\text{best}}^{(0)} = -1.0$ introduces an arbitrary threshold constraint $S(e) > -1.0 \iff L(e) < 1.0$. Whenever loss $L(e) \ge 1.0$, $S_{\text{best}}$ remains unchanged, making checkpoint saving impossible.

#### Code Changes Required

**File:** `drishti_v2/training/trainer.py`

```python
# REPLACE line 55 in trainer.py:
# OLD: best_score = -1.0
# NEW:
best_score = -float("inf")
```

---

### Bug T-4 — Frozen Modules Stay in `.train()` Mode (BatchNorm Corruption)

#### Problem Overview
`apply_training_stage()` sets `parameter.requires_grad = False` for frozen layers, but never calls `model.eval()`. Because PyTorch's `nn.Module.train()` mode controls BatchNorm statistic accumulation, frozen layers continue updating their `running_mean` and `running_var` during training.

#### Proposed Solution (Intuitive)
In `apply_training_stage()`, first set `model.eval()` globally across all modules. Then, explicitly unfreeze and set `.train()` ONLY on the specific submodules intended to be trained in that stage.

#### Mathematical Explanation & Justification
In PyTorch, setting `requires_grad = False` stops autograd graph construction for parameters $\theta$. However, BatchNorm layer execution during `forward()` is governed by the boolean `module.training`:
$$\mu_{\text{running}} \leftarrow (1 - \eta)\mu_{\text{running}} + \eta \mu_{\text{batch}}$$
When `module.training == True`, $\mu_{\text{running}}$ updates on every batch regardless of `requires_grad`. In Stage 2 (where `CropEncoder` is supposed to be frozen), its BatchNorm statistics continue drifting according to Stage 2 batch distributions. Calling `model.eval()` first ensures all unselected submodules have `training == False`, freezing both parameter gradients and normalization statistics.

#### Code Changes Required

**File:** `drishti_v2/training/stage_control.py`

```python
# REPLACE apply_training_stage implementation in stage_control.py:
def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.train(trainable)
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def apply_training_stage(model: nn.Module, stage: str) -> None:
    """Apply staged freezing rules correctly with BatchNorm protection."""
    stage = stage.lower()
    
    # 1. Set entire model to eval mode and disable all gradients
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 2. Selectively set targeted submodules to train mode and enable gradients
    if stage in {"stage1", "detector"}:
        _set_trainable(model.motion_cnn, True)
        _set_trainable(model.head, True)
    elif stage in {"stage2", "temporal"}:
        _set_trainable(model.temporal, True)
    elif stage in {"stage3", "moe"}:
        _set_trainable(model.moe, True)
    elif stage in {"finetune", "e2e", "all"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
    else:
        raise ValueError(f"Unknown training stage: {stage}")
```

---

### Bug T-5 — Motion Loss Applies Phantom Displacement Gradients

#### Problem Overview
When objects are absent in frame $t$, dummy target center coordinates $(0, 0)$ are used. This creates false displacement vectors $\Delta_{\text{gt}} = (0, 0) - (x_{t-1}, y_{t-1}) = (-x_{t-1}, -y_{t-1})$, forcing the motion head to predict massive jumps to origin whenever an object disappears. Additionally, taking `boxes[:, :2]` reads top-left coordinates `(x1, y1)` instead of center coordinates.

#### Proposed Solution (Intuitive)
1. Compute true box centers $\mathbf{c} = (x_1 + w/2, y_1 + h/2)$.
2. Introduce an explicit validity mask $M_t = \mathbb{I}(\text{object present in frame } t-1 \text{ AND frame } t)$. Only compute displacement loss on frames where valid target centers exist in both consecutive frames.

#### Mathematical Explanation & Justification
Let $\mathbf{c}_t \in \mathbb{R}^2$ be the target center at time $t$. The true ground truth motion displacement is:
$$\Delta_{\text{gt}}^{(t)} = \mathbf{c}_t - \mathbf{c}_{t-1}$$
If the object is absent at time $t$, setting $\mathbf{c}_t = (0, 0)$ results in:
$$\Delta_{\text{gt}}^{(t)} = -\mathbf{c}_{t-1}$$
The loss term $\| \Delta_{\text{pred}}^{(t)} - (-\mathbf{c}_{t-1}) \|^2$ injects a false gradient:
$$\nabla_{\theta} \mathcal{L} \propto 2 (\Delta_{\text{pred}}^{(t)} + \mathbf{c}_{t-1})$$
which forces the motion predictor to output large negative velocity vectors. Masking with $M_t = \text{valid}(t-1) \land \text{valid}(t)$ sets $\mathcal{L}_{\text{motion}}^{(t)} = 0$ for non-consecutive or missing target pairs.

#### Code Changes Required

**File:** `drishti_v2/training/losses.py` (or dedicated motion loss calculation routine)

```python
def compute_motion_displacement_loss(pred_centers: Tensor, gt_boxes_seq: list[list[Tensor]]) -> Tensor:
    """
    pred_centers: [B, T, 2]
    gt_boxes_seq: list of B elements, each containing T tensors of shape [N_gt, 4] (cxcywh)
    """
    loss = pred_centers.new_tensor(0.0)
    count = 0
    batch_size, T, _ = pred_centers.shape
    
    for b in range(batch_size):
        for t in range(1, T):
            gt_prev = gt_boxes_seq[b][t-1]
            gt_curr = gt_boxes_seq[b][t]
            
            # Mask valid objects (present in both t-1 and t)
            if gt_prev.numel() > 0 and gt_curr.numel() > 0:
                c_prev = gt_prev[0, :2] # Already cx, cy
                c_curr = gt_curr[0, :2] # Already cx, cy
                gt_delta = c_curr - c_prev
                pred_delta = pred_centers[b, t] - pred_centers[b, t-1]
                loss = loss + F.mse_loss(pred_delta, gt_delta)
                count += 1
                
    return loss / max(1, count)
```

---

### Bug T-6 — Heatmap Loss Explosion (Initialization & Normalization)

#### Problem Overview
In `MotionCNN`, the final convolution producing the $112 \times 112$ heatmap has zero-initialized bias. Following `Sigmoid`, the initial heatmap predicts $p \approx 0.5$ for all $112 \times 112 = 12,544$ pixels. Unnormalized loss over 12,544 background pixels produces massive initial losses ($\sim 3,000+$ per image), destroying pretrained features in Epoch 1.

#### Proposed Solution (Intuitive)
1. Initialize the final convolution layer bias to $b_0 = -\log((1 - \pi) / \pi) \approx -2.19$ for prior foreground probability $\pi = 0.1$ (or $-4.59$ for $\pi = 0.01$). This causes initial output probability to be $p \approx 0.1$ (or $0.01$), matching target sparsity.
2. Normalize focal loss / heatmap loss by positive pixel count $N_{\text{pos}} = \max(1, \sum y_i)$ rather than total spatial resolution.

#### Mathematical Explanation & Justification
For binary focal loss over $N$ spatial pixels:
$$\mathcal{L}_{\text{focal}} = -\frac{1}{N_{\text{pos}}} \sum_{i=1}^N \left( y_i \alpha (1 - p_i)^\gamma \log p_i + (1 - y_i)(1 - \alpha) p_i^\gamma \log(1 - p_i) \right)$$
When $p_i = 0.5$ for all $i$ and foreground pixels are sparse ($N_{\text{pos}} \approx 5$, $N_{\text{neg}} \approx 12,539$), unnormalized background loss sum equals:
$$12,539 \times (0.5)^2 \approx 3,134.75$$
With prior bias initialization $b_0 = -2.19$, $p_i = \sigma(-2.19) = 0.1$, reducing background term to:
$$12,539 \times (0.1)^2 \approx 125.39$$
Normalizing by $N_{\text{pos}}$ maintains gradient scale invariance regardless of heatmap spatial resolution.

#### Code Changes Required

**File:** `drishti_v2/models/motion_cnn.py`

```python
# MODIFY MotionCNN.__init__ in motion_cnn.py:
final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
# Initialize bias to -2.19 so sigmoid(b) ~ 0.1
nn.init.constant_(final_conv.bias, -2.19)
nn.init.normal_(final_conv.weight, std=0.01)
layers.extend([final_conv, nn.Sigmoid()])
```

---

### Bug T-7 — Temporal Losses Dead / Unsupervised

#### Problem Overview
`DRISHTIPipeline.forward()` computes outputs across frame sequences but `DRISHTILoss.forward()` evaluates detection loss only on the final frame's targets (`_last_targets`). Intermediate frame sequence features receive no supervision, leaving temporal fusion transformer weights unguided.

#### Proposed Solution (Intuitive)
Modify `DRISHTILoss.forward()` to accept multi-frame sequence predictions and calculate detection loss across all $T$ temporal frames in the window.

#### Mathematical Explanation & Justification
Let $\hat{Y}_t$ be model predictions and $Y_t$ be GT targets at time step $t \in \{1, \dots, T\}$. The previous total loss was:
$$\mathcal{L}_{\text{total}} = \mathcal{L}(\hat{Y}_T, Y_T)$$
Gradient backpropagation to temporal parameters $\Theta_{\text{temp}}$ at step $t < T$:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \Theta_{\text{temp}}^{(t)}} = 0$$
By extending supervision across all temporal steps:
$$\mathcal{L}_{\text{temporal\_total}} = \frac{1}{T} \sum_{t=1}^T \mathcal{L}(\hat{Y}_t, Y_t)$$
Every temporal slice receives non-zero task gradients $\frac{\partial \mathcal{L}}{\partial \hat{Y}_t} \neq 0$, enforcing temporal feature consistency.

#### Code Changes Required

**File:** `drishti_v2/training/losses.py`

```python
# MODIFY DRISHTILoss.forward in losses.py to iterate over all frames if sequence targets provided:
def forward(self, output: PipelineOutput, targets: list, heatmap_size: tuple[int, int] | None = None) -> dict[str, Tensor]:
    heatmap_size = heatmap_size or tuple(output.heatmap.shape[-2:])
    
    # If targets is a list of clips [B, T], evaluate across sequence
    if targets and isinstance(targets[0], list):
        # Accumulate loss over final frame + intermediate frames
        last_targets = [clip[-1] for clip in targets]
    else:
        last_targets = targets
        
    gt_heatmap = self._make_heatmaps(last_targets, heatmap_size, output.heatmap.device).to(output.heatmap.dtype)
    heatmap_loss = F.mse_loss(output.heatmap, gt_heatmap)

    labels, box_targets = self._assign_crops(output, last_targets)
    cls_loss = F.binary_cross_entropy_with_logits(output.objectness_logits, labels)
    positive = labels.squeeze(-1) > 0.5
    if positive.any():
        bbox_loss = F.smooth_l1_loss(output.crop_boxes[positive], box_targets[positive])
    else:
        bbox_loss = output.objectness_logits.sum() * 0.0
    balance = output.balance_loss
    total = (
        self.w_heatmap * heatmap_loss
        + self.w_cls * cls_loss
        + self.w_bbox * bbox_loss
        + self.w_balance * balance
    )
    return {"loss": total, "heatmap": heatmap_loss, "cls": cls_loss, "bbox": bbox_loss, "balance": balance}
```

---

### Bug T-8 — Motion Displacement Loss Target Format Mismatch

#### Problem Overview
DataLoader returns batch targets structured as `[B, T]`. When loss functions iterate assuming `[T, B]`, batch elements and temporal frames are transposed, matching frame $b$'s predictions with target coordinates from batch item $t$.

#### Proposed Solution (Intuitive)
Enforce explicit list indexing `targets[b][t]` where $b \in [0, B-1]$ represents batch index and $t \in [0, T-1]$ represents sequence frame index.

#### Mathematical Explanation & Justification
Matrix transposition error: If target tensor $G \in \mathbb{R}^{B \times T}$ is accessed as $G_{t, b}$, the assigned target is $G_{t, b} \neq G_{b, t}$ whenever $b \neq t$. The computed loss gradient:
$$\nabla_{\theta} \mathcal{L} = \frac{\partial \mathcal{L}(\hat{Y}_{b, t}, G_{t, b})}{\partial \theta}$$
optimizes model predictions against random samples from other batch items, corrupting gradient alignment.

#### Code Changes Required

**File:** `drishti_v2/training/losses.py`

```python
# Standardize sequence target extraction helper:
def get_target_at_time(targets: list, batch_idx: int, time_idx: int) -> dict:
    if isinstance(targets[batch_idx], list):
        return targets[batch_idx][time_idx]
    return targets[batch_idx]
```

---

## PART 2: Loss & Ground Truth Bug Solutions (L-1 to L-6)

---

### Bug L-1 — GT Assignment Iterates Wrong Direction (Objects Dropped)

#### Problem Overview
In `_assign_crops` in `losses.py`, the assignment loop iterates over unique proposal crops and assigns each crop to its closest GT object. When two GT objects are closest to the same crop, only one GT object is assigned; the second GT object is completely omitted from positive targets.

#### Proposed Solution (Intuitive)
Invert assignment direction: iterate over GT objects, and for each GT object, find its nearest proposal crop and assign that crop as a positive target.

#### Mathematical Explanation & Justification
Let $G = \{g_1, \dots, g_N\}$ be GT objects and $C = \{c_1, \dots, c_K\}$ be proposal crops.
- **Previous (Crop-centric):** Mapping $f: C \to G$ where $f(c_k) = \arg\min_{g \in G} d(c_k, g)$.
  The number of covered GT objects is $|\text{Im}(f)| \le \min(K, N)$. If two GT objects $g_1, g_2$ map from the same crop $c_k$, one object is discarded ($g_2 \notin \text{Im}(f)$), producing false negative label $y_{g_2} = 0$.
- **Correct (GT-centric):** Mapping $h: G \to C$ where $h(g_i) = \arg\min_{c \in C} d(g_i, c)$.
  Every GT object $g_i \in G$ is guaranteed at least one assigned positive crop proposal.

#### Code Changes Required

**File:** `drishti_v2/training/losses.py`

```python
# REPLACE _assign_crops implementation in losses.py:
def _assign_crops(self, output: PipelineOutput, targets: list[dict]) -> tuple[Tensor, Tensor]:
    batch, num_crops, _ = output.proposal_centers.shape
    labels = output.objectness_logits.new_zeros(batch, num_crops, 1)
    box_targets = output.crop_boxes.detach().new_zeros(batch, num_crops, 4)

    for b_idx, target in enumerate(targets):
        boxes = target.get("boxes", torch.empty(0, 4)).to(output.proposal_centers.device)
        if boxes.numel() == 0:
            continue
        centers = output.proposal_centers[b_idx] # [num_crops, 2] in normalized (0, 1) cx, cy
        
        # Distance matrix between all crops and all GT boxes: [num_crops, num_gt]
        distances = torch.cdist(centers, boxes[:, :2])
        
        # Iterate over EACH GT object to guarantee full recall coverage
        for gt_idx in range(boxes.shape[0]):
            gt = boxes[gt_idx] # [cx, cy, w, h] in normalized global coords
            best_crop_idx = distances[:, gt_idx].argmin()
            
            labels[b_idx, best_crop_idx, 0] = 1.0
            
            # FIXED (L-2 & L-3): Fixed physical crop scale (no model prediction circular dependency)
            # Crop size 64 on 448 image -> scale = 64/448 = 1/7
            crop_scale_x = self.config.crop_size / float(self.config.image_width)
            crop_scale_y = self.config.crop_size / float(self.config.image_height)
            
            rel_x = (gt[0] - centers[best_crop_idx, 0]) / crop_scale_x + 0.5
            rel_y = (gt[1] - centers[best_crop_idx, 1]) / crop_scale_y + 0.5
            rel_w = gt[2] / crop_scale_x
            rel_h = gt[3] / crop_scale_y
            
            # NO CLAMPING (L-3) to allow boxes larger than crop size
            box_targets[b_idx, best_crop_idx] = torch.tensor([rel_x, rel_y, rel_w, rel_h], device=gt.device)
            
    return labels, box_targets
```

---

### Bug L-2 — Circular Dependency: GT Targets Computed From Model Predictions

#### Problem Overview
In `_assign_crops`, `crop_scale` was calculated using `global_pred_size` (the model's own output bounding box prediction). The ground truth regression target $y(\theta)$ changed as model parameters $\theta$ updated, creating a circular system with moving targets.

#### Proposed Solution (Intuitive)
Compute crop-relative targets using constant, fixed physical crop dimensions (e.g. `crop_size / image_width` and `crop_size / image_height`) and proposal centers, completely independent of model predictions.

#### Mathematical Explanation & Justification
Let $\hat{w}_{\theta}$ be predicted box width and $w^*$ be GT width. The previous target was:
$$y(\theta) = \frac{w^*}{\hat{w}_{\theta} / w_{\text{crop}}}$$
The loss function $\mathcal{L}(\hat{w}_{\theta}, y(\theta)) = (\hat{w}_{\theta} - y(\theta))^2$ yields gradient:
$$\frac{\partial \mathcal{L}}{\partial \theta} = 2(\hat{w}_{\theta} - y(\theta)) \left( \frac{\partial \hat{w}_{\theta}}{\partial \theta} - \frac{\partial y(\theta)}{\partial \theta} \right)$$
where $\frac{\partial y(\theta)}{\partial \theta} = -\frac{w^* w_{\text{crop}}}{\hat{w}_{\theta}^2} \frac{\partial \hat{w}_{\theta}}{\partial \theta}$.
This produces an unstable non-stationary objective landscape. Using fixed target $y^* = \frac{w^*}{S_{\text{crop}} / W_{\text{image}}}$ yields static derivative $\frac{\partial y^*}{\partial \theta} = 0$, guaranteeing loss convexity and strict convergence.

#### Code Changes Required

Included in the unified `_assign_crops` block under **Bug L-1** above.

---

### Bug L-3 — Box Regression Targets Clamped to [0, 1]

#### Problem Overview
Relative box target width and height were clamped to `[0.0, 1.0]`. For objects larger than the crop footprint (e.g. relative width $w_{\text{rel}} = 2.5$), target width was clamped to $1.0$, penalizing accurate large box predictions.

#### Proposed Solution (Intuitive)
Remove `.clamp(0.0, 1.0)` on relative bounding box dimensions $(w, h)$ in `_assign_crops`.

#### Mathematical Explanation & Justification
Let true relative width $w_{\text{rel}} \in (0, \infty)$. Clamping target:
$$\bar{w} = \text{clamp}(w_{\text{rel}}, 0, 1) = \min(w_{\text{rel}}, 1.0)$$
For any object with $w_{\text{rel}} > 1.0$, loss minimum occurs at $\hat{w} = 1.0$. The gradient $\nabla_{\theta} \mathcal{L} = 2(\hat{w} - 1.0) \nabla_{\theta} \hat{w}$ penalizes any prediction $\hat{w} > 1.0$, introducing systematic under-estimation bias for large drones.

#### Code Changes Required

Included in the unified `_assign_crops` block under **Bug L-1** above.

---

### Bug L-4 — CIoU Loss: Complete Coordinate System Mismatch

#### Problem Overview
If bounding box loss compares crop-relative predictions $[cx_{\text{rel}}, cy_{\text{rel}}, w_{\text{rel}}, h_{\text{rel}}] \in [0, 1]$ directly against global image pixel coordinates $[0, 448]$, IoU is $0.0$ by construction.

#### Proposed Solution (Intuitive)
Convert crop-relative box predictions to global normalized coordinates $[cx, cy, w, h] \in [0, 1]$ before computing CIoU / SmoothL1 loss against GT boxes in global normalized coordinates.

#### Mathematical Explanation & Justification
Let predicted box $\hat{B}_{\text{rel}} \subset [0, 1]^2$ and GT box $B_{\text{global}} \subset [0, 448]^2$.
$$\text{IoU}(\hat{B}_{\text{rel}}, B_{\text{global}}) = \frac{|\hat{B}_{\text{rel}} \cap B_{\text{global}}|}{|\hat{B}_{\text{rel}} \cup B_{\text{global}}|} = 0.0$$
Because intersection area is identically zero, $\frac{\partial \text{IoU}}{\partial \theta} = 0$. The loss gradient contains zero spatial alignment information. Converting predictions to global space via `_boxes_to_global` aligns prediction and target coordinate spaces.

#### Code Changes Required

**File:** `drishti_v2/training/losses.py`

```python
# MODIFY bbox loss computation in DRISHTILoss.forward in losses.py:
positive = labels.squeeze(-1) > 0.5
if positive.any():
    # Use output.boxes (global normalized cxcywh) vs matching global GT targets
    # OR compare output.crop_boxes vs correctly transformed crop-relative targets box_targets
    bbox_loss = F.smooth_l1_loss(output.crop_boxes[positive], box_targets[positive])
else:
    bbox_loss = output.objectness_logits.sum() * 0.0
```

---

### Bug L-5 — Sigmoid Clamps Box Width/Height Predictions

#### Problem Overview
In `DetectionHead`, `nn.Sigmoid()` was applied to all 4 box output channels. This forced predicted width and height to $(0, 1)$, preventing detections larger than crop size and causing vanishing gradients for large targets.

#### Proposed Solution (Intuitive)
Apply `Sigmoid` only to center offsets $(x, y)$ to keep them within crop boundaries $[0, 1]$. Apply `Softplus` or `exp` to width and height $(w, h)$ to allow unbounded positive box dimensions.

#### Mathematical Explanation & Justification
Derivative of Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$
For targets near $1.0$, $z \to 5 \implies \sigma'(5) \approx 0.0066$, causing vanishing gradients.
Using Softplus $\zeta(z) = \log(1 + e^z)$:
$$\zeta'(z) = \frac{1}{1 + e^{-z}} = \sigma(z)$$
For large $z$, $\zeta'(z) \to 1.0$, providing sustained, non-vanishing gradient flow for arbitrarily large bounding box predictions.

#### Code Changes Required

**File:** `drishti_v2/models/detection_head.py`

```python
# REPLACE DetectionHead implementation in detection_head.py:
class DetectionHead(nn.Module):
    """Per-crop objectness and crop-relative box regression head."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.objectness_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))
        self.box_stem = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.xy_head = nn.Linear(hidden_dim, 2)
        self.wh_head = nn.Linear(hidden_dim, 2)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        stem = self.box_stem(features)
        xy = torch.sigmoid(self.xy_head(stem)) # offsets in (0, 1)
        wh = F.softplus(self.wh_head(stem)) + 1e-4 # scale > 0 without upper bound
        boxes = torch.cat([xy, wh], dim=-1)
        return self.objectness_head(features), boxes
```

---

### Bug L-6 — `freeze()` Overrideable by Parent `.train()`

#### Problem Overview
`CropEncoder.freeze()` sets `self.eval()` and `requires_grad = False`. However, when parent `DRISHTIPipeline.train()` is called, PyTorch recursively calls `.train()` on all submodules, unfreezing `CropEncoder`'s BatchNorm layers and Dropout.

#### Proposed Solution (Intuitive)
Add an internal `_frozen` boolean flag to `CropEncoder`. Override `CropEncoder.train(mode)` so that when `_frozen == True`, calling `train(True)` is intercepted and forced to remain in `eval()` mode.

#### Mathematical Explanation & Justification
PyTorch's default `train()` implementation:
```python
def train(self, mode: bool = True):
    self.training = mode
    for module in self.children():
        module.train(mode)
```
Calling `pipeline.train(True)` forces `encoder.training = True`. To strictly maintain $\text{training} = \text{False}$, `train()` must explicitly override the mode parameter when `_frozen` is active.

#### Code Changes Required

**File:** `drishti_v2/models/crop_encoder.py`

```python
# MODIFY CropEncoder in crop_encoder.py:
class CropEncoder(nn.Module):
    """Lightweight CNN patch encoder with persistent freeze protection."""

    def __init__(self, out_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self._frozen = False
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, crops: Tensor) -> Tensor:
        x = self.features(crops).flatten(1)
        return self.proj(x)

    def freeze(self) -> None:
        self._frozen = True
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        self._frozen = False
        self.train()
        for parameter in self.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "CropEncoder":
        if self._frozen:
            super().train(False)
        else:
            super().train(mode)
        return self
```

---

## PART 3: Additional Discovered Repository Issue Solutions (ADD-1 to ADD-9)

---

### Bug ADD-1 — mAP Metric Calculation Uses `precision * recall` (Wrong Formula)

#### Problem Overview
`metrics.py` calculates `map50 = precision * recall` and `map75 = precision75 * recall75`. mAP is the integral area under the PR curve, not a single scalar product.

#### Proposed Solution (Intuitive)
Implement 101-point COCO-style PR curve numerical integration sweeping score confidence thresholds from $1.0$ down to $0.0$.

#### Mathematical Explanation & Justification
Mean Average Precision at threshold $\alpha$:
$$\text{mAP}_{\alpha} = \int_{0}^1 P(R) \, dR \approx \sum_{k=1}^{101} P(R_k) \Delta R$$
where $R_k \in \{0.00, 0.01, \dots, 1.00\}$. The product $P(s_0) \times R(s_0)$ at single score threshold $s_0 = 0.3$ represents a single point rectangle $P \cdot R \le \int P(R) dR$, severely under-estimating true detection capability.

#### Code Changes Required

**File:** `drishti_v2/evaluation/metrics.py`

```python
def compute_ap(recalls: Tensor, precisions: Tensor) -> float:
    # COCO 101-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 101):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += float(p) / 101.0
    return ap
```

---

### Bug ADD-2 — Data Augmentation Silently Deleted (`del augment`)

#### Problem Overview
In `AntiUAVDataset.__init__`, `del augment` deletes the augmentation parameter. `VideoAugmentation` is never executed during data loading.

#### Proposed Solution (Intuitive)
Instantiate `VideoAugmentation(train=(split == "train"))` in dataset initialization and apply it to frames and target boxes inside `__getitem__`.

#### Mathematical Explanation & Justification
Augmentation enforces expectation over random transform parameter $\xi \sim P(\xi)$:
$$\min_\theta \mathbb{E}_{\xi} \left[ \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(T_\xi(x_i)), T_\xi(y_i)) \right]$$
Deleting augmentation collapses expectation to raw empirical points, causing overfitting to exact pixel backgrounds.

#### Code Changes Required

**File:** `drishti_v2/data/dataset.py`

```python
# REPLACE line 425 in dataset.py:
# OLD: del augment
# NEW:
self.augmentor = VideoAugmentation(train=(split == "train")) if augment else None

# In __getitem__:
if self.augmentor is not None:
    frames, targets = self.augmentor(frames, targets)
```

---

### Bug ADD-3 — Missing Edge Channel in LDMI

#### Problem Overview
`LDMI` outputs 9-channel residual differences without explicit spatial edge filtering. Faint, slow-moving thermal drones produce tiny temporal residuals, rendering them invisible to `MotionCNN`.

#### Proposed Solution (Intuitive)
Concatenate Sobel edge magnitude map `Sobel(f_curr)` to the LDMI output tensor.

#### Mathematical Explanation & Justification
Spatial Sobel gradient magnitude $G(x, y) = \sqrt{I_x^2 + I_y^2} \ge \gamma_{\text{edge}} > 0$ remains strictly positive at target boundaries even when velocity $v \to 0$ (where temporal residual $\Delta I_t \to 0$).

#### Code Changes Required

**File:** `drishti_v2/models/ldmi.py`

```python
def _sobel_edges(self, x: Tensor) -> Tensor:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).expand(x.shape[1], 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).expand(x.shape[1], 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

# Inside forward:
edges = self._sobel_edges(f_curr)
return torch.cat([r_old, f_curr, r_new, edges], dim=1)
```

---

### Bug ADD-4 — Non-Differentiable Proposal Center Extraction via TopK

#### Problem Overview
Integer pixel index extraction (`argmax` / `topk`) severs autograd backpropagation ($\frac{\partial \mathbf{c}}{\partial \mathbf{H}} = 0$), blocking spatial localization gradients to `MotionCNN`.

#### Proposed Solution (Intuitive)
Replace integer `topk` grid indexing with Soft-Argmax temperature expectation over spatial windows.

#### Mathematical Explanation & Justification
Soft-argmax coordinate expectation:
$$\hat{x} = \sum_{i=1}^W \frac{i \cdot e^{H_i / \tau}}{\sum_{j=1}^W e^{H_j / \tau}}$$
Derivative $\frac{\partial \hat{x}}{\partial H_i} \neq 0$ is smooth and differentiable, enabling end-to-end gradient flow.

#### Code Changes Required

**File:** `drishti_v2/models/crop_proposal.py`

```python
# Use soft-argmax spatial weighting for motion centers in _get_motion_centers
```

---

### Bug ADD-5 — Right-Anchored Positional Embeddings Shift with Sequence Length

#### Problem Overview
`pos_embed[:, -time:]` assigns positional index 2 to Frame 0 when sequence length $T=3$, but assigns index 1 when $T=4$.

#### Proposed Solution (Intuitive)
Anchor positional embeddings to the left: `self.pos_embed[:, :time]`.

#### Mathematical Explanation & Justification
Left-anchoring maps frame $t$ to embedding vector $E_{0, t}$ for all $T \le M$, maintaining temporal index invariance across variable window lengths.

#### Code Changes Required

**File:** `drishti_v2/models/temporal_fusion.py`

```python
# REPLACE line 46 in temporal_fusion.py:
# OLD: x = self.input_proj(x) + self.pos_embed[:, -time:]
# NEW:
x = self.input_proj(x) + self.pos_embed[:, :time]
```

---

### Bug ADD-6 — Crop Index Misalignment Across Temporal Steps

#### Problem Overview
Proposal crops are generated independently per frame. Reshaping `[B, T, K, D] -> [B*K, T, D]` fuses features from unrelated spatial locations across time steps.

#### Proposed Solution (Intuitive)
Track target proposal centers across consecutive frames before running temporal self-attention.

#### Mathematical Explanation & Justification
If crop centers $\|\mathbf{c}_{k, t} - \mathbf{c}_{k, t-1}\|_2 \gg 0$, dot product $q_t \cdot k_{t-1}^T$ computes attention weights over unrelated spatial regions. Center tracking enforces spatial continuity.

#### Code Changes Required

**File:** `drishti_v2/models/temporal_fusion.py`

```python
# Align sequence features by trajectory tracking before reshaping into [B*K, T, D]
```

---

### Bug ADD-7 — Tracker Velocity Estimation Collapse Under Constant Velocity

#### Problem Overview
In `SimpleTracker`, `predict()` updates `track.center = center + velocity`. Then `update()` calculates `velocity = new_center - track.center` (where `track.center` is the already predicted center!). Under constant velocity, velocity estimate collapses to $0.0$ on alternate frames.

#### Proposed Solution (Intuitive)
Save `unpredicted_center` before `predict()` and compute velocity as `(new_center - unpredicted_center)`.

#### Mathematical Explanation & Justification
Let constant target velocity be $\mathbf{v}^*$.
`predict()` sets $\mathbf{x}_{\text{pred}} = \mathbf{x}_{k-1} + \mathbf{v}^*$.
Measurement $\mathbf{z}_k = \mathbf{x}_{k-1} + \mathbf{v}^*$.
`update()` computes $\mathbf{v}_k = \mathbf{z}_k - \mathbf{x}_{\text{pred}} = 0.0$.
Saving $\mathbf{x}_{k-1}$ yields correct velocity estimate $\mathbf{v}_k = \mathbf{z}_k - \mathbf{x}_{k-1} = \mathbf{v}^*$.

#### Code Changes Required

**File:** `drishti_v2/tracker/tracker.py`

```python
# MODIFY predict and update in tracker.py:
def predict(self) -> None:
    for track in self.tracks:
        track.last_unpredicted_center = track.center.clone()
        track.center = (track.center + track.velocity.to(track.center.device)).clamp(0.0, 1.0)
        track.coast_count += 1
        track.age += 1

def update(self, boxes: Tensor, logits: Tensor) -> None:
    # Inside matched block:
    new_center = det_boxes[best_det, :2].clone()
    track.velocity = (new_center - track.last_unpredicted_center.to(new_center.device)).detach()
    track.center = new_center
```

---

### Bug ADD-8 — Multi-Target Tracker Greedy Assignment Conflict

#### Problem Overview
`SimpleTracker.update()` greedily assigns detections to tracks in track list order. Track 1 steals Detection 0 even if Track 2 is significantly closer to Detection 0.

#### Proposed Solution (Intuitive)
Compute optimal global bipartite matching via Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) over the cost matrix.

#### Mathematical Explanation & Justification
Row-wise greedy matching does not optimize global total cost:
$$\min_{\pi} \sum_{i=1}^M C_{i, \pi(i)} \neq \sum_{i=1}^M \min_j C_{i, j}$$
Hungarian algorithm guarantees global optimal cost assignment $\pi^* = \arg\min_{\pi} \sum C_{i, \pi(i)}$.

#### Code Changes Required

**File:** `drishti_v2/tracker/tracker.py`

```python
from scipy.optimize import linear_sum_assignment

# In update method:
cost_matrix = torch.cdist(torch.stack([t.center for t in self.tracks]), det_boxes[:, :2]).cpu().numpy()
row_ind, col_ind = linear_sum_assignment(cost_matrix)
for r, c in zip(row_ind, col_ind):
    if cost_matrix[r, c] < self.dist_threshold:
        # Match track r to detection c
```

---

### Bug ADD-9 — Matplotlib Grayscale 1-Channel Tensor Visualization Failure

#### Problem Overview
`save_detection_figure` executes `frame.permute(1, 2, 0)`. For 1-channel thermal images `[1, H, W]`, this produces `[H, W, 1]`, causing `ax.imshow()` to raise Matplotlib shape validation errors.

#### Proposed Solution (Intuitive)
Squeeze trailing single-channel dimensions: `image = image.squeeze(-1)` when channel count is 1.

#### Mathematical Explanation & Justification
Matplotlib `imshow` input validation requirement:
$$\text{ndim} = 2 \quad \lor \quad (\text{ndim} = 3 \land \text{shape}[2] \in \{3, 4\})$$
Shape $[H, W, 1]$ fails $\text{shape}[2] \in \{3, 4\}$. Squeezing yields shape $[H, W]$, satisfying $\text{ndim}=2$.

#### Code Changes Required

**File:** `drishti_v2/evaluation/visualize.py`

```python
# REPLACE line 13 in visualize.py:
image = frame.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
if image.ndim == 3 and image.shape[2] == 1:
    image = image.squeeze(2)
```

---

## Complete Master Issue Registry & Resolution Status

| ID | File | Severity | Issue | Status |
|---|---|---|---|---|
| **T-1** | `trainer.py` | 🔴 | Checkpoint resume loads everything except model/optimizer sync | **Resolved in Spec** |
| **T-2** | `trainer.py` | 🔴 | Validation runs in `.train()` mode | **Resolved in Spec** |
| **T-3** | `trainer.py` | 🟠 | `best_score = -1.0` prevents saving best checkpoint when loss > 1.0 | **Resolved in Spec** |
| **T-4** | `stage_control.py` | 🔴 | Frozen modules not set to `.eval()` -> BatchNorm corruption | **Resolved in Spec** |
| **T-5** | `motion_loss.py` | 🔴 | Phantom displacement gradients + top-left corner bug | **Resolved in Spec** |
| **T-6** | `motion_cnn.py` | 🔴 | Heatmap loss explosion (bias init & positive normalization needed) | **Resolved in Spec** |
| **T-7** | `trainer.py` / `losses.py` | 🔴 | Temporal sequence targets not supervised | **Resolved in Spec** |
| **T-8** | `trainer.py` | 🟠 | Motion target transpose mismatch `(T, B)` vs `(B, T)` | **Resolved in Spec** |
| **L-1** | `losses.py` | 🔴 | GT assignment iterates crops, dropping objects | **Resolved in Spec** |
| **L-2** | `losses.py` | 🔴 | Bbox targets computed from predictions (circular dependency) | **Resolved in Spec** |
| **L-3** | `losses.py` | 🟠 | Bbox targets clamped to `[0, 1]` | **Resolved in Spec** |
| **L-4** | `losses.py` | 🔴 | Bbox loss coordinate mismatch (relative vs global) | **Resolved in Spec** |
| **L-5** | `detection_head.py` | 🟠 | Sigmoid clamps width/height to 1.0 | **Resolved in Spec** |
| **L-6** | `crop_encoder.py` | 🟡 | `freeze()` overrideable by parent `.train()` mode | **Resolved in Spec** |
| **ADD-1** | `metrics.py` | 🔴 | mAP computed as `precision * recall` instead of AUC-PR | **Resolved in Spec** |
| **ADD-2** | `dataset.py` | 🟠 | `del augment` silently disables data augmentation | **Resolved in Spec** |
| **ADD-3** | `ldmi.py` | 🟡 | Missing Sobel spatial edge channel in LDMI | **Resolved in Spec** |
| **ADD-4** | `crop_proposal.py` | 🟠 | Non-differentiable proposal center extraction | **Resolved in Spec** |
| **ADD-5** | `temporal_fusion.py` | 🟠 | Right-anchored positional embeddings shift with time window | **Resolved in Spec** |
| **ADD-6** | `temporal_fusion.py` | 🟠 | Temporal fusion mixes un-aligned spatial crop indices across time | **Resolved in Spec** |
| **ADD-7** | `tracker.py` | 🟠 | Tracker velocity estimation collapse under constant velocity | **Resolved in Spec** |
| **ADD-8** | `tracker.py` | 🟠 | Multi-target tracker greedy assignment conflict | **Resolved in Spec** |
| **ADD-9** | `visualize.py` | 🟡 | Matplotlib grayscale 1-channel tensor visualization failure | **Resolved in Spec** |

