# VLSI Placement: Kevin Joseph

Solution to the [par.tcl 2026 intern challenge](https://github.com/partcleda/intern_challenge). Achieves **0.0000 overlap** and **0.2480 normalized wirelength** (rank 1) across all 10 test cases.

## Results

| Overlap | Wirelength (um) | Runtime (s) |
|---------|-----------------|-------------|
| 0.0000  | 0.2480          | 435.52      |

## Problem

Given a set of rectangular cells (macros and standard cells) with fixed sizes and wire connections between them, find (x, y) positions that:
1. **Eliminate all overlaps** (primary objective)
2. **Minimize total wirelength** (secondary objective)

## Approach: 3-Phase Gradient Descent

### Phase 1: Wirelength-only initialization

Cells start at the origin and are pulled toward the centroid of their connections using a pure wirelength loss (smooth Chebyshev / log-sum-exp approximation of Manhattan distance). This produces a compact, connectivity-aware layout before any overlap penalty is introduced.

**Why this matters:** Starting from a good wirelength-minimizing layout means phase 2 only needs to spread cells apart slightly rather than rearrange them entirely. This is what drives the low final wirelength, since the topology is already good before overlap elimination begins.

Phase 1 uses patience-based early stopping: exits if the loss doesn't improve by more than `1e-6` for 50 consecutive epochs.

### Phase 2: Overlap elimination with exponential λ-annealing

Rather than switching abruptly to a pure overlap loss (which would discard the phase 1 layout), phase 2 uses a combined loss with an exponentially annealed schedule:

```
t        = epoch / (phase_max_epochs - 1)           # 0 -> 1
progress = (1 - exp(-k*t)) / (1 - exp(-k))          # rescaled, k=19
λ1       = 0.99 * (1 - progress)                    # wirelength weight: 0.99 -> 0
λ2       = 0.01 + 0.99 * progress                   # overlap weight:    0.01 -> 1
loss     = λ1 * wl_loss + λ2 * overlap_loss
```

The exponential shape (k=19) is steep early, so overlap dominates well before the epoch budget runs out. This ensures large designs (test 10: 2010 cells) converge reliably.

The zero-overlap check happens **before** `optimizer.step()`. This prevents Adam's momentum from nudging positions into overlap after the stopping condition is detected.

### Phase 3: Wirelength fine-tuning with overlap guard

Optimizes purely for wirelength at a low learning rate (`1e-4`). A no-grad overlap check runs each epoch; if any overlap reappears, the previous zero-overlap snapshot is restored and training stops. Empirically found to be redundant.

## Loss Functions

**Overlap repulsion loss:** For each pair of cells, computes the 1D overlap in x and y independently, then multiplies them to get 2D overlap area. Squared to penalize large overlaps more heavily. Uses `torch.triu` to avoid double-counting pairs. Fully vectorized via PyTorch broadcasting with no Python loops.

**Wirelength attraction loss:** Smooth Chebyshev distance (log-sum-exp approximation of Manhattan distance). The same function is used for training in all three phases and for the leaderboard metric, so there is no discrepancy between training and evaluation.

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| 3-phase separation | Overlap and wirelength gradients conflict; joint optimization converges poorly |
| Phase 1 WL-only init | Seeds a good topology before overlap is introduced; directly responsible for low final WL |
| Exponential λ schedule (k=19) | Linear annealing converges too slowly for large designs; steep early transition is critical for test 10 |
| Zero-overlap check before `optimizer.step()` | Adam momentum can overshoot; checking after the step misses this |
| Smooth Chebyshev for WL | Differentiable approximation of Manhattan distance; Worked better than squared Euclidean distance |
| Patience-based early stop in phase 1 | Avoids wasting epochs once the centroid layout stabilizes |

## Running

```bash
# Install dependencies
uv pip install torch numpy matplotlib

# Run tests 1-10
uv run python test.py

# With per-phase verbose output
uv run python test.py --verbose

# Save 4-panel placement visualizations in output/
uv run python test.py --visualize
```

## Environment

- Python 3.11, PyTorch, CUDA 13.0
- GPU: NVIDIA RTX 5070 Ti Mobile
