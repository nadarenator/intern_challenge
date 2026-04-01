"""
Unit tests for overlap_repulsion_loss().

Each test constructs a minimal cell_features tensor directly and checks
a specific property of the loss function.
"""

import torch
import pytest
from placement import overlap_repulsion_loss, CellFeatureIdx


def make_cells(*cells):
    """
    Build a cell_features tensor from a list of (x, y, w, h) tuples.
    area and num_pins are set to dummy values (not used by overlap loss).
    """
    rows = []
    for (x, y, w, h) in cells:
        # [area, num_pins, x, y, width, height]
        rows.append([w * h, 1.0, x, y, w, h])
    t = torch.tensor(rows, dtype=torch.float32)
    t.requires_grad_(False)
    # Make positions require grad so we can check gradients
    positions = t[:, 2:4].clone().detach().requires_grad_(True)
    t = t.clone()
    t[:, 2:4] = positions
    return t, positions


# ---------------------------------------------------------------------------
# 1. Zero loss when no cells overlap
# ---------------------------------------------------------------------------

def test_no_overlap_returns_zero():
    """Two cells placed far apart should give exactly zero loss."""
    # Cell A: 2×2 centered at (0, 0)  → occupies [-1,1] × [-1,1]
    # Cell B: 2×2 centered at (10, 0) → occupies [9,11] × [-1,1]
    # min_sep_x = (2+2)/2 = 2, actual dist = 10 → no overlap
    t, _ = make_cells((0, 0, 2, 2), (10, 0, 2, 2))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(0.0), f"Expected 0, got {loss.item()}"


def test_touching_but_not_overlapping_is_zero():
    """Cells exactly touching (edge-to-edge) should give zero loss."""
    # Cell A: 4×2 at (0,0), Cell B: 2×2 at (3,0)
    # min_sep_x = (4+2)/2 = 3, dist = 3 → relu(3-3) = 0
    t, _ = make_cells((0, 0, 4, 2), (3, 0, 2, 2))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(0.0), f"Expected 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 2. Positive loss when cells overlap
# ---------------------------------------------------------------------------

def test_full_overlap_is_positive():
    """Two cells at the same position should give a large positive loss."""
    t, _ = make_cells((0, 0, 4, 4), (0, 0, 4, 4))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() > 0.0, "Expected positive loss for fully overlapping cells"


def test_partial_overlap_is_positive():
    """Partially overlapping cells should give positive loss."""
    # Cell A: 4×4 at (0,0) → [-2,2]×[-2,2]
    # Cell B: 4×4 at (3,0) → [1,5]×[-2,2]  (1-unit x overlap)
    # overlap_x = (4+4)/2 - 3 = 1,  overlap_y = (4+4)/2 - 0 = 4
    # overlap_area = 4,  num_pairs = 1
    t, _ = make_cells((0, 0, 4, 4), (3, 0, 4, 4))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(4.0), f"Expected 4.0, got {loss.item()}"


def test_x_overlap_only_is_zero():
    """Cells overlapping in x but NOT in y should give zero loss."""
    # Cell A: 4×2 at (0, 0)  → [-2,2] × [-1,1]
    # Cell B: 4×2 at (2, 5)  → [0,4]  × [4,6]
    # x-overlap = (4+4)/2 - 2 = 2 > 0, but y-dist = 5, min_sep_y = 2 → no y-overlap
    t, _ = make_cells((0, 0, 4, 2), (2, 5, 4, 2))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(0.0), f"Expected 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 3. Loss value is correct
# ---------------------------------------------------------------------------

def test_overlap_area_calculation():
    """Verify the exact overlap area for a known configuration."""
    # Cell A: 6×4 at (0, 0)
    # Cell B: 4×6 at (4, 3)
    # overlap_x = (6+4)/2 - 4 = 1
    # overlap_y = (4+6)/2 - 3 = 2
    # overlap_area = 1 * 2 = 2,  num_pairs = 1
    t, _ = make_cells((0, 0, 6, 4), (4, 3, 4, 6))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(2.0), f"Expected 2.0, got {loss.item()}"


def test_three_cells_normalization():
    """With 3 cells (3 pairs), loss should be sum_of_areas / 3."""
    # All three cells stacked at the origin, each 2×2
    # Each pair: overlap_x = 2, overlap_y = 2, overlap_area = 4
    # 3 pairs, total = 12, normalized = 12/3 = 4
    t, _ = make_cells((0, 0, 2, 2), (0, 0, 2, 2), (0, 0, 2, 2))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(4.0), f"Expected 4.0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 4. Gradient flows correctly
# ---------------------------------------------------------------------------

def test_gradients_are_nonzero_when_overlapping():
    """Overlapping cells must produce nonzero gradients on positions."""
    rows = torch.tensor([[4.0, 1.0, 0.0, 0.0, 4.0, 4.0],
                         [4.0, 1.0, 1.0, 0.0, 4.0, 4.0]], dtype=torch.float32)
    positions = rows[:, 2:4].clone().detach().requires_grad_(True)
    rows = rows.clone()
    rows[:, 2:4] = positions

    loss = overlap_repulsion_loss(rows, None, None)
    loss.backward()

    assert positions.grad is not None, "No gradients computed"
    assert positions.grad.abs().sum().item() > 0, "Gradients are all zero"


def test_gradients_push_cells_apart():
    """Gradient on an overlapping cell should point away from the other cell."""
    # Cell A at (0,0), Cell B at (1,0), both 4×4 → overlapping in x
    # In gradient descent: x -= lr * grad, so:
    #   grad_a_x > 0 → Cell A moves left (away from B) ✓
    #   grad_b_x < 0 → Cell B moves right (away from A) ✓
    rows = torch.tensor([[16.0, 1.0, 0.0, 0.0, 4.0, 4.0],
                         [16.0, 1.0, 1.0, 0.0, 4.0, 4.0]], dtype=torch.float32)
    positions = rows[:, 2:4].clone().detach().requires_grad_(True)
    rows = rows.clone()
    rows[:, 2:4] = positions

    loss = overlap_repulsion_loss(rows, None, None)
    loss.backward()

    grad_a_x = positions.grad[0, 0].item()
    grad_b_x = positions.grad[1, 0].item()

    assert grad_a_x > 0, f"Cell A should be pushed left (positive grad → x decreases in GD), got {grad_a_x}"
    assert grad_b_x < 0, f"Cell B should be pushed right (negative grad → x increases in GD), got {grad_b_x}"


def test_no_gradients_when_not_overlapping():
    """Non-overlapping cells should produce zero gradients (relu is flat outside)."""
    rows = torch.tensor([[4.0, 1.0,  0.0, 0.0, 2.0, 2.0],
                         [4.0, 1.0, 10.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
    positions = rows[:, 2:4].clone().detach().requires_grad_(True)
    rows = rows.clone()
    rows[:, 2:4] = positions

    loss = overlap_repulsion_loss(rows, None, None)
    loss.backward()

    assert positions.grad.abs().sum().item() == pytest.approx(0.0), \
        "Expected zero gradients for non-overlapping cells"


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_single_cell_returns_zero():
    """A single cell cannot overlap with anything."""
    t, _ = make_cells((0, 0, 5, 5))
    loss = overlap_repulsion_loss(t, None, None)
    assert loss.item() == pytest.approx(0.0)


def test_loss_is_symmetric():
    """Swapping cell order should not change the loss."""
    t_ab, _ = make_cells((0, 0, 4, 4), (2, 2, 3, 3))
    t_ba, _ = make_cells((2, 2, 3, 3), (0, 0, 4, 4))
    loss_ab = overlap_repulsion_loss(t_ab, None, None).item()
    loss_ba = overlap_repulsion_loss(t_ba, None, None).item()
    assert loss_ab == pytest.approx(loss_ba), \
        f"Loss should be symmetric: {loss_ab} vs {loss_ba}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
