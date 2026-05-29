"""Smoke tests that the registered solvers handle their target patterns."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from neurogolf.solvers.bbox_strip import solve_bbox_strip  # noqa: E402
from neurogolf.solvers.identity import solve_identity  # noqa: E402
from neurogolf.solvers.kron_scale import solve_kron_scale  # noqa: E402
from neurogolf.solvers.remap import solve_remap  # noqa: E402
from neurogolf.solvers.resize_scale import solve_resize_scale  # noqa: E402
from neurogolf.solvers.shape_aware_flip import (  # noqa: E402
    solve_flip_h_aware, solve_flip_v_aware, solve_rot180_aware,
    solve_rot90_ccw_aware,
)
from neurogolf.solvers.shift import solve_shift  # noqa: E402
from neurogolf.solvers.spatial import solve_transpose  # noqa: E402
from neurogolf.solvers.static_crop import solve_static_crop  # noqa: E402
from neurogolf.solvers.majority_fill import solve_majority_fill  # noqa: E402
from neurogolf.solvers.palindrome import (  # noqa: E402
    solve_palindrome_2d, solve_palindrome_h, solve_palindrome_v,
)
from neurogolf.solvers.tile_h import solve_tile_h  # noqa: E402


def _make_task(pairs):
    train = [{"input": i, "output": o} for i, o in pairs]
    return {"train": train, "test": [], "arc-gen": []}


def test_identity_picks_only_identity():
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    model = solve_identity(task)
    assert model is not None
    assert model.graph.node[0].op_type == "Identity"


def test_identity_rejects_remap():
    task = _make_task([([[1, 2], [3, 4]], [[2, 1], [4, 3]])])
    assert solve_identity(task) is None


def test_remap_simple_swap():
    task = _make_task([
        ([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
        ([[1, 1], [0, 0]], [[2, 2], [0, 0]]),
    ])
    model = solve_remap(task)
    assert model is not None
    assert model.graph.node[0].op_type == "Conv"
    assert len(model.graph.initializer) == 1
    w = model.graph.initializer[0]
    assert list(w.dims) == [10, 10, 1, 1]


def test_remap_rejects_position_dependent():
    task = _make_task([([[1, 1], [1, 1]], [[2, 3], [4, 5]])])
    assert solve_remap(task) is None


def test_transpose():
    task = _make_task([([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]])])
    model = solve_transpose(task)
    assert model is not None
    assert model.graph.node[0].op_type == "Transpose"


def test_static_crop_picks_fixed_offset():
    # Output is always the 2x2 top-right of a 4x4 input.
    task = _make_task([
        ([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[3, 4], [7, 8]]),
    ])
    model = solve_static_crop(task)
    assert model is not None
    op_types = {n.op_type for n in model.graph.node}
    assert {"Slice", "Pad"} <= op_types


def test_kron_scale_picks_2x():
    task = _make_task([
        ([[1, 2], [3, 4]],
         [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
    ])
    model = solve_kron_scale(task)
    assert model is not None
    assert sum(n.op_type == "Gather" for n in model.graph.node) == 2


def test_resize_scale_picks_variable_shape_2x():
    # kron_scale wants ONE input shape; resize_scale accepts varying ones.
    task = _make_task([
        ([[1]], [[1, 1], [1, 1]]),
        ([[1, 2], [3, 4]],
         [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
    ])
    assert solve_kron_scale(task) is None
    model = solve_resize_scale(task)
    assert model is not None
    op_types = [n.op_type for n in model.graph.node]
    assert "Resize" in op_types


def test_shape_aware_flips_emit_dynamic_pipeline():
    # 2x3 grid: flipping horizontally swaps columns within each row.
    h_task = _make_task([
        ([[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [6, 5, 4]]),
    ])
    h_model = solve_flip_h_aware(h_task)
    assert h_model is not None
    assert any(n.op_type == "Gather" for n in h_model.graph.node)
    # Flip-v solver should NOT match a flip-h task.
    assert solve_flip_v_aware(h_task) is None

    v_task = _make_task([
        ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
    ])
    assert solve_flip_v_aware(v_task) is not None
    assert solve_flip_h_aware(v_task) is None


def test_rot_solvers_distinguish_directions():
    cw_input = [[1, 2], [3, 4]]
    # rot180: [[4,3],[2,1]]; rot90 ccw: [[2,4],[1,3]]; cw: [[3,1],[4,2]]
    rot180 = _make_task([(cw_input, [[4, 3], [2, 1]])])
    rot_ccw = _make_task([(cw_input, [[2, 4], [1, 3]])])
    assert solve_rot180_aware(rot180) is not None
    assert solve_rot180_aware(rot_ccw) is None
    assert solve_rot90_ccw_aware(rot_ccw) is not None
    assert solve_rot90_ccw_aware(rot180) is None


def test_bbox_strip_picks_non_bg():
    # The bbox of non-zero cells is the 2x2 block (1..2, 1..2).
    task = _make_task([
        ([[0, 0, 0, 0],
          [0, 1, 2, 0],
          [0, 3, 4, 0],
          [0, 0, 0, 0]],
         [[1, 2], [3, 4]]),
    ])
    model = solve_bbox_strip(task)
    assert model is not None
    op_types = [n.op_type for n in model.graph.node]
    assert "ArgMax" in op_types
    assert "Less" in op_types
    assert sum(t == "Gather" for t in op_types) == 2


def test_shift_constant_shape_down_one():
    # 3x3 grid shifted down by one — top row becomes color 0.
    task = _make_task([
        ([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
         [[0, 0, 0], [1, 2, 3], [4, 5, 6]]),
    ])
    model = solve_shift(task)
    assert model is not None
    op_types = [n.op_type for n in model.graph.node]
    # Slice the source, Concat with a color-0 fill row, Pad to canvas.
    assert op_types == ["Slice", "Concat", "Pad"]
    # The fill initializer must encode color-0 cells: channel 0 = 1.0.
    fill = next(i for i in model.graph.initializer if i.name == "fill")
    assert list(fill.dims) == [1, 10, 1, 3]


def test_shift_rejects_non_translation():
    task = _make_task([([[1, 2], [3, 4]], [[5, 5], [5, 5]])])
    assert solve_shift(task) is None


def test_tile_h_picks_factor_2():
    task = _make_task([
        ([[1, 2], [3, 4]], [[1, 2, 1, 2], [3, 4, 3, 4]]),
    ])
    model = solve_tile_h(task)
    assert model is not None
    op_types = {n.op_type for n in model.graph.node}
    # Tile-h is shape-aware: it must compute W, take a Mod, gather, then mask.
    assert {"Mod", "Gather", "Less"} <= op_types


def test_tile_h_picks_factor_3():
    task = _make_task([
        ([[1, 2]], [[1, 2, 1, 2, 1, 2]]),
    ])
    assert solve_tile_h(task) is not None


def test_tile_h_rejects_non_tile():
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_tile_h(task) is None


def test_palindrome_h_picks_mirror_right():
    task = _make_task([
        ([[1, 2, 3]], [[1, 2, 3, 3, 2, 1]]),
    ])
    model = solve_palindrome_h(task)
    assert model is not None
    op_types = {n.op_type for n in model.graph.node}
    assert {"Where", "Gather", "Less"} <= op_types


def test_palindrome_v_picks_mirror_bottom():
    task = _make_task([
        ([[1], [2]], [[1], [2], [2], [1]]),
    ])
    assert solve_palindrome_v(task) is not None
    assert solve_palindrome_h(task) is None


def test_palindrome_2d_picks_four_quadrants():
    task = _make_task([
        ([[1, 2], [3, 4]],
         [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3], [1, 2, 2, 1]]),
    ])
    assert solve_palindrome_2d(task) is not None


def test_majority_fill_picks_3x3_majority():
    task = _make_task([
        ([[0, 1, 1], [2, 1, 0], [3, 0, 0]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[2, 2, 2], [0, 1, 0], [0, 3, 0]],
         [[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
    ])
    model = solve_majority_fill(task)
    assert model is not None
    op_types = {n.op_type for n in model.graph.node}
    assert {"TopK", "Greater", "OneHot", "Where"} <= op_types


def test_majority_fill_rejects_when_output_varies():
    # Output color isn't a majority in input
    task = _make_task([
        ([[1, 2], [1, 2]], [[5, 5], [5, 5]]),
    ])
    assert solve_majority_fill(task) is None
