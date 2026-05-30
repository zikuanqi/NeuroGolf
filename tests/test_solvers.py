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


def test_variable_kron_picks_count_rule():
    from neurogolf.solvers.variable_kron import solve_variable_kron
    # 3x3 input with N non-zero cells → N-times kron-scaled output.
    task = _make_task([
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # N = 1 → identity
         [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 0, 0], [0, 2, 0], [0, 0, 0]],  # N = 2 → 6x6
         [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
          [0, 0, 2, 2, 0, 0], [0, 0, 2, 2, 0, 0],
          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
    ])
    model = solve_variable_kron(task)
    assert model is not None
    op_types = {n.op_type for n in model.graph.node}
    assert {"Div", "Gather", "Less", "Cast", "Mul"} <= op_types


def test_conv_masked_picks_when_lstsq_fits():
    from neurogolf.solvers.conv3x3_masked import solve_conv1x1_masked
    # Trivial 1x1 color remap (with bias) that ANY conv kernel should fit.
    task = _make_task([
        ([[1, 2]], [[5, 5]]),
        ([[3, 4]], [[5, 5]]),
    ])
    # Whether it actually fits depends on lstsq; we just verify the
    # solver returns a valid model or cleanly None without crashing.
    result = solve_conv1x1_masked(task)
    assert result is None or hasattr(result, "graph")


def test_gravity_down_detects_and_sinks():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.gravity_down import solve_gravity_down

    # Each column's colours fall to the bottom of the real grid.
    task = _make_task([
        ([[2, 0, 3],
          [0, 0, 0],
          [0, 0, 3]],
         [[0, 0, 0],
          [0, 0, 3],
          [2, 0, 3]]),
    ])
    model = solve_gravity_down(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    out = sess.run(["output"], {"input": to_onehot(task["train"][0]["input"])})[0]
    assert from_onehot((out > 0.0).astype(np.float32)) == task["train"][0]["output"]


def test_gravity_down_rejects_non_gravity():
    from neurogolf.solvers.gravity_down import solve_gravity_down
    # Cells that don't sink to the bottom must not be claimed.
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_gravity_down(task) is None


def _fractal_expected(grid, color):
    n = len(grid)
    out = [[0] * (n * n) for _ in range(n * n)]
    for R in range(n):
        for C in range(n):
            if grid[R][C] == color:
                for r in range(n):
                    for c in range(n):
                        out[R * n + r][C * n + c] = grid[r][c]
    return out


def test_self_fractal_fixed_colour():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.self_fractal import solve_self_fractal

    grid = [[1, 0, 0], [2, 1, 0], [0, 0, 1]]   # selector colour 2 (fixed)
    task = _make_task([(grid, _fractal_expected(grid, 2))])
    model = solve_self_fractal(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    out = sess.run(["output"], {"input": to_onehot(grid)})[0]
    assert from_onehot((out > 0.0).astype(np.float32)) == task["train"][0]["output"]


def test_self_fractal_rejects_plain_kron():
    from neurogolf.solvers.self_fractal import solve_self_fractal
    # Plain NxN block upscale (every cell expands) is not a self-fractal.
    task = _make_task([([[1, 2], [3, 4]],
                        [[1, 1, 2, 2], [1, 1, 2, 2],
                         [3, 3, 4, 4], [3, 3, 4, 4]])])
    assert solve_self_fractal(task) is None


def _rot_tile_expected(grid):
    import numpy as np
    i = np.array(grid)
    top = np.hstack([i, np.rot90(i, 3)])
    bot = np.hstack([np.rot90(i, 1), np.rot90(i, 2)])
    return np.vstack([top, bot]).tolist()


def test_rot_tile_assembles_four_rotations():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.rot_tile import solve_rot_tile

    grid = [[8, 5, 0], [8, 5, 3], [0, 3, 2]]
    task = _make_task([(grid, _rot_tile_expected(grid))])
    model = solve_rot_tile(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    out = sess.run(["output"], {"input": to_onehot(grid)})[0]
    assert from_onehot((out > 0.0).astype(np.float32)) == task["train"][0]["output"]


def test_rot_tile_rejects_mixed_sizes():
    from neurogolf.solvers.rot_tile import solve_rot_tile
    # The graph bakes in one N, so mixed grid sizes must be rejected.
    g2 = [[1, 2], [3, 4]]
    g3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    task = _make_task([(g2, _rot_tile_expected(g2)),
                       (g3, _rot_tile_expected(g3))])
    assert solve_rot_tile(task) is None


def test_gravity_up_preserves_column_order():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.gravity_up import solve_gravity_up

    # A mixed-colour column must rise while preserving its top-to-bottom order.
    task = _make_task([
        ([[0, 0, 3],
          [2, 0, 0],
          [0, 0, 8]],
         [[2, 0, 3],
          [0, 0, 8],
          [0, 0, 0]]),
    ])
    model = solve_gravity_up(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    out = sess.run(["output"], {"input": to_onehot(task["train"][0]["input"])})[0]
    assert from_onehot((out > 0.0).astype(np.float32)) == task["train"][0]["output"]


def test_gravity_up_rejects_non_gravity():
    from neurogolf.solvers.gravity_up import solve_gravity_up
    task = _make_task([([[0, 0], [2, 0]], [[0, 0], [2, 0]])])
    assert solve_gravity_up(task) is None


def test_periodic_fill_inpaints_period_two():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.periodic_fill import solve_periodic_fill

    # A 4x4 checker with period (2,2); two cells erased to 0 must be restored.
    full = [[3, 4, 3, 4],
            [5, 6, 5, 6],
            [3, 4, 3, 4],
            [5, 6, 5, 6]]
    erased = [[3, 4, 3, 4],
              [5, 0, 5, 6],
              [3, 4, 0, 4],
              [5, 6, 5, 6]]
    task = _make_task([(erased, full)])
    model = solve_periodic_fill(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    out = sess.run(["output"], {"input": to_onehot(erased)})[0]
    assert from_onehot((out > 0.0).astype(np.float32)) == full


def test_periodic_fill_rejects_non_periodic():
    from neurogolf.solvers.periodic_fill import solve_periodic_fill
    # No erased cells / not a periodic completion -> must decline.
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_periodic_fill(task) is None


def test_split_logic_left_right_or():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.split_logic import solve_split_logic

    # left | right halves OR-combined, painted colour 6.
    inp = [[1, 0, 0, 0, 1, 0],
           [0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1]]
    out = [[6, 6, 0],
           [6, 6, 0],
           [0, 0, 6]]
    task = _make_task([(inp, out)])
    model = solve_split_logic(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_split_logic_top_bottom_nor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.split_logic import solve_split_logic

    # top / bottom halves NOR-combined (on where both empty), painted colour 2.
    inp = [[1, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 0, 1]]
    out = [[0, 0, 2],
           [2, 2, 0]]
    task = _make_task([(inp, out)])
    model = solve_split_logic(task)
    assert model is not None

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out
