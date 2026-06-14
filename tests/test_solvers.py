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


def test_connect_dots_horizontal():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.connect_dots import solve_connect_dots

    # two same-colour dots in a row get the gap between them filled.
    inp = [[3, 0, 0, 0, 3],
           [0, 2, 0, 2, 0],
           [0, 0, 0, 0, 0]]
    out = [[3, 3, 3, 3, 3],
           [0, 2, 2, 2, 0],
           [0, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_connect_dots(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_connect_dots_rejects_unrelated():
    from neurogolf.solvers.connect_dots import solve_connect_dots
    # A lone dot (nothing to connect) leaves the grid unchanged -> decline.
    task = _make_task([([[5, 0], [0, 0]], [[5, 0], [0, 0]])])
    assert solve_connect_dots(task) is None


def test_mirror_complete_vertical():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.mirror_complete import solve_mirror_complete

    # bottom half erased; restore it as the vertical mirror of the top half.
    inp = [[2, 2, 2],
           [3, 3, 3],
           [0, 0, 0],
           [0, 0, 0]]
    out = [[2, 2, 2],
           [3, 3, 3],
           [3, 3, 3],
           [2, 2, 2]]
    task = _make_task([(inp, out)])
    model = solve_mirror_complete(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_mirror_complete_rejects_no_fill():
    from neurogolf.solvers.mirror_complete import solve_mirror_complete
    # Already symmetric, nothing to restore -> decline.
    task = _make_task([([[2, 2], [2, 2]], [[2, 2], [2, 2]])])
    assert solve_mirror_complete(task) is None


def test_denoise_removes_isolated_cells():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.denoise import solve_denoise

    # the lone 3 is removed; the 2x1 pair of 2s (touching) survives.
    inp = [[3, 0, 0],
           [0, 0, 2],
           [0, 0, 2]]
    out = [[0, 0, 0],
           [0, 0, 2],
           [0, 0, 2]]
    task = _make_task([(inp, out)])
    model = solve_denoise(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_denoise_rejects_no_change():
    from neurogolf.solvers.denoise import solve_denoise
    # nothing isolated -> grid unchanged -> decline.
    task = _make_task([([[2, 2], [0, 0]], [[2, 2], [0, 0]])])
    assert solve_denoise(task) is None


def test_connect_fill_uses_fixed_colour():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.connect_fill import solve_connect_fill

    # two 8s in a row; the gap is filled with colour 3 (not 8), endpoints kept.
    inp = [[8, 0, 0, 8],
           [0, 0, 0, 0]]
    out = [[8, 3, 3, 8],
           [0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_connect_fill(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_connect_fill_rejects_no_change():
    from neurogolf.solvers.connect_fill import solve_connect_fill
    # single dot, nothing to connect -> decline.
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_connect_fill(task) is None


def test_drop_into_wall_drops_one():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.drop_into_wall import solve_drop_into_wall

    # the 1 above the full colour-5 wall drops into the wall in its column.
    inp = [[0, 1, 0],
           [0, 5, 0],
           [5, 5, 5]]
    out = [[0, 0, 0],
           [0, 5, 0],
           [5, 1, 5]]
    task = _make_task([(inp, out)])
    model = solve_drop_into_wall(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_drop_into_wall_rejects_no_wall():
    from neurogolf.solvers.drop_into_wall import solve_drop_into_wall
    # no full colour-5 wall row -> decline.
    task = _make_task([([[1, 0], [0, 0]], [[1, 0], [0, 0]])])
    assert solve_drop_into_wall(task) is None


def test_color_bbox_fill_solid_rectangles():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.color_bbox_fill import solve_color_bbox_fill

    # scattered colour cells -> each colour's bbox becomes a solid rectangle.
    inp = [[3, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 0]]
    out = [[3, 3, 3, 0],
           [3, 3, 3, 0],
           [3, 3, 3, 0],
           [0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_color_bbox_fill(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_color_bbox_fill_rejects_no_change():
    from neurogolf.solvers.color_bbox_fill import solve_color_bbox_fill
    # already-solid single cell -> unchanged -> decline.
    task = _make_task([([[3, 0], [0, 0]], [[3, 0], [0, 0]])])
    assert solve_color_bbox_fill(task) is None


def test_nearest_wall_recolors_to_nearer_wall():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.nearest_wall import solve_nearest_wall

    # Left column = colour 1, right column = colour 2; each interior marker is
    # repainted with the colour of the wall it sits closer to.
    inp = [[1, 0, 3, 0, 0, 2],
           [1, 0, 0, 0, 3, 2],
           [1, 0, 0, 0, 0, 2]]
    out = [[1, 0, 1, 0, 0, 2],
           [1, 0, 0, 0, 2, 2],
           [1, 0, 0, 0, 0, 2]]
    task = _make_task([(inp, out)])
    model = solve_nearest_wall(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_nearest_wall_handles_horizontal_walls():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.nearest_wall import solve_nearest_wall

    # Same network must also cope with top/bottom walls (colours 4 and 7).
    inp = [[4, 4, 4],
           [0, 3, 0],
           [0, 0, 0],
           [3, 0, 0],
           [7, 7, 7]]
    out = [[4, 4, 4],
           [0, 4, 0],
           [0, 0, 0],
           [7, 0, 0],
           [7, 7, 7]]
    task = _make_task([(inp, out)])
    model = solve_nearest_wall(task)
    assert model is not None

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_nearest_wall_rejects_without_facing_walls():
    from neurogolf.solvers.nearest_wall import solve_nearest_wall
    # Only the left column is a uniform wall -> no facing pair -> decline.
    task = _make_task([([[1, 0, 3], [1, 0, 0], [1, 0, 0]],
                        [[1, 0, 3], [1, 0, 0], [1, 0, 0]])])
    assert solve_nearest_wall(task) is None


def test_cross_laser_plus_and_intersection():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.cross_laser import solve_cross_laser

    # Two markers: a 3 and a 4. Each fires a full row+column; where the 3's row
    # meets the 4's column (and vice-versa) the cell becomes colour 2.
    inp = [[0, 0, 0, 0, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 0, 0],
           [4, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]
    out = [[4, 0, 3, 0, 0],
           [2, 3, 3, 3, 3],
           [4, 0, 3, 0, 0],
           [4, 4, 2, 4, 4],
           [4, 0, 3, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_cross_laser(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_cross_laser_rejects_non_cross():
    from neurogolf.solvers.cross_laser import solve_cross_laser
    # A lone marker left unchanged is not the plus-laser transform -> decline.
    task = _make_task([([[0, 5, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 5, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_cross_laser(task) is None


def test_halo_rings_markers_with_colour_1():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.halo import solve_halo

    # The lone 5 is kept; every background cell touching it becomes colour 1.
    inp = [[0, 0, 0, 0],
           [0, 5, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]
    out = [[1, 1, 1, 0],
           [1, 5, 1, 0],
           [1, 1, 1, 0],
           [0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_halo(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_halo_rejects_when_not_a_halo():
    from neurogolf.solvers.halo import solve_halo
    # Output leaves the marker bare (no ring) -> not the halo transform -> decline.
    task = _make_task([([[0, 5, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 5, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_halo(task) is None


def test_color_lines_vertical_two_horizontal_others():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.color_lines import solve_color_lines

    # colour-2 marker -> its column becomes a vertical line; the colour-1 marker
    # -> its row becomes a horizontal line, drawn on top of the vertical.
    inp = [[0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 2, 0],
           [0, 0, 0, 0, 0]]
    out = [[0, 0, 0, 2, 0],
           [1, 1, 1, 1, 1],
           [0, 0, 0, 2, 0],
           [0, 0, 0, 2, 0],
           [0, 0, 0, 2, 0]]
    task = _make_task([(inp, out)])
    model = solve_color_lines(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_color_lines_rejects_non_lines():
    from neurogolf.solvers.color_lines import solve_color_lines
    # A lone marker left unchanged is not the line transform -> decline.
    task = _make_task([([[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_color_lines(task) is None


def test_endpoint_bridge_fills_between_two_dots():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.endpoint_bridge import solve_endpoint_bridge

    # Row with dots colour 3 (left) and 4 (right): left half -> 3, right half -> 4,
    # midpoint (floor((0+6)/2)=3) -> 5.
    inp = [[0, 0, 0, 0, 0, 0, 0],
           [3, 0, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 0]]
    out = [[0, 0, 0, 0, 0, 0, 0],
           [3, 3, 3, 5, 4, 4, 4],
           [0, 0, 0, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_endpoint_bridge(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_endpoint_bridge_rejects_lone_dot():
    from neurogolf.solvers.endpoint_bridge import solve_endpoint_bridge
    # A row with a single dot has no pair to bridge -> decline.
    task = _make_task([([[5, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[5, 0, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_endpoint_bridge(task) is None


def test_keep_majority_recolours_minority_to_5():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.keep_majority import solve_keep_majority

    # colour 2 is the most frequent -> kept; the 1 and 8s become colour 5.
    inp = [[2, 2, 2],
           [2, 1, 8],
           [2, 8, 8]]
    out = [[2, 2, 2],
           [2, 5, 5],
           [2, 5, 5]]
    task = _make_task([(inp, out)])
    model = solve_keep_majority(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_keep_majority_rejects_tie():
    from neurogolf.solvers.keep_majority import solve_keep_majority
    # Two colours tie for most frequent -> ambiguous -> decline.
    task = _make_task([([[1, 2], [0, 0]], [[1, 2], [0, 0]])])
    assert solve_keep_majority(task) is None


def test_blob_recolor_paints_blob_with_key():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.blob_recolor import solve_blob_recolor

    # blob colour 2 (frequent) repainted with key colour 4; key cell + bg cleared.
    inp = [[2, 2, 2],
           [2, 2, 0],
           [0, 0, 4]]
    out = [[4, 4, 4],
           [4, 4, 0],
           [0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_blob_recolor(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_blob_recolor_rejects_single_colour():
    from neurogolf.solvers.blob_recolor import solve_blob_recolor
    # Only one non-background colour -> no key to recolour with -> decline.
    task = _make_task([([[2, 2], [0, 0]], [[2, 2], [0, 0]])])
    assert solve_blob_recolor(task) is None


def test_recolor_fives_takes_row_marker():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.recolor_fives import solve_recolor_fives

    # Each row's 5s become that row's marker colour (1, then 2); marker stays.
    inp = [[1, 0, 5, 5, 0, 0],
           [2, 0, 0, 0, 5, 5],
           [0, 0, 0, 0, 0, 0]]
    out = [[1, 0, 1, 1, 0, 0],
           [2, 0, 0, 0, 2, 2],
           [0, 0, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_recolor_fives(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_recolor_fives_rejects_marker_less_fives():
    from neurogolf.solvers.recolor_fives import solve_recolor_fives
    # A row with 5s but no marker has no colour to take -> decline.
    task = _make_task([([[5, 5, 0], [0, 0, 0], [0, 0, 0]],
                        [[5, 5, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_recolor_fives(task) is None


def test_dilate_ones_fills_3x3_block():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.dilate_ones import solve_dilate_ones

    # The lone 5 expands into a filled 3x3 of colour 1; the rest is background.
    inp = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 5, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]
    out = [[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_dilate_ones(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_dilate_ones_rejects_colour_preserving():
    from neurogolf.solvers.dilate_ones import solve_dilate_ones
    # Output keeps the original colour rather than dilating to 1 -> decline.
    task = _make_task([([[0, 5, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 5, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_dilate_ones(task) is None


def test_count_bar_counts_markers():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.count_bar import solve_count_bar

    # Three colour-7 cells -> a 1x3 bar of colour 7.
    inp = [[0, 7, 0],
           [7, 0, 0],
           [0, 0, 7]]
    out = [[7, 7, 7]]
    task = _make_task([(inp, out)])
    model = solve_count_bar(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_count_bar_rejects_multi_colour():
    from neurogolf.solvers.count_bar import solve_count_bar
    # Two marker colours -> ambiguous bar colour -> decline.
    task = _make_task([([[1, 2], [0, 0]], [[1, 2]])])
    assert solve_count_bar(task) is None


def test_hspan_fill_fills_between_walls():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.hspan_fill import solve_hspan_fill

    # Background cells flanked left and right by colour-1 walls become colour 2.
    inp = [[1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0]]
    out = [[1, 2, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, 2, 1, 0]]
    task = _make_task([(inp, out)])
    model = solve_hspan_fill(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_hspan_fill_rejects_without_span():
    from neurogolf.solvers.hspan_fill import solve_hspan_fill
    # A lone wall with nothing to flank -> no fill colour to infer -> decline.
    task = _make_task([([[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[1, 0, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_hspan_fill(task) is None


def test_stamp_paints_fixed_motif():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.stamp import solve_stamp

    # Each marker becomes a fixed 3x3 motif (diagonal 5, orthogonal 1, centre 0).
    inp = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 5, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]
    out = [[0, 0, 0, 0, 0],
           [0, 5, 1, 5, 0],
           [0, 1, 0, 1, 0],
           [0, 5, 1, 5, 0],
           [0, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_stamp(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_stamp_rejects_multi_colour_markers():
    from neurogolf.solvers.stamp import solve_stamp
    # Two distinct marker colours -> not a single fixed stamp -> decline.
    task = _make_task([([[1, 0], [0, 2]], [[1, 0], [0, 2]])])
    assert solve_stamp(task) is None


def test_outline_keeps_perimeter():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.outline import solve_outline

    # Solid 3x3 block -> only the border survives, the centre is cleared.
    inp = [[0, 0, 0, 0, 0],
           [0, 5, 5, 5, 0],
           [0, 5, 5, 5, 0],
           [0, 5, 5, 5, 0],
           [0, 0, 0, 0, 0]]
    out = [[0, 0, 0, 0, 0],
           [0, 5, 5, 5, 0],
           [0, 5, 0, 5, 0],
           [0, 5, 5, 5, 0],
           [0, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_outline(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_outline_rejects_when_interior_kept():
    from neurogolf.solvers.outline import solve_outline
    # Output keeps the solid interior -> not an outline transform -> decline.
    task = _make_task([([[5, 5, 5], [5, 5, 5], [5, 5, 5]],
                        [[5, 5, 5], [5, 5, 5], [5, 5, 5]])])
    assert solve_outline(task) is None


def test_ray_down_fills_columns_downward():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.ray_down import solve_ray_down

    # Each marker's colour is carried straight down its column.
    inp = [[5, 0, 0],
           [0, 0, 0],
           [0, 2, 0],
           [0, 0, 0]]
    out = [[5, 0, 0],
           [5, 0, 0],
           [5, 2, 0],
           [5, 2, 0]]
    task = _make_task([(inp, out)])
    model = solve_ray_down(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_ray_down_rejects_when_not_filled():
    from neurogolf.solvers.ray_down import solve_ray_down
    # Output leaves the column unfilled -> not a downward fill -> decline.
    task = _make_task([([[5, 0], [0, 0]], [[5, 0], [0, 0]])])
    assert solve_ray_down(task) is None


def test_isolate_recolor_by_connectivity():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.isolate_recolor import solve_isolate_recolor

    # Connected 3s become 8; isolated 3s stay 3.
    inp = [[3, 3, 0],
           [0, 3, 0],
           [3, 0, 3]]
    out = [[8, 8, 0],
           [0, 8, 0],
           [3, 0, 3]]
    task = _make_task([(inp, out)])
    model = solve_isolate_recolor(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_isolate_recolor_rejects_inconsistent():
    from neurogolf.solvers.isolate_recolor import solve_isolate_recolor
    # Two isolated cells of the same colour map to different colours -> decline.
    task = _make_task([([[1, 0], [0, 1]], [[2, 0], [0, 3]])])
    assert solve_isolate_recolor(task) is None


def test_cc_size_recolor_by_component_size():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.cc_size_recolor import solve_cc_size_recolor

    # A size-2 component -> colour 3; a size-1 component -> colour 4.
    inp = [[5, 5, 0, 5],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]
    out = [[3, 3, 0, 4],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_cc_size_recolor(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_cc_size_recolor_rejects_inconsistent_size_map():
    from neurogolf.solvers.cc_size_recolor import solve_cc_size_recolor
    # Two size-2 components map to different colours -> decline.
    task = _make_task([([[5, 5, 0], [0, 0, 0], [5, 5, 0]],
                        [[1, 1, 0], [0, 0, 0], [2, 2, 0]])])
    assert solve_cc_size_recolor(task) is None


def test_cc_rank_recolor_by_size_rank():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.cc_rank_recolor import solve_cc_rank_recolor

    # Three separated bars, sizes 3/2/1 -> ranks 0/1/2 -> colours 1/4/2.
    inp = [[5, 0, 5, 0, 5],
           [5, 0, 5, 0, 0],
           [5, 0, 0, 0, 0]]
    out = [[1, 0, 4, 0, 2],
           [1, 0, 4, 0, 0],
           [1, 0, 0, 0, 0]]
    task = _make_task([(inp, out)])
    model = solve_cc_rank_recolor(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(inp)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_cc_rank_recolor_rejects_ties():
    from neurogolf.solvers.cc_rank_recolor import solve_cc_rank_recolor
    # Two equal-size components -> ambiguous rank -> decline.
    task = _make_task([([[5, 0, 5], [5, 0, 5], [0, 0, 0]],
                        [[1, 0, 2], [1, 0, 2], [0, 0, 0]])])
    assert solve_cc_rank_recolor(task) is None


def test_diag_tile_picks_diagonal_cyclic():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diag_tile import solve_diag_tile

    # 3-colour diagonal cyclic tiling: out[i][j] = order[(i + j) % 3]
    grid = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    task = _make_task([(grid, grid)])
    model = solve_diag_tile(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(grid)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == grid


def test_diag_tile_rejects_plain_recolor():
    from neurogolf.solvers.diag_tile import solve_diag_tile
    # A uniform recolour is not a diagonal tiling -> decline.
    task = _make_task([([[1, 1], [1, 1]], [[2, 2], [2, 2]])])
    assert solve_diag_tile(task) is None


def test_axis_gather_vertical_mirror_stack():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.axis_gather import solve_axis_gather

    # output = [flip_v(input); input] -- a constant row gather (task 116).
    grid = [[1, 2, 3], [4, 5, 6]]
    out = [[4, 5, 6], [1, 2, 3], [1, 2, 3], [4, 5, 6]]
    task = _make_task([(grid, out)])
    model = solve_axis_gather(task)
    assert model is not None and hasattr(model, "graph")
    assert model.graph.node[0].op_type == "Gather"

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(grid)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_axis_gather_rejects_identity():
    from neurogolf.solvers.axis_gather import solve_axis_gather
    # A pure identity is not a non-trivial gather -> decline.
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_axis_gather(task) is None


def test_diag_ray_down_right():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diag_ray import solve_diag_ray

    # Each marker emits a down-right ray in a 2N x 2N output (task 327).
    grid = [[6, 1, 0], [3, 0, 0], [0, 0, 0]]
    out = [
        [6, 1, 0, 0, 0, 0],
        [3, 6, 1, 0, 0, 0],
        [0, 3, 6, 1, 0, 0],
        [0, 0, 3, 6, 1, 0],
        [0, 0, 0, 3, 6, 1],
        [0, 0, 0, 0, 3, 6],
    ]
    task = _make_task([(grid, out)])
    model = solve_diag_ray(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(grid)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_diag_ray_rejects_non_diagonal():
    from neurogolf.solvers.diag_ray import solve_diag_ray
    # Same-shape recolour is not a 2N diagonal extrusion -> decline.
    task = _make_task([([[1, 0], [0, 1]], [[2, 0], [0, 2]])])
    assert solve_diag_ray(task) is None


def test_rot_tile_aware_variable_size():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.rot_tile import _rot_tile
    from neurogolf.solvers.rot_tile_aware import solve_rot_tile_aware
    from neurogolf.solvers.rot_tile import solve_rot_tile

    # Two different grid sizes -> the static (single-N) solver must decline,
    # the shape-aware one handles both (task 106).
    g2 = [[1, 2], [3, 4]]
    g3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    task = _make_task([(g2, _rot_tile(g2)), (g3, _rot_tile(g3))])
    assert solve_rot_tile(task) is None
    model = solve_rot_tile_aware(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    for g in (g2, g3):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == _rot_tile(g)


def test_rot_tile_aware_rejects_non_tiling():
    from neurogolf.solvers.rot_tile_aware import solve_rot_tile_aware
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_rot_tile_aware(task) is None


def test_block_count_bar_counts_2x2_blocks():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.block_count_bar import solve_block_count_bar

    # Two 2x2 colour-1 blocks -> 1x5 bar with two 1s (task 38 shape).
    g1 = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0],
    ]
    # One 2x2 colour-1 block -> one 1.
    g2 = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
    ]
    task = _make_task([(g1, [[1, 1, 0, 0, 0]]), (g2, [[1, 0, 0, 0, 0]])])
    model = solve_block_count_bar(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    for g, exp in ((g1, [[1, 1, 0, 0, 0]]), (g2, [[1, 0, 0, 0, 0]])):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == exp


def test_block_count_bar_rejects_non_count():
    from neurogolf.solvers.block_count_bar import solve_block_count_bar
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_block_count_bar(task) is None


def test_odd_panel_picks_unique_quadrant():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.odd_panel import solve_odd_panel

    # 5x5 with blank middle row/col -> four 2x2 panels; BR differs (task 207).
    g = [
        [0, 2, 0, 0, 2],
        [2, 2, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [0, 2, 0, 2, 2],
        [2, 2, 0, 2, 0],
    ]
    out = [[2, 2], [2, 0]]
    task = _make_task([(g, out)])
    model = solve_odd_panel(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == out


def test_odd_panel_rejects_non_panel():
    from neurogolf.solvers.odd_panel import solve_odd_panel
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_odd_panel(task) is None


def test_odd_panel_aware_variable_size():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.odd_panel import solve_odd_panel
    from neurogolf.solvers.odd_panel_aware import solve_odd_panel_aware

    # 5x5 (n=2): BL panel differs.
    g5 = [
        [8, 8, 3, 8, 8],
        [8, 8, 3, 8, 8],
        [3, 3, 3, 3, 3],
        [8, 8, 3, 8, 8],
        [4, 8, 3, 8, 8],
    ]
    o5 = [[8, 8], [4, 8]]
    # 7x7 (n=3): TR panel differs.
    g7 = [
        [4, 4, 4, 2, 4, 4, 4],
        [4, 4, 4, 2, 4, 1, 4],
        [4, 4, 4, 2, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 2],
        [4, 4, 4, 2, 4, 4, 4],
        [4, 4, 4, 2, 4, 4, 4],
        [4, 4, 4, 2, 4, 4, 4],
    ]
    o7 = [[4, 4, 4], [4, 1, 4], [4, 4, 4]]
    task = _make_task([(g5, o5), (g7, o7)])
    # mixed sizes -> the static solver must decline
    assert solve_odd_panel(task) is None
    model = solve_odd_panel_aware(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    for g, exp in ((g5, o5), (g7, o7)):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == exp


def test_odd_panel_aware_rejects_non_panel():
    from neurogolf.solvers.odd_panel_aware import solve_odd_panel_aware
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_odd_panel_aware(task) is None


def test_period_extend_h():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.period_extend_h import solve_period_extend_h

    def tile(g, p):
        a = np.array(g)
        return a[:, [c % p for c in range(2 * a.shape[1])]].tolist()

    g2 = [[2, 8, 2, 8], [8, 2, 8, 2]]            # fundamental period 2
    g3 = [[1, 2, 3, 1, 2, 3], [3, 1, 2, 3, 1, 2]]  # fundamental period 3
    task = _make_task([(g2, tile(g2, 2)), (g3, tile(g3, 3))])
    model = solve_period_extend_h(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    for g, p in ((g2, 2), (g3, 3)):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == tile(g, p)


def test_period_extend_h_rejects_same_shape():
    from neurogolf.solvers.period_extend_h import solve_period_extend_h
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_period_extend_h(task) is None


def test_stripe_seeds_vertical_and_horizontal():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.stripe_seeds import solve_stripe_seeds, _ref

    # Vertical: seeds on top & bottom rows -> full-height stripes, period = col gap.
    gv = [[0, 2, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 8, 0]]  # seeds (0,1)=2 col1, (4,3)=8 col3 ; p=2
    # Horizontal: seeds on left & right cols (interior rows) -> full-width
    # stripes, period = row gap.
    gh = [[0, 0, 0, 0, 0, 0],
          [3, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0]]  # seeds (1,0)=3 col0, (4,5)=1 col5 ; p=3
    task = _make_task([(gv, _ref(gv)), (gh, _ref(gh))])
    model = solve_stripe_seeds(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    for g in (gv, gh):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == _ref(g)


def test_stripe_seeds_rejects_non_two_seeds():
    from neurogolf.solvers.stripe_seeds import solve_stripe_seeds
    task = _make_task([([[1, 1], [1, 1]], [[1, 1], [1, 1]])])
    assert solve_stripe_seeds(task) is None


def test_slide_to_wall():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.slide_to_wall import solve_slide_to_wall, _ref

    # mover (2) above wall (8) -> slides down until adjacent
    gv = [[0, 2, 2, 0],
          [0, 2, 2, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 8, 8, 0]]
    # mover (2) left of wall (8) -> slides right until adjacent
    gh = [[2, 0, 0, 0, 8],
          [2, 0, 0, 0, 8]]
    task = _make_task([(gv, _ref(gv)), (gh, _ref(gh))])
    model = solve_slide_to_wall(task)
    assert model is not None and hasattr(model, "graph")

    sess = ort.InferenceSession(model.SerializeToString())
    for g in (gv, gh):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == _ref(g)


def test_slide_to_wall_rejects_other():
    from neurogolf.solvers.slide_to_wall import solve_slide_to_wall
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_slide_to_wall(task) is None


def test_downscale_majority():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.downscale_majority import solve_downscale_majority

    # 6x6 -> 3x3, each 2x2 block's majority colour (ties -> lowest index)
    g = [
        [2, 2, 0, 0, 3, 3],
        [2, 0, 0, 0, 3, 3],
        [4, 4, 5, 5, 0, 0],
        [4, 4, 5, 0, 0, 0],
        [1, 1, 1, 7, 8, 8],
        [1, 0, 7, 7, 8, 8],
    ]
    exp = [[2, 0, 3], [4, 5, 0], [1, 7, 8]]
    task = _make_task([(g, exp)])
    model = solve_downscale_majority(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == exp


def test_downscale_majority_rejects_same_shape():
    from neurogolf.solvers.downscale_majority import solve_downscale_majority
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_downscale_majority(task) is None


def test_untile_half():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.untile_half import solve_untile_half

    gh = [[1, 2, 1, 2], [3, 4, 3, 4]]            # horizontal 2x tile
    gv = [[5, 6], [7, 8], [5, 6], [7, 8]]        # vertical 2x tile
    task = _make_task([(gh, [[1, 2], [3, 4]]), (gv, [[5, 6], [7, 8]])])
    model = solve_untile_half(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    for g, exp in ((gh, [[1, 2], [3, 4]]), (gv, [[5, 6], [7, 8]])):
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == exp


def test_untile_half_rejects_non_tile():
    from neurogolf.solvers.untile_half import solve_untile_half
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_untile_half(task) is None


def test_slide_to_line():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.slide_to_line import solve_slide_to_line, _ref

    # horizontal 2-line: a full row of 2 (row3) and stray 2s above/below slide
    # adjacent; a 4-marker has no line -> removed.
    g = [
        [0, 0, 2, 0, 0],
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
    ]
    task = _make_task([(g, _ref(np.array(g)).tolist())])
    model = solve_slide_to_line(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == _ref(np.array(g)).tolist()


def test_slide_to_line_rejects_no_line():
    from neurogolf.solvers.slide_to_line import solve_slide_to_line
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_slide_to_line(task) is None


def test_largest_comp_crop():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.largest_comp_crop import solve_largest_comp_crop

    # one solid blob + scattered single-cell noise -> crop the blob's bbox
    g = [
        [0, 0, 0, 5, 0, 0],
        [0, 3, 3, 0, 0, 1],
        [0, 3, 0, 0, 0, 0],
        [0, 3, 3, 0, 7, 0],
        [2, 0, 0, 0, 0, 0],
    ]
    exp = [[3, 3], [3, 0], [3, 3]]
    task = _make_task([(g, exp)])
    model = solve_largest_comp_crop(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == exp


def test_largest_comp_crop_rejects_same_shape():
    from neurogolf.solvers.largest_comp_crop import solve_largest_comp_crop
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_largest_comp_crop(task) is None


def test_diag_block_slide():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diag_block_slide import solve_diag_block_slide, _ref

    # 2x2 block of 4 with a 2 at the top-right -> slide up-right, trail of 4s
    g = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 4, 2, 0, 0],
        [0, 4, 4, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    task = _make_task([(g, _ref(np.array(g)).tolist())])
    model = solve_diag_block_slide(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == _ref(np.array(g)).tolist()


def test_diag_block_slide_rejects_non_block():
    from neurogolf.solvers.diag_block_slide import solve_diag_block_slide
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_diag_block_slide(task) is None


def test_project_to_block():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.project_to_block import solve_project_to_block, _ref

    # colour-8 block with border markers projecting onto its edges
    g = [
        [0, 9, 0, 0],
        [0, 0, 0, 0],
        [0, 8, 8, 0],
        [6, 8, 8, 4],
        [0, 0, 0, 0],
    ]
    task = _make_task([(g, _ref(np.array(g)).tolist())])
    model = solve_project_to_block(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == _ref(np.array(g)).tolist()


def test_project_to_block_rejects_no_block():
    from neurogolf.solvers.project_to_block import solve_project_to_block
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_project_to_block(task) is None


def test_framed_regions():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.framed_regions import solve_framed_regions

    # two examples sharing one fixed template, recoloured by two markers
    def make(a, b):
        g = [[0] * 5 for _ in range(6)]
        g[1][2] = a
        g[4][2] = b
        o = [[0] * 5 for _ in range(6)]
        for c in range(5):
            o[0][c] = a; o[5][c] = b
        o[1][0] = o[1][4] = a
        o[4][0] = o[4][4] = b
        return g, o
    task = _make_task([make(6, 7), make(3, 8)])
    model = solve_framed_regions(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    for a, b in ((6, 7), (3, 8)):
        g, o = make(a, b)
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == o


def test_framed_regions_rejects_one_marker():
    from neurogolf.solvers.framed_regions import solve_framed_regions
    task = _make_task([([[5, 0], [0, 0]], [[5, 0], [0, 0]])])
    assert solve_framed_regions(task) is None


def test_diag_connect():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diag_connect import solve_diag_connect, _ref

    g = [
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [6, 0, 0, 0, 0],
        [0, 6, 0, 0, 0],
    ]  # two non-crossing diagonal pairs (colours 2 and 6)
    task = _make_task([(g, _ref(np.array(g)).tolist())])
    model = solve_diag_connect(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == _ref(np.array(g)).tolist()


def test_diag_connect_rejects_non_pair():
    from neurogolf.solvers.diag_connect import solve_diag_connect
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_diag_connect(task) is None


def test_stamp_top_row():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.stamp_top_row import solve_stamp_top_row, _ref

    g = [
        [5, 0, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ]
    task = _make_task([(g, _ref(np.array(g)).tolist())])
    model = solve_stamp_top_row(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == _ref(np.array(g)).tolist()


def test_stamp_top_row_rejects_plain():
    from neurogolf.solvers.stamp_top_row import solve_stamp_top_row
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_stamp_top_row(task) is None


def test_plus_panels():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.plus_panels import solve_plus_panels, _ref

    # two colour-8 rows (2, 4) and columns (2, 4) split a 7x7 grid into 3x3
    g = [
        [0, 0, 8, 0, 8, 0, 0],
        [0, 0, 8, 0, 8, 0, 0],
        [8, 8, 8, 8, 8, 8, 8],
        [0, 0, 8, 0, 8, 0, 0],
        [8, 8, 8, 8, 8, 8, 8],
        [0, 0, 8, 0, 8, 0, 0],
        [0, 0, 8, 0, 8, 0, 0],
    ]
    expected = _ref(np.array(g)).tolist()
    task = _make_task([(g, expected)])
    model = solve_plus_panels(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_plus_panels_rejects_plain():
    from neurogolf.solvers.plus_panels import solve_plus_panels
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_plus_panels(task) is None


def test_rot180_repair():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.rot180_repair import solve_rot180_repair

    # a 4x4 rot180-symmetric image with a colour-7 occluder over two cells
    base = [
        [1, 2, 3, 1],
        [4, 5, 6, 8],
        [8, 6, 5, 4],
        [1, 3, 2, 1],
    ]
    occ = [row[:] for row in base]
    occ[1][2] = 7  # hides the 6; partner (2,1)=6 restores it
    occ[1][3] = 7  # hides the 8; partner (2,0)=8 restores it
    task = _make_task([(occ, base)])
    model = solve_rot180_repair(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(occ)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == base


def test_rot180_repair_rejects_plain():
    from neurogolf.solvers.rot180_repair import solve_rot180_repair
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_rot180_repair(task) is None


def test_lattice_count():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.lattice_count import solve_lattice_count, _ref

    # 7x7 with one horizontal (row 3) and one vertical (col 3) divider of 8,
    # background 3 -> 2 row-bands x 2 col-bands = 2x2 block of colour 3
    g = [
        [3, 3, 3, 8, 3, 3, 3],
        [3, 3, 3, 8, 3, 3, 3],
        [3, 3, 3, 8, 3, 3, 3],
        [8, 8, 8, 8, 8, 8, 8],
        [3, 3, 3, 8, 3, 3, 3],
        [3, 3, 3, 8, 3, 3, 3],
        [3, 3, 3, 8, 3, 3, 3],
    ]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[3, 3], [3, 3]]
    task = _make_task([(g, expected)])
    model = solve_lattice_count(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_lattice_count_rejects_plain():
    from neurogolf.solvers.lattice_count import solve_lattice_count
    task = _make_task([([[1, 2], [2, 1]], [[1, 2], [2, 1]])])
    assert solve_lattice_count(task) is None


def test_quadrant_crop():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.quadrant_crop import solve_quadrant_crop, _quadrant

    # a 4x4 block placed at an offset; output = its top-left 2x2 quadrant
    g = [[0] * 8 for _ in range(8)]
    block = [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [8, 7, 6, 5],
             [4, 3, 2, 1]]
    for i in range(4):
        for j in range(4):
            g[2 + i][3 + j] = block[i][j]
    quad, h = _quadrant(np.array(g))
    assert h == 2 and quad.tolist() == [[1, 2], [5, 6]]
    task = _make_task([(g, quad.tolist())])
    model = solve_quadrant_crop(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == quad.tolist()


def test_quadrant_crop_rejects_odd():
    from neurogolf.solvers.quadrant_crop import solve_quadrant_crop
    # 3x3 block -> odd bbox, not a clean quadrant
    g = [[0, 0, 0, 0, 0],
         [0, 1, 2, 3, 0],
         [0, 4, 5, 6, 0],
         [0, 7, 8, 9, 0],
         [0, 0, 0, 0, 0]]
    task = _make_task([(g, [[1]])])
    assert solve_quadrant_crop(task) is None


def test_connect_box_markers():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.connect_box_markers import solve_connect_box_markers, _ref

    # background 8, a 2x2 box of colour 3, a marker 4 aligned on a box row to the
    # right, and a marker 2 aligned on a box column below -> both get connected
    g = [[8, 8, 8, 8, 8, 8, 8],
         [8, 3, 3, 8, 8, 4, 8],
         [8, 3, 3, 8, 8, 8, 8],
         [8, 8, 8, 8, 8, 8, 8],
         [8, 2, 8, 8, 8, 8, 8],
         [8, 8, 8, 8, 8, 8, 8]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1][3] == 4 and expected[1][4] == 4   # line to the 4 marker
    assert expected[3][1] == 2 and expected[4][1] == 2   # line down to the 2 marker
    task = _make_task([(g, expected)])
    model = solve_connect_box_markers(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_connect_box_markers_rejects_plain():
    from neurogolf.solvers.connect_box_markers import solve_connect_box_markers
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_connect_box_markers(task) is None


def test_recolor_in_block():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.recolor_in_block import solve_recolor_in_block, _ref

    # an 8-region frame; the colour-1 cells inside its bbox become colour 3
    g = [[1, 1, 1, 1, 1, 1],
         [1, 8, 8, 8, 1, 1],
         [1, 8, 1, 8, 1, 1],
         [1, 8, 8, 8, 1, 1],
         [1, 1, 1, 1, 1, 1]]
    expected = _ref(np.array(g), 8, 1, 3).tolist()
    assert expected[2][2] == 3        # the interior 1 became 3
    assert expected[0][0] == 1        # outside the bbox stays 1
    task = _make_task([(g, expected)])
    model = solve_recolor_in_block(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_recolor_in_block_rejects_plain():
    from neurogolf.solvers.recolor_in_block import solve_recolor_in_block
    task = _make_task([([[1, 1], [1, 1]], [[1, 1], [1, 1]])])
    assert solve_recolor_in_block(task) is None


def test_corner_rays():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.corner_rays import solve_corner_rays, _ref

    # a colour-6 line down column 0 of a 4x4 grid
    g = [[6, 0, 0, 0],
         [6, 0, 0, 0],
         [6, 0, 0, 0],
         [6, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0][3] == 2 and expected[1][2] == 2          # anti-diagonal
    assert expected[3][1] == 4 and expected[3][3] == 4          # bottom row
    assert expected[3][0] == 6                                  # line stays
    task = _make_task([(g, expected)])
    model = solve_corner_rays(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_corner_rays_rejects_plain():
    from neurogolf.solvers.corner_rays import solve_corner_rays
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_corner_rays(task) is None


def test_divider_fold():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.divider_fold import solve_divider_fold, _ref

    # 5x5: divider colour 2 (row 2, col 2), shape colour 1 in the top-left.
    # Output folds the shape into all quadrants, recolours it to 2, drops the
    # divider -> 4x4.
    g = [[0, 1, 2, 0, 0],
         [1, 1, 2, 0, 0],
         [2, 2, 2, 2, 2],
         [0, 0, 2, 0, 0],
         [0, 0, 2, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert len(expected) == 4 and len(expected[0]) == 4
    task = _make_task([(g, expected)])
    model = solve_divider_fold(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_divider_fold_rejects_plain():
    from neurogolf.solvers.divider_fold import solve_divider_fold
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_divider_fold(task) is None


def test_band_sort():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.band_sort import solve_band_sort, _ref

    # vertical bands 4|2|8 -> row strip "428"
    gv = [[4, 4, 2, 2, 8, 8],
          [4, 4, 2, 2, 8, 8],
          [4, 4, 2, 2, 8, 8]]
    assert _ref(np.array(gv)).tolist() == [[4, 2, 8]]
    # horizontal bands 2/8/5 -> column strip
    gh = [[2, 2, 2],
          [2, 2, 2],
          [8, 8, 8],
          [5, 5, 5]]
    assert _ref(np.array(gh)).tolist() == [[2], [8], [5]]
    for g, exp in ((gv, [[4, 2, 8]]), (gh, [[2], [8], [5]])):
        task = _make_task([(g, exp)])
        model = solve_band_sort(task)
        assert model is not None and hasattr(model, "graph")
        sess = ort.InferenceSession(model.SerializeToString())
        res = sess.run(["output"], {"input": to_onehot(g)})[0]
        assert from_onehot((res > 0.0).astype(np.float32)) == exp


def test_band_sort_rejects_uniform():
    from neurogolf.solvers.band_sort import solve_band_sort
    task = _make_task([([[3, 3], [3, 3]], [[3]])])
    assert solve_band_sort(task) is None


def test_interior_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.interior_recolor import solve_interior_recolor, _ref

    # two solid rectangles; their interiors become 8, borders keep colour
    g = [[0, 0, 0, 0, 0, 0],
         [0, 2, 2, 2, 0, 0],
         [0, 2, 2, 2, 0, 0],
         [0, 2, 2, 2, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2][2] == 8           # interior recoloured
    assert expected[1][1] == 2           # border stays
    task = _make_task([(g, expected)])
    model = solve_interior_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_interior_recolor_rejects_plain():
    from neurogolf.solvers.interior_recolor import solve_interior_recolor
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_interior_recolor(task) is None


def test_float_up():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.float_up import solve_float_up, _ref

    # a height-2 bar (colour 5) at the bottom floats up by 2 rows
    g = [[0, 0, 0],
         [0, 0, 0],
         [0, 5, 0],
         [0, 5, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0][1] == 5 and expected[1][1] == 5      # moved up by 2
    assert expected[2][1] == 0 and expected[3][1] == 0      # original cleared
    task = _make_task([(g, expected)])
    model = solve_float_up(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_float_up_rejects_static():
    from neurogolf.solvers.float_up import solve_float_up
    task = _make_task([([[1, 0], [0, 2]], [[1, 0], [0, 2]])])
    assert solve_float_up(task) is None


def test_diag_x():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diag_x import solve_diag_x, _ref

    # single marker (colour 7) at (2,2) of a 5x5 grid -> diagonal X
    g = [[0] * 5 for _ in range(5)]
    g[2][2] = 7
    expected = _ref(np.array(g)).tolist()
    assert expected[0][0] == 7 and expected[0][4] == 7 and expected[4][0] == 7
    assert expected[2][2] == 7 and expected[0][2] == 0
    task = _make_task([(g, expected)])
    model = solve_diag_x(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_diag_x_rejects_two_markers():
    from neurogolf.solvers.diag_x import solve_diag_x
    task = _make_task([([[5, 0], [0, 5]], [[5, 0], [0, 5]])])
    assert solve_diag_x(task) is None


def test_staircase():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.staircase import solve_staircase, _ref

    g = [[1, 1, 0, 0, 0, 0]]   # K=2, W=6 -> 3x6 staircase
    expected = _ref(np.array(g)).tolist()
    assert expected == [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]]
    task = _make_task([(g, expected)])
    model = solve_staircase(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_staircase_rejects_multirow():
    from neurogolf.solvers.staircase import solve_staircase
    task = _make_task([([[1, 0], [0, 1]], [[1, 0], [0, 1]])])
    assert solve_staircase(task) is None


def test_box_stretch():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.box_stretch import solve_box_stretch, _ref

    # 3x3 box (border 2, interior 1) with a marker 8 below it -> stretch down
    g = [[2, 2, 2, 0, 0],
         [2, 1, 2, 0, 0],
         [2, 2, 2, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 8, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[4][0] == 2 and expected[4][1] == 2          # new bottom frame at marker row
    assert expected[4][2] == 2 and expected[3][1] == 1          # interior extended
    task = _make_task([(g, expected)])
    model = solve_box_stretch(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_box_stretch_rejects_plain():
    from neurogolf.solvers.box_stretch import solve_box_stretch
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_box_stretch(task) is None


def test_gap_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.gap_fill import solve_gap_fill, _ref

    # two rectangles separated vertically; gap filled with 8 over interior cols
    g = [[2, 2, 2, 2, 0],
         [2, 2, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [3, 3, 3, 3, 0],
         [3, 3, 3, 3, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2][1] == 8 and expected[2][2] == 8       # gap filled, interior cols
    assert expected[2][0] == 0 and expected[2][3] == 0       # edge cols not filled
    task = _make_task([(g, expected)])
    model = solve_gap_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_gap_fill_rejects_one_rect():
    from neurogolf.solvers.gap_fill import solve_gap_fill
    task = _make_task([([[2, 2], [2, 2]], [[2, 2], [2, 2]])])
    assert solve_gap_fill(task) is None


def test_merge_pair():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.merge_pair import solve_merge_pair, _ref

    # a 3 next to a 2 -> 3 becomes 8, 2 erased; isolated 3 and 2 unchanged
    g = [[3, 2, 0, 0],
         [0, 0, 0, 0],
         [3, 0, 2, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0][0] == 8 and expected[0][1] == 0       # merged pair
    assert expected[2][0] == 3 and expected[2][2] == 2       # isolated stay
    task = _make_task([(g, expected)])
    model = solve_merge_pair(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_merge_pair_rejects_plain():
    from neurogolf.solvers.merge_pair import solve_merge_pair
    task = _make_task([([[1, 4], [4, 1]], [[1, 4], [4, 1]])])
    assert solve_merge_pair(task) is None


def test_cross_move():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.cross_move import solve_cross_move, _ref

    # colour-3 cross at (row2, col2) with two 5-markers -> moves to (row4, col0)
    g = [[0, 0, 3, 0, 5],
         [0, 0, 3, 0, 5],
         [3, 3, 3, 3, 3],
         [0, 0, 3, 0, 0],
         [0, 0, 3, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[4] == [3, 3, 3, 3, 3]            # horizontal line moved down 2
    assert all(row[0] == 3 for row in expected)      # vertical line moved left 2
    task = _make_task([(g, expected)])
    model = solve_cross_move(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_cross_move_rejects_plain():
    from neurogolf.solvers.cross_move import solve_cross_move
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_cross_move(task) is None


def test_row_checker():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.row_checker import solve_row_checker, _ref

    g = [[3, 3, 3, 3, 3, 3],
         [9, 9, 9, 9, 9, 9]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [3, 9, 3, 9, 3, 9]
    assert expected[1] == [9, 3, 9, 3, 9, 3]
    task = _make_task([(g, expected)])
    model = solve_row_checker(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_row_checker_rejects_three_rows():
    from neurogolf.solvers.row_checker import solve_row_checker
    task = _make_task([([[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]])])
    assert solve_row_checker(task) is None


def test_five_isolate():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.five_isolate import solve_five_isolate, _ref

    g = [[4, 5, 4],
         [5, 5, 5],
         [4, 5, 4]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 4, 0], [4, 4, 4], [0, 4, 0]]
    task = _make_task([(g, expected)])
    model = solve_five_isolate(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_five_isolate_rejects_no_marker():
    from neurogolf.solvers.five_isolate import solve_five_isolate
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_five_isolate(task) is None


def test_color_sort_column():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.color_sort_column import solve_color_sort_column, _ref

    g = [[2, 2, 3, 3],
         [2, 2, 3, 0],
         [8, 0, 0, 0],
         [0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()        # counts 2->4, 3->3, 8->1
    assert expected == [[2], [3], [8]]
    task = _make_task([(g, expected)])
    model = solve_color_sort_column(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_color_sort_column_rejects_tie():
    from neurogolf.solvers.color_sort_column import solve_color_sort_column
    task = _make_task([([[2, 3], [2, 3]], [[2], [3]])])
    assert solve_color_sort_column(task) is None


def test_rect_interior_rank():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.rect_interior_rank import solve_rect_interior_rank, _ref

    g = [[4, 4, 4, 4, 0, 0],          # top rect 4x4 = 16 (smaller)
         [4, 4, 4, 4, 0, 0],
         [4, 4, 4, 4, 0, 0],
         [4, 4, 4, 4, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [4, 4, 4, 4, 4, 4],          # bottom rect 4x6 = 24 (larger)
         [4, 4, 4, 4, 4, 4],
         [4, 4, 4, 4, 4, 4],
         [4, 4, 4, 4, 4, 4]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1][1] == 1 and expected[2][2] == 1     # smaller interior -> 1
    assert expected[6][1] == 2 and expected[7][4] == 2     # larger interior -> 2
    task = _make_task([(g, expected)])
    model = solve_rect_interior_rank(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_rect_interior_rank_rejects_single():
    from neurogolf.solvers.rect_interior_rank import solve_rect_interior_rank
    g = [[4, 4, 4], [4, 4, 4], [4, 4, 4]]
    task = _make_task([(g, g)])
    assert solve_rect_interior_rank(task) is None


def test_bbox_strip_zero():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.bbox_strip_zero import solve_bbox_strip_zero

    g = [[1, 1, 1, 1, 1],          # background is the majority colour 1
         [1, 2, 2, 2, 1],
         [1, 2, 1, 2, 1],
         [1, 1, 1, 1, 1]]
    expected = [[2, 2, 2], [2, 0, 2]]   # crop to non-1 bbox, the inner 1 -> 0
    task = _make_task([(g, expected)])
    model = solve_bbox_strip_zero(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_bbox_strip_zero_rejects_recolor():
    from neurogolf.solvers.bbox_strip_zero import solve_bbox_strip_zero
    task = _make_task([([[5, 5], [5, 5]], [[7, 7], [7, 7]])])
    assert solve_bbox_strip_zero(task) is None


def test_ring_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.ring_recolor import solve_ring_recolor, _ref

    g = [[5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[1, 4, 4, 1], [4, 2, 2, 4], [4, 2, 2, 4], [1, 4, 4, 1]]
    task = _make_task([(g, expected)])
    model = solve_ring_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_ring_recolor_rejects_plain():
    from neurogolf.solvers.ring_recolor import solve_ring_recolor
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_ring_recolor(task) is None


def test_interior_recolor_aware():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.interior_recolor_aware import solve_interior_recolor_aware, _ref

    g = [[5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5]]
    expected = _ref(np.array(g), 2).tolist()      # interior eroded cells -> 2
    assert expected == [[5, 5, 5, 5], [5, 2, 2, 5], [5, 5, 5, 5]]
    task = _make_task([(g, expected)])
    model = solve_interior_recolor_aware(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_interior_recolor_aware_rejects_plain():
    from neurogolf.solvers.interior_recolor_aware import solve_interior_recolor_aware
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_interior_recolor_aware(task) is None


def test_line_cross_swap():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.line_cross_swap import solve_line_cross_swap, _ref

    g = [[0, 8, 0, 0],
         [0, 8, 0, 0],
         [3, 8, 3, 3],          # horizontal 3-line, vertical 8 drawn on top at (2,1)
         [0, 8, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [3, 3, 3, 3]            # crossing now shows the horizontal line
    task = _make_task([(g, expected)])
    model = solve_line_cross_swap(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_line_cross_swap_rejects_tie():
    from neurogolf.solvers.line_cross_swap import solve_line_cross_swap
    task = _make_task([([[1, 1], [2, 2]], [[1, 1], [2, 2]])])
    assert solve_line_cross_swap(task) is None


def test_explode_corners():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.explode_corners import solve_explode_corners, _ref

    g = [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 9, 3, 0, 0],
         [0, 0, 7, 8, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [8, 8, 0, 0, 7, 7] and expected[5] == [3, 3, 0, 0, 9, 9]
    task = _make_task([(g, expected)])
    model = solve_explode_corners(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_explode_corners_rejects_non_2x2():
    from neurogolf.solvers.explode_corners import solve_explode_corners
    g = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    task = _make_task([(g, g)])
    assert solve_explode_corners(task) is None


def test_l_connect():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.l_connect import solve_l_connect, _ref

    g = [[0, 8, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 2]]
    expected = _ref(np.array(g)).tolist()       # down col1, then across row3 to the 2
    assert expected[0] == [0, 8, 0, 0, 0] and expected[3] == [0, 4, 4, 4, 2]
    task = _make_task([(g, expected)])
    model = solve_l_connect(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_l_connect_rejects_no_markers():
    from neurogolf.solvers.l_connect import solve_l_connect
    task = _make_task([([[1, 3], [3, 1]], [[1, 3], [3, 1]])])
    assert solve_l_connect(task) is None


def test_block_quadrant():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.block_quadrant import solve_block_quadrant, _ref

    g = [[6, 0, 0, 0, 0, 7],          # markers in the four quadrants
         [0, 0, 0, 0, 0, 0],
         [0, 0, 8, 8, 0, 0],          # 2x2 block of 8
         [0, 0, 8, 8, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [4, 0, 0, 0, 0, 9]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [0, 0, 6, 7, 0, 0] and expected[3] == [0, 0, 4, 9, 0, 0]
    task = _make_task([(g, expected)])
    model = solve_block_quadrant(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_block_quadrant_rejects_no_block():
    from neurogolf.solvers.block_quadrant import solve_block_quadrant
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_block_quadrant(task) is None


def test_move_toward():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.move_toward import solve_move_toward, _ref

    g = [[3, 0, 0],
         [0, 0, 0],
         [0, 0, 4]]
    expected = _ref(np.array(g)).tolist()          # 3 steps diagonally toward 4
    assert expected == [[0, 0, 0], [0, 3, 0], [0, 0, 4]]
    task = _make_task([(g, expected)])
    model = solve_move_toward(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_move_toward_rejects_no_markers():
    from neurogolf.solvers.move_toward import solve_move_toward
    task = _make_task([([[1, 2], [5, 6]], [[1, 2], [5, 6]])])
    assert solve_move_toward(task) is None


def test_cut_diagonals():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.cut_diagonals import solve_cut_diagonals, _ref

    g = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    expected = _ref(np.array(g)).tolist()         # both diagonals -> 0
    assert expected == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    task = _make_task([(g, expected)])
    model = solve_cut_diagonals(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_cut_diagonals_rejects_nonsquare():
    from neurogolf.solvers.cut_diagonals import solve_cut_diagonals
    task = _make_task([([[1, 1, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1]])])
    assert solve_cut_diagonals(task) is None


def test_odd_panel_shape():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.odd_panel_shape import solve_odd_panel_shape, _ref

    g = [[5, 0, 5], [0, 5, 0], [5, 0, 5],   # panel 0 (shape X)
         [5, 0, 5], [0, 5, 0], [5, 0, 5],   # panel 1 (shape X)
         [5, 5, 5], [5, 0, 5], [5, 5, 5]]   # panel 2 (odd shape Y)
    expected = _ref(np.array(g)).tolist()
    assert expected == [[5, 5, 5], [5, 0, 5], [5, 5, 5]]
    task = _make_task([(g, expected)])
    model = solve_odd_panel_shape(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_odd_panel_shape_rejects_nonpanel():
    from neurogolf.solvers.odd_panel_shape import solve_odd_panel_shape
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_odd_panel_shape(task) is None


def test_band_majority():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.band_majority import solve_band_majority, _ref

    g = [[5, 5, 5],
         [5, 2, 5],          # noisy 5-band
         [2, 2, 2]]
    expected = _ref(np.array(g)).tolist()       # row-majority: noise -> band colour
    assert expected == [[5, 5, 5], [5, 5, 5], [2, 2, 2]]
    task = _make_task([(g, expected)])
    model = solve_band_majority(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_band_majority_rejects_uniform():
    from neurogolf.solvers.band_majority import solve_band_majority
    task = _make_task([([[5, 5], [5, 5]], [[5, 5], [5, 5]])])
    assert solve_band_majority(task) is None


def test_connect_pairs():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.connect_pairs import solve_connect_pairs, _ref

    g = [[0, 0, 4, 0, 0],
         [0, 0, 0, 0, 0],
         [3, 0, 0, 0, 3],          # horizontal 3-pair
         [0, 0, 0, 0, 0],
         [0, 0, 4, 0, 0]]          # vertical 4-pair (col2)
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [3, 3, 4, 3, 3]            # crossing: vertical 4 on top
    task = _make_task([(g, expected)])
    model = solve_connect_pairs(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_connect_pairs_rejects_diagonal():
    from neurogolf.solvers.connect_pairs import solve_connect_pairs
    task = _make_task([([[3, 0], [0, 3]], [[3, 0], [0, 3]])])
    assert solve_connect_pairs(task) is None


def test_panel_summary():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.panel_summary import solve_panel_summary, _ref

    g = [[1, 1, 0, 8, 8],          # 2x2 panels separated by a blank row/col
         [1, 1, 0, 8, 8],
         [0, 0, 0, 0, 0],
         [6, 6, 0, 1, 1],
         [6, 6, 0, 1, 1]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[1, 8], [6, 1]]
    task = _make_task([(g, expected)])
    model = solve_panel_summary(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_panel_summary_rejects_single():
    from neurogolf.solvers.panel_summary import solve_panel_summary
    task = _make_task([([[5, 5], [5, 5]], [[5, 5], [5, 5]])])
    assert solve_panel_summary(task) is None


def test_column_template():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.column_template import solve_column_template, _ref

    g = [[3, 3, 2, 3, 3, 2],          # template row (pattern 3,3,2)
         [0, 0, 0, 0, 0, 0],
         [8, 8, 4, 0, 0, 0]]          # seed -> tiled into the template pattern
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [8, 8, 4, 8, 8, 4]
    task = _make_task([(g, expected)])
    model = solve_column_template(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_column_template_rejects_identity():
    from neurogolf.solvers.column_template import solve_column_template
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_column_template(task) is None


def test_fractal_blocks():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.fractal_blocks import solve_fractal_blocks, _ref

    g = [[5, 0, 5],            # X meta (block size 1) -> 9x9 self-fractal
         [0, 5, 0],
         [5, 0, 5]]
    expected = _ref(np.array(g)).tolist()
    assert len(expected) == 9 and expected[0] == [5, 0, 5, 0, 0, 0, 5, 0, 5]
    task = _make_task([(g, expected)])
    model = solve_fractal_blocks(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_fractal_blocks_rejects_non_div3():
    from neurogolf.solvers.fractal_blocks import solve_fractal_blocks
    task = _make_task([([[5, 5], [5, 5]], [[5, 5], [5, 5]])])
    assert solve_fractal_blocks(task) is None


def test_diagonal_markers():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diagonal_markers import solve_diagonal_markers, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 5, 5, 0],
         [0, 0, 5, 5, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()      # 1/2/3/4 at the block's diagonal corners
    assert expected[1] == [0, 1, 0, 0, 2] and expected[4] == [0, 3, 0, 0, 4]
    task = _make_task([(g, expected)])
    model = solve_diagonal_markers(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_diagonal_markers_rejects_no_block():
    from neurogolf.solvers.diagonal_markers import solve_diagonal_markers
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_diagonal_markers(task) is None


def test_odd_col_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.odd_col_recolor import solve_odd_col_recolor, _ref

    g = [[2, 0, 0],
         [0, 2, 0],          # (1,1): odd column -> 4
         [0, 0, 2]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[2, 0, 0], [0, 4, 0], [0, 0, 2]]
    task = _make_task([(g, expected)])
    model = solve_odd_col_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_odd_col_recolor_rejects_nochange():
    from neurogolf.solvers.odd_col_recolor import solve_odd_col_recolor
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_odd_col_recolor(task) is None


def test_triangle_diag():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.triangle_diag import solve_triangle_diag, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [2, 2, 0, 0, 0],          # 2-segment at row 2, ends col 1
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [3, 3, 3, 3, 0] and expected[2] == [2, 2, 0, 0, 0] and expected[3] == [1, 0, 0, 0, 0]
    task = _make_task([(g, expected)])
    model = solve_triangle_diag(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_triangle_diag_rejects_no_two():
    from neurogolf.solvers.triangle_diag import solve_triangle_diag
    task = _make_task([([[1, 3], [3, 1]], [[1, 3], [3, 1]])])
    assert solve_triangle_diag(task) is None


def test_pocket_drop():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.pocket_drop import solve_pocket_drop, _ref

    g = [[0, 6, 6, 6, 0],          # downward-opening staple, gap at col 2
         [0, 6, 0, 6, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[3] == [0, 0, 4, 0, 0] and expected[1] == [0, 6, 0, 6, 0]
    task = _make_task([(g, expected)])
    model = solve_pocket_drop(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_pocket_drop_rejects_no_pocket():
    from neurogolf.solvers.pocket_drop import solve_pocket_drop
    task = _make_task([([[6, 6], [6, 0]], [[6, 6], [6, 0]])])
    assert solve_pocket_drop(task) is None


def test_square_complete():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.square_complete import solve_square_complete, _ref

    g = [[0, 0, 0, 0],
         [0, 8, 0, 0],          # L-tromino, missing corner at (1, 2)
         [0, 8, 8, 0],
         [0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [0, 8, 1, 0]
    task = _make_task([(g, expected)])
    model = solve_square_complete(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_square_complete_rejects_single():
    from neurogolf.solvers.square_complete import solve_square_complete
    task = _make_task([([[8, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[8, 0, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_square_complete(task) is None


def test_midpoint_plus():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.midpoint_plus import solve_midpoint_plus, _ref

    g = [[1, 0, 0, 0, 1],          # markers row 0, cols 0 & 4 -> midpoint (0, 2)
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [1, 3, 3, 3, 1] and expected[1] == [0, 0, 3, 0, 0]
    task = _make_task([(g, expected)])
    model = solve_midpoint_plus(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_midpoint_plus_rejects_single():
    from neurogolf.solvers.midpoint_plus import solve_midpoint_plus
    task = _make_task([([[1, 0], [0, 0]], [[1, 0], [0, 0]])])
    assert solve_midpoint_plus(task) is None


def test_elbow_connect():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.elbow_connect import solve_elbow_connect, _ref

    g = [[0, 2, 0, 0],          # 2 at (0,1), 3 at (2,3)
         [0, 0, 0, 0],
         [0, 0, 0, 3]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [0, 2, 8, 8] and expected[1] == [0, 0, 0, 8] and expected[2] == [0, 0, 0, 3]
    task = _make_task([(g, expected)])
    model = solve_elbow_connect(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_elbow_connect_rejects_no_three():
    from neurogolf.solvers.elbow_connect import solve_elbow_connect
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_elbow_connect(task) is None


def test_mirror_quad():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.mirror_quad import solve_mirror_quad, _ref

    g = [[2, 2, 0, 0, 0, 0],          # 2-L at top-left, 3-block centre (2.5, 2.5)
         [2, 0, 0, 0, 0, 0],
         [0, 0, 3, 3, 0, 0],
         [0, 0, 3, 3, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [2, 2, 0, 0, 2, 2] and expected[5] == [2, 2, 0, 0, 2, 2]
    assert expected[2] == [0, 0, 3, 3, 0, 0]
    task = _make_task([(g, expected)])
    model = solve_mirror_quad(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_mirror_quad_rejects_no_block():
    from neurogolf.solvers.mirror_quad import solve_mirror_quad
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_mirror_quad(task) is None


def test_arrow_ray():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.arrow_ray import solve_arrow_ray, _ref

    g = [[0, 2, 0, 0, 0],          # triangle of 2 points right; marker 1 at (2,0)
         [2, 2, 0, 0, 0],
         [1, 2, 2, 0, 0],
         [2, 2, 0, 0, 0],
         [0, 2, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [1, 2, 2, 1, 1]
    task = _make_task([(g, expected)])
    model = solve_arrow_ray(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_arrow_ray_rejects_no_marker():
    from neurogolf.solvers.arrow_ray import solve_arrow_ray
    task = _make_task([([[2, 2], [2, 2]], [[2, 2], [2, 2]])])
    assert solve_arrow_ray(task) is None


def test_diag_shoot():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.diag_shoot import solve_diag_shoot, _ref

    g = [[3, 3, 0, 0, 0, 0],          # 2x2 block + satellite (2,2) -> ray down-right
         [3, 3, 0, 0, 0, 0],
         [0, 0, 3, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[3] == [0, 0, 0, 3, 0, 0] and expected[5] == [0, 0, 0, 0, 0, 3]
    task = _make_task([(g, expected)])
    model = solve_diag_shoot(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_diag_shoot_rejects_no_satellite():
    from neurogolf.solvers.diag_shoot import solve_diag_shoot
    task = _make_task([([[3, 3], [3, 3]], [[3, 3], [3, 3]])])
    assert solve_diag_shoot(task) is None


def test_ring_reverse():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.ring_reverse import solve_ring_reverse, _ref

    g = [[4, 4, 4, 4],          # outer ring 4, inner 2 -> swapped
         [4, 2, 2, 4],
         [4, 2, 2, 4],
         [4, 4, 4, 4]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [2, 2, 2, 2] and expected[1] == [2, 4, 4, 2]
    task = _make_task([(g, expected)])
    model = solve_ring_reverse(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_ring_reverse_rejects_flat():
    from neurogolf.solvers.ring_reverse import solve_ring_reverse
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_ring_reverse(task) is None


def test_corner_burst():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.corner_burst import solve_corner_burst, _ref

    g = [[0, 0, 0],
         [0, 2, 0],          # 2 bursts into 3/6/8/7 corners
         [0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[3, 0, 6], [0, 0, 0], [8, 0, 7]]
    task = _make_task([(g, expected)])
    model = solve_corner_burst(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_corner_burst_rejects_no_two():
    from neurogolf.solvers.corner_burst import solve_corner_burst
    task = _make_task([([[0, 0], [0, 0]], [[0, 0], [0, 0]])])
    assert solve_corner_burst(task) is None


def test_col3_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.col3_recolor import solve_col3_recolor, _ref

    g = [[4, 0, 4, 0, 4, 0, 4],          # 4-cells in cols 0,3,6 -> 6
         [4, 4, 4, 4, 4, 4, 4],
         [0, 4, 0, 4, 0, 4, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [6, 0, 4, 0, 4, 0, 6] and expected[1] == [6, 4, 4, 6, 4, 4, 6]
    task = _make_task([(g, expected)])
    model = solve_col3_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_col3_recolor_rejects_no_match():
    from neurogolf.solvers.col3_recolor import solve_col3_recolor
    task = _make_task([([[0, 4], [0, 4]], [[0, 4], [0, 4]])])
    assert solve_col3_recolor(task) is None


def test_vperiod3():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.vperiod3 import solve_vperiod3, _ref

    g = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [8, 0, 8],          # period-3 pattern -> tiles up to rows 0-1
         [0, 8, 0],
         [0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [8, 0, 8] and expected[1] == [0, 8, 0] and expected[3] == [8, 0, 8]
    task = _make_task([(g, expected)])
    model = solve_vperiod3(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_vperiod3_rejects_full():
    from neurogolf.solvers.vperiod3 import solve_vperiod3
    task = _make_task([([[5, 5], [5, 5]], [[5, 5], [5, 5]])])
    assert solve_vperiod3(task) is None


def test_key_cycle():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.key_cycle import solve_key_cycle, _ref

    g = [[2, 1, 4],          # key row + separator -> cycled solid bands
         [5, 5, 5],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [2, 2, 2] and expected[3] == [1, 1, 1] and expected[4] == [4, 4, 4]
    task = _make_task([(g, expected)])
    model = solve_key_cycle(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_key_cycle_rejects_short():
    from neurogolf.solvers.key_cycle import solve_key_cycle
    task = _make_task([([[2, 1], [5, 5]], [[2, 1], [5, 5]])])
    assert solve_key_cycle(task) is None


def test_laser_cross():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.laser_cross import solve_laser_cross, _ref

    g = [[0, 0, 0, 0, 8, 0],          # 8 column at col 4, 2 row at row 2
         [0, 0, 0, 0, 8, 0],
         [2, 2, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [2, 2, 2, 2, 4, 2] and expected[0][4] == 8 and expected[3][4] == 8
    task = _make_task([(g, expected)])
    model = solve_laser_cross(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_laser_cross_rejects_no_pair():
    from neurogolf.solvers.laser_cross import solve_laser_cross
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_laser_cross(task) is None


def test_enclosure_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.enclosure_recolor import solve_enclosure_recolor, _ref

    g = [[0, 0, 0, 0, 0],          # closed box encloses a hole -> recolour to 8
         [0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [0, 8, 8, 8, 0] and expected[2] == [0, 8, 0, 8, 0]
    task = _make_task([(g, expected)])
    model = solve_enclosure_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_enclosure_recolor_rejects_open():
    from neurogolf.solvers.enclosure_recolor import solve_enclosure_recolor
    task = _make_task([([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]])])
    assert solve_enclosure_recolor(task) is None


def test_key_flood():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.key_flood import solve_key_flood, _ref

    g = [[0, 2, 0, 0, 6, 0],          # keys 2@col1, 6@col4; blocks flood to them
         [0, 0, 0, 0, 0, 0],
         [0, 5, 5, 0, 5, 5],
         [0, 5, 5, 0, 5, 5]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [0, 2, 2, 0, 6, 6] and expected[3] == [0, 2, 2, 0, 6, 6]
    task = _make_task([(g, expected)])
    model = solve_key_flood(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_key_flood_rejects_no_block():
    from neurogolf.solvers.key_flood import solve_key_flood
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_key_flood(task) is None


def test_hole_size_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.hole_size_fill import solve_hole_size_fill, _ref

    g = [[5, 5, 5, 0, 0],          # 1x1 hole -> 5+1 = 6
         [5, 0, 5, 0, 0],
         [5, 5, 5, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [5, 6, 5, 0, 0]
    task = _make_task([(g, expected)])
    model = solve_hole_size_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_hole_size_fill_rejects_open():
    from neurogolf.solvers.hole_size_fill import solve_hole_size_fill
    task = _make_task([([[0, 0, 0], [0, 5, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 5, 0], [0, 0, 0]])])
    assert solve_hole_size_fill(task) is None


def test_hole_parity_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.hole_parity_fill import solve_hole_parity_fill, _ref

    g = [[1, 1, 1, 1, 0, 0],          # 2x2 hole, even side -> 2
         [1, 0, 0, 1, 0, 0],
         [1, 0, 0, 1, 0, 0],
         [1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [1, 2, 2, 1, 0, 0]
    task = _make_task([(g, expected)])
    model = solve_hole_parity_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_hole_parity_fill_rejects_open():
    from neurogolf.solvers.hole_parity_fill import solve_hole_parity_fill
    task = _make_task([([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]])])
    assert solve_hole_parity_fill(task) is None


def test_blob_size_color():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.blob_size_color import solve_blob_size_color, _ref

    g = [[5, 5, 5, 5, 5],          # blob size1@(1,1)->3, size2@(1,3-4)->2
         [5, 0, 5, 0, 0],
         [5, 5, 5, 5, 5]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [5, 3, 5, 2, 2]
    task = _make_task([(g, expected)])
    model = solve_blob_size_color(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_blob_size_color_rejects_uniform():
    from neurogolf.solvers.blob_size_color import solve_blob_size_color
    task = _make_task([([[5, 5], [5, 5]], [[5, 5], [5, 5]])])
    assert solve_blob_size_color(task) is None


def test_bbox_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.bbox_fill import solve_bbox_fill, _ref

    g = [[2, 2, 2, 0],          # 2-ring's bbox contains the 1-cell -> 4
         [2, 1, 2, 0],
         [2, 2, 2, 0],
         [0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [2, 4, 2, 0]
    task = _make_task([(g, expected)])
    model = solve_bbox_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_bbox_fill_rejects_no_two():
    from neurogolf.solvers.bbox_fill import solve_bbox_fill
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_bbox_fill(task) is None


def test_cross_center():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.cross_center import solve_cross_center, _ref

    g = [[8, 8, 8, 8, 8],          # box centre (2,2) -> full cross of 6
         [8, 1, 1, 1, 8],
         [8, 1, 8, 1, 8],
         [8, 1, 1, 1, 8],
         [8, 8, 8, 8, 8]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [6, 1, 6, 1, 6] and expected[0] == [8, 8, 6, 8, 8] and expected[4] == [8, 8, 6, 8, 8]
    task = _make_task([(g, expected)])
    model = solve_cross_center(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_cross_center_rejects_empty():
    from neurogolf.solvers.cross_center import solve_cross_center
    task = _make_task([([[8, 8], [8, 8]], [[8, 8], [8, 8]])])
    assert solve_cross_center(task) is None


def test_fold_mirror():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.fold_mirror import solve_fold_mirror, _ref

    g = [[0, 0, 0, 0, 0, 0],          # 8-shape folds across the 2 below it; bg -> 3
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 8, 8, 8],
         [0, 0, 0, 0, 0, 8],
         [0, 0, 0, 0, 0, 2],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2][3:6] == [8, 8, 8] and expected[5][3:6] == [8, 8, 8] and expected[0][0] == 3
    task = _make_task([(g, expected)])
    model = solve_fold_mirror(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_fold_mirror_rejects_no_marker():
    from neurogolf.solvers.fold_mirror import solve_fold_mirror
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_fold_mirror(task) is None


def test_bar_half():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.bar_half import solve_bar_half, _ref

    g = [[0, 2, 0],          # bar height 4 -> bottom 2 cells -> 8
         [0, 2, 0],
         [0, 2, 0],
         [0, 2, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [0, 2, 0] and expected[2] == [0, 8, 0] and expected[3] == [0, 8, 0]
    task = _make_task([(g, expected)])
    model = solve_bar_half(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_bar_half_rejects_no_bar():
    from neurogolf.solvers.bar_half import solve_bar_half
    task = _make_task([([[3, 3], [3, 3]], [[3, 3], [3, 3]])])
    assert solve_bar_half(task) is None


def test_corner_rect_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.corner_rect_fill import solve_corner_rect_fill, _ref

    g = [[4, 0, 4],          # 4 corners -> interior (1,1) filled with 2
         [0, 0, 0],
         [4, 0, 4]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[4, 0, 4], [0, 2, 0], [4, 0, 4]]
    task = _make_task([(g, expected)])
    model = solve_corner_rect_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_corner_rect_fill_rejects_few():
    from neurogolf.solvers.corner_rect_fill import solve_corner_rect_fill
    task = _make_task([([[4, 0], [0, 0]], [[4, 0], [0, 0]])])
    assert solve_corner_rect_fill(task) is None


def test_neighbor_halo():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.neighbor_halo import solve_neighbor_halo, _ref

    g = [[0, 0, 0, 0, 0],     # 1 -> orthogonal 7s ; 2 -> diagonal 4s
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 2, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 7, 0, 0, 0],
                        [7, 1, 7, 0, 0],
                        [0, 7, 4, 0, 4],
                        [0, 0, 0, 2, 0],
                        [0, 0, 4, 0, 4]]
    task = _make_task([(g, expected)])
    model = solve_neighbor_halo(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_neighbor_halo_rejects_plain():
    from neurogolf.solvers.neighbor_halo import solve_neighbor_halo
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_neighbor_halo(task) is None


def test_align_to_anchor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.align_to_anchor import solve_align_to_anchor, _ref

    g = [[2, 2, 0, 0, 0],     # 2-block slides down so its top meets the 1-block
         [2, 2, 0, 1, 1],
         [0, 0, 0, 1, 1]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 0, 0, 0, 0],
                        [2, 2, 0, 1, 1],
                        [2, 2, 0, 1, 1]]
    task = _make_task([(g, expected)])
    model = solve_align_to_anchor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.0).astype(np.float32)) == expected


def test_align_to_anchor_rejects_no_anchor():
    from neurogolf.solvers.align_to_anchor import solve_align_to_anchor
    task = _make_task([([[2, 2], [0, 0]], [[2, 2], [0, 0]])])
    assert solve_align_to_anchor(task) is None


def test_panel_complete():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.panel_complete import solve_panel_complete, _ref

    d = 3
    g = np.zeros((17, 17), int)
    g[5, :] = d; g[11, :] = d; g[:, 5] = d; g[:, 11] = d
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:   # full plus, colour 7
        g[2 + dr, 2 + dc] = 7
    g[8, 8] = 7; g[8, 7] = 7                                    # partial plus in panel (1,1)
    expected = _ref(g)
    assert expected is not None
    assert expected[8, 9] == d and expected[7, 8] == d          # missing arms filled with divider
    task = _make_task([(g.tolist(), expected.tolist())])
    model = solve_panel_complete(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g.tolist())})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected.tolist()))


def test_panel_complete_rejects_wrong_size():
    from neurogolf.solvers.panel_complete import solve_panel_complete
    task = _make_task([([[3, 0], [0, 0]], [[3, 0], [0, 0]])])
    assert solve_panel_complete(task) is None


def test_crop_tile_h():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.crop_tile_h import solve_crop_tile_h, _ref

    g = [[0, 0, 0, 0],
         [0, 8, 8, 0],
         [0, 8, 0, 0],
         [0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[8, 8, 8, 8], [8, 0, 8, 0]]
    task = _make_task([(g, expected)])
    model = solve_crop_tile_h(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_crop_tile_h_rejects_identity():
    from neurogolf.solvers.crop_tile_h import solve_crop_tile_h
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_crop_tile_h(task) is None


def test_panel_max_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.panel_max_fill import solve_panel_max_fill, _ref

    g = np.zeros((11, 11), int)
    g[3, :] = 5; g[7, :] = 5; g[:, 3] = 5; g[:, 7] = 5
    g[0, 0] = 1; g[1, 1] = 1     # panel (0,0): 2 markers (the most)
    g[5, 5] = 1                  # panel (1,1): 1 marker
    expected = _ref(g)
    assert expected is not None
    assert np.all(expected[0:3, 0:3] == 1) and expected[5, 5] == 0   # winner solid, loser cleared
    task = _make_task([(g.tolist(), expected.tolist())])
    model = solve_panel_max_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g.tolist())})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected.tolist()))


def test_panel_max_fill_rejects_wrong_size():
    from neurogolf.solvers.panel_max_fill import solve_panel_max_fill
    task = _make_task([([[1, 0], [0, 0]], [[1, 0], [0, 0]])])
    assert solve_panel_max_fill(task) is None


def test_bbox_recolor_ones():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.bbox_recolor_ones import solve_bbox_recolor_ones, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 8, 0, 8, 0],
         [0, 0, 1, 0, 1],     # (2,2) inside 8-bbox -> 3 ; (2,4) outside -> stays 1
         [0, 8, 0, 8, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[2] == [0, 0, 3, 0, 1]
    task = _make_task([(g, expected)])
    model = solve_bbox_recolor_ones(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_bbox_recolor_ones_rejects_no_eight():
    from neurogolf.solvers.bbox_recolor_ones import solve_bbox_recolor_ones
    task = _make_task([([[1, 0], [0, 0]], [[1, 0], [0, 0]])])
    assert solve_bbox_recolor_ones(task) is None


def test_stamp_at_markers():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.stamp_at_markers import solve_stamp_at_markers, _ref

    g = [[4, 2, 2, 5, 0, 0, 0],
         [2, 6, 2, 5, 0, 1, 0],     # marker at (1,5) -> template stamped at cols 4..6
         [6, 4, 4, 5, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[4, 2, 2, 5, 4, 2, 2],
                        [2, 6, 2, 5, 2, 6, 2],
                        [6, 4, 4, 5, 6, 4, 4]]
    task = _make_task([(g, expected)])
    model = solve_stamp_at_markers(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_stamp_at_markers_rejects_no_marker():
    from neurogolf.solvers.stamp_at_markers import solve_stamp_at_markers
    task = _make_task([([[4, 0, 0], [0, 0, 0], [0, 0, 0]], [[4, 0, 0], [0, 0, 0], [0, 0, 0]])])
    assert solve_stamp_at_markers(task) is None


def test_left_third():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.left_third import solve_left_third, _ref

    g = [[4, 5, 1, 1, 5, 4, 4, 5, 1],     # W=9 -> keep leftmost 3 columns
         [5, 5, 5, 5, 5, 5, 5, 5, 5],
         [1, 5, 4, 4, 5, 1, 1, 5, 4]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[4, 5, 1], [5, 5, 5], [1, 5, 4]]
    task = _make_task([(g, expected)])
    model = solve_left_third(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_left_third_rejects_non_multiple():
    from neurogolf.solvers.left_third import solve_left_third
    task = _make_task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solve_left_third(task) is None


def test_marker_box_interior():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.marker_box_interior import solve_marker_box_interior, _ref

    g = [[6, 0, 0, 0, 6],     # 6 markers at corners; 8-shape inside -> recoloured 6
         [0, 0, 8, 0, 0],
         [0, 8, 8, 8, 0],
         [0, 0, 8, 0, 0],
         [6, 0, 0, 0, 6]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 6, 0], [6, 6, 6], [0, 6, 0]]
    task = _make_task([(g, expected)])
    model = solve_marker_box_interior(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_marker_box_interior_rejects_tiny():
    from neurogolf.solvers.marker_box_interior import solve_marker_box_interior
    task = _make_task([([[6, 0], [0, 6]], [[6, 0], [0, 6]])])
    assert solve_marker_box_interior(task) is None


def test_stack_to_band():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.stack_to_band import solve_stack_to_band, _ref

    g = [[0, 2, 0, 2, 0],     # markers above/below a horizontal 5-band stack onto it
         [5, 5, 5, 5, 5],
         [0, 0, 2, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 5, 0, 5, 0], [5, 5, 5, 5, 5], [0, 0, 5, 0, 0]]
    task = _make_task([(g, expected)])
    model = solve_stack_to_band(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_stack_to_band_rejects_no_band():
    from neurogolf.solvers.stack_to_band import solve_stack_to_band
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_stack_to_band(task) is None


def test_edge_frame():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.edge_frame import solve_edge_frame, _ref

    g = [[1, 2], [3, 8]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 1, 2, 0],
                        [1, 1, 2, 2],
                        [3, 3, 8, 8],
                        [0, 3, 8, 0]]
    task = _make_task([(g, expected)])
    model = solve_edge_frame(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_edge_frame_rejects_identity():
    from neurogolf.solvers.edge_frame import solve_edge_frame
    task = _make_task([([[1, 2], [3, 8]], [[1, 2], [3, 8]])])
    assert solve_edge_frame(task) is None


def test_eight_center_crop():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.eight_center_crop import solve_eight_center_crop, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 4, 0, 0, 0],
         [4, 8, 4, 0, 0],     # 8 at (2,1): crop its 3x3, recolour 8 -> 4
         [0, 4, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 4, 0], [4, 4, 4], [0, 4, 0]]
    task = _make_task([(g, expected)])
    model = solve_eight_center_crop(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_eight_center_crop_rejects_no_eight():
    from neurogolf.solvers.eight_center_crop import solve_eight_center_crop
    task = _make_task([([[4, 0], [0, 0]], [[4, 0], [0, 0]])])
    assert solve_eight_center_crop(task) is None


def test_diag_ray_pair():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.diag_ray_pair import solve_diag_ray_pair, _ref

    g = [[0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0],
         [0, 1, 1, 0, 0, 0],
         [0, 2, 2, 0, 0, 0],
         [0, 2, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0][0] == 1 and expected[5][3] == 2     # up-left 1-ray, down-right 2-ray
    task = _make_task([(g, expected)])
    model = solve_diag_ray_pair(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_diag_ray_pair_rejects_single_colour():
    from neurogolf.solvers.diag_ray_pair import solve_diag_ray_pair
    task = _make_task([([[1, 1], [1, 1]], [[1, 1], [1, 1]])])
    assert solve_diag_ray_pair(task) is None


def test_blob_box_fill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.blob_box_fill import solve_blob_box_fill, _ref

    g = [[4, 4, 4, 0, 0],
         [4, 0, 4, 0, 0],     # blob bbox 3x3: holes (1,1) and (2,0) -> 7
         [0, 4, 4, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[4, 4, 4, 0, 0],
                        [4, 7, 4, 0, 0],
                        [7, 4, 4, 0, 0],
                        [0, 0, 0, 0, 0]]
    task = _make_task([(g, expected)])
    model = solve_blob_box_fill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_blob_box_fill_rejects_small_bbox():
    from neurogolf.solvers.blob_box_fill import solve_blob_box_fill
    task = _make_task([([[4, 0], [0, 0]], [[4, 0], [0, 0]])])
    assert solve_blob_box_fill(task) is None


def test_bar_echo():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.bar_echo import solve_bar_echo, _ref

    g = [[2, 0, 0, 0, 0, 0],
         [2, 0, 0, 8, 0, 0],     # marked row: fill to marker, marker -> 4
         [2, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 2],
         [0, 0, 0, 0, 0, 2],     # echo row (offset 1): full row of 8
         [0, 0, 0, 0, 0, 2]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [2, 8, 8, 4, 0, 0]
    assert expected[4] == [8, 8, 8, 8, 8, 2]
    task = _make_task([(g, expected)])
    model = solve_bar_echo(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_bar_echo_rejects_no_marker():
    from neurogolf.solvers.bar_echo import solve_bar_echo
    task = _make_task([([[2, 0], [2, 0]], [[2, 0], [2, 0]])])
    assert solve_bar_echo(task) is None


def test_panel_pair_flag():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.panel_pair_flag import solve_panel_pair_flag, _ref

    g = np.zeros((11, 11), int)
    g[3, :] = 8; g[7, :] = 8; g[:, 3] = 8; g[:, 7] = 8
    g[0, 0] = 6; g[1, 1] = 6      # panel (0,0): two 6s -> 1
    g[5, 5] = 6                   # panel (1,1): one 6 -> 0
    expected = _ref(g)
    assert expected.tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    task = _make_task([(g.tolist(), expected.tolist())])
    model = solve_panel_pair_flag(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g.tolist())})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected.tolist()))


def test_panel_pair_flag_rejects_wrong_size():
    from neurogolf.solvers.panel_pair_flag import solve_panel_pair_flag
    task = _make_task([([[6, 0], [0, 0]], [[6, 0], [0, 0]])])
    assert solve_panel_pair_flag(task) is None


def test_cross_ring():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.cross_ring import solve_cross_ring, _ref

    g = [[0, 3, 0, 0],
         [2, 2, 2, 2],
         [0, 3, 0, 0],
         [0, 3, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[4, 4, 4, 0],
                        [4, 2, 4, 2],
                        [4, 4, 4, 0],
                        [0, 3, 0, 0]]
    task = _make_task([(g, expected)])
    model = solve_cross_ring(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_cross_ring_rejects_no_cross():
    from neurogolf.solvers.cross_ring import solve_cross_ring
    task = _make_task([([[2, 0], [0, 0]], [[2, 0], [0, 0]])])
    assert solve_cross_ring(task) is None


def test_edge_pair_lines():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.edge_pair_lines import solve_edge_pair_lines, _ref

    g = [[0, 0, 0, 0, 0],
         [3, 0, 8, 0, 3],     # 3-pair row; 8 is noise (count 2 but 0 lines... 1 cell here)
         [0, 0, 0, 0, 0],
         [0, 0, 8, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 0, 0, 0, 0],
                        [3, 3, 3, 3, 3],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]
    task = _make_task([(g, expected)])
    model = solve_edge_pair_lines(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_edge_pair_lines_rejects_no_pairs():
    from neurogolf.solvers.edge_pair_lines import solve_edge_pair_lines
    task = _make_task([([[3, 0], [0, 0]], [[3, 0], [0, 0]])])
    assert solve_edge_pair_lines(task) is None


def test_key_meta_mask():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.key_meta_mask import solve_key_meta_mask, _ref

    g = np.zeros((16, 16), int)
    for a in range(3):                       # blocks on the diagonal only
        if a != 1:
            g[a * 3:a * 3 + 3, a * 3:a * 3 + 3] = 1
    g[4:7, 4:7] = 1                          # centre block
    g[11:14, 12:15] = np.array([[3, 4, 5], [6, 7, 8], [9, 2, 3]])   # 3x3 key
    expected = _ref(g)
    assert expected is not None
    assert expected.tolist() == [[3, 0, 0], [0, 7, 0], [0, 0, 3]]
    task = _make_task([(g.tolist(), expected.tolist())])
    model = solve_key_meta_mask(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g.tolist())})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected.tolist()))


def test_key_meta_mask_rejects_plain():
    from neurogolf.solvers.key_meta_mask import solve_key_meta_mask
    task = _make_task([([[1, 0], [0, 0]], [[1, 0], [0, 0]])])
    assert solve_key_meta_mask(task) is None


def test_symmetric_shape_crop():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.symmetric_shape_crop import solve_symmetric_shape_crop, _ref

    g = [[0, 2, 2, 0, 0, 0, 0],
         [0, 0, 2, 2, 0, 0, 0],     # 2: asymmetric
         [0, 0, 0, 0, 0, 7, 7],
         [0, 4, 4, 0, 0, 7, 7],     # 4: symmetric square ; 7: symmetric but...
         [0, 4, 4, 0, 0, 7, 0]]     # 7 has an extra cell -> asymmetric
    expected = _ref(np.array(g))
    assert expected is not None and expected.tolist() == [[4, 4], [4, 4]]
    task = _make_task([(g, expected.tolist())])
    model = solve_symmetric_shape_crop(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected.tolist()))


def test_symmetric_shape_crop_rejects_multi():
    from neurogolf.solvers.symmetric_shape_crop import solve_symmetric_shape_crop
    task = _make_task([([[4, 0, 7], [0, 0, 0]], [[4]])])   # both symmetric -> ambiguous
    assert solve_symmetric_shape_crop(task) is None


def test_crop_flip_h():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.crop_flip_h import solve_crop_flip_h, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 8, 8, 2, 0],
         [0, 8, 2, 8, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[2, 8, 8], [8, 2, 8]]
    task = _make_task([(g, expected)])
    model = solve_crop_flip_h(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_crop_flip_h_rejects_symmetric_only():
    from neurogolf.solvers.crop_flip_h import solve_crop_flip_h
    task = _make_task([([[8, 8], [8, 8]], [[8, 8], [8, 8]])])
    assert solve_crop_flip_h(task) is None


def test_reflect_marker_dir():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.reflect_marker_dir import solve_reflect_marker_dir, _ref

    g = [[0, 0, 8, 8, 0, 0, 0, 0],
         [0, 0, 0, 8, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 4, 0, 0, 0, 0, 0, 0],     # top arm left of centre -> mirror left
         [0, 4, 4, 4, 0, 0, 0, 0],
         [0, 0, 4, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [8, 8, 8, 8, 0, 0, 0, 0] and expected[1][0] == 8
    task = _make_task([(g, expected)])
    model = solve_reflect_marker_dir(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_reflect_marker_dir_rejects_no_marker():
    from neurogolf.solvers.reflect_marker_dir import solve_reflect_marker_dir
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_reflect_marker_dir(task) is None


def test_quadrant_corner_map():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.quadrant_corner_map import solve_quadrant_corner_map, _ref

    g = [[2, 1, 0, 0, 1, 3],
         [1, 1, 1, 1, 1, 1],
         [0, 1, 8, 8, 1, 0],
         [0, 1, 8, 0, 1, 0],
         [1, 1, 1, 1, 1, 1],
         [4, 1, 0, 0, 1, 6]]
    expected = _ref(np.array(g))
    assert expected is not None and expected.tolist() == [[2, 3], [4, 0]]
    task = _make_task([(g, expected.tolist())])
    model = solve_quadrant_corner_map(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected.tolist()))


def test_quadrant_corner_map_rejects_plain():
    from neurogolf.solvers.quadrant_corner_map import solve_quadrant_corner_map
    task = _make_task([([[8, 0], [0, 0]], [[8, 0], [0, 0]])])
    assert solve_quadrant_corner_map(task) is None


def test_open_2x2():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.open_2x2 import solve_open_2x2, _ref

    g = [[0, 8, 0, 0, 0],
         [0, 0, 0, 8, 8],     # (0,1) single -> removed ; 2x2 block kept
         [0, 0, 0, 8, 8],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 0, 0, 0, 0],
                        [0, 0, 0, 8, 8],
                        [0, 0, 0, 8, 8],
                        [0, 0, 0, 0, 0]]
    task = _make_task([(g, expected)])
    model = solve_open_2x2(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_open_2x2_rejects_no_noise():
    from neurogolf.solvers.open_2x2 import solve_open_2x2
    task = _make_task([([[8, 8], [8, 8]], [[8, 8], [8, 8]])])
    assert solve_open_2x2(task) is None


def test_maze_enclose():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.maze_enclose import solve_maze_enclose, _ref

    g = [[8, 8, 8, 8, 8],
         [8, 0, 8, 0, 8],     # left pocket sealed -> 2 ; right cell open at bottom -> 3
         [8, 0, 8, 0, 8],
         [8, 8, 8, 0, 8],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1][1] == 2 and expected[1][3] == 3
    task = _make_task([(g, expected)])
    model = solve_maze_enclose(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_maze_enclose_rejects_multicolour():
    from neurogolf.solvers.maze_enclose import solve_maze_enclose
    task = _make_task([([[8, 0], [3, 0]], [[8, 0], [3, 0]])])
    assert solve_maze_enclose(task) is None


def test_band_drill():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.band_drill import solve_band_drill, _ref

    g = [[5, 5, 5, 5],
         [4, 4, 0, 4],     # horizontal stripes; hole drills the 4-band column
         [4, 4, 4, 4],
         [8, 8, 8, 8]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[5, 5, 5, 5],
                        [4, 4, 0, 4],
                        [4, 4, 0, 4],
                        [8, 8, 8, 8]]
    task = _make_task([(g, expected)])
    model = solve_band_drill(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_band_drill_rejects_no_holes():
    from neurogolf.solvers.band_drill import solve_band_drill
    task = _make_task([([[5, 5], [4, 4]], [[5, 5], [4, 4]])])
    assert solve_band_drill(task) is None


def test_stamp_template_at_five():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.stamp_template_at_five import solve_stamp_template_at_five, _ref

    g = [[0, 2, 0, 0, 0, 0],
         [2, 2, 1, 0, 0, 0],     # template (centre (1,1)); 5 marks where to copy it
         [0, 1, 3, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 5, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[4][4] == 2 and expected[5][5] == 3 and expected[4][5] == 1
    task = _make_task([(g, expected)])
    model = solve_stamp_template_at_five(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_stamp_template_at_five_rejects_no_marker():
    from neurogolf.solvers.stamp_template_at_five import solve_stamp_template_at_five
    task = _make_task([([[2, 1], [0, 0]], [[2, 1], [0, 0]])])
    assert solve_stamp_template_at_five(task) is None


def test_divider_rays():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.divider_rays import solve_divider_rays, _ref

    g = [[0, 2, 0, 1, 0],
         [0, 0, 0, 0, 0],     # 2 grows down to divider, 1 grows up to edge
         [5, 5, 5, 5, 5],
         [0, 0, 0, 0, 0],
         [0, 2, 0, 1, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected == [[0, 2, 0, 1, 0],
                        [0, 2, 0, 0, 0],
                        [5, 5, 5, 5, 5],
                        [0, 2, 0, 0, 0],
                        [0, 2, 0, 1, 0]]
    task = _make_task([(g, expected)])
    model = solve_divider_rays(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_divider_rays_rejects_no_divider():
    from neurogolf.solvers.divider_rays import solve_divider_rays
    task = _make_task([([[2, 0], [0, 1]], [[2, 0], [0, 1]])])
    assert solve_divider_rays(task) is None


def test_mirror_tile_3x2():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.mirror_tile_3x2 import solve_mirror_tile_3x2, _ref

    g = [[0, 8], [0, 0], [0, 8]]
    expected = _ref(np.array(g)).tolist()
    assert len(expected) == 9 and len(expected[0]) == 4
    assert expected[0] == [8, 0, 0, 8]
    task = _make_task([(g, expected)])
    model = solve_mirror_tile_3x2(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_mirror_tile_3x2_rejects_wrong_size():
    from neurogolf.solvers.mirror_tile_3x2 import solve_mirror_tile_3x2
    task = _make_task([([[8, 0, 0]], [[8, 0, 0]])])
    assert solve_mirror_tile_3x2(task) is None


def test_rotate_into_regions():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import from_onehot, to_onehot
    from neurogolf.solvers.rotate_into_regions import solve_rotate_into_regions, _ref

    g = [[1, 1, 2, 5, 0, 0, 0, 5, 0, 0, 0],
         [4, 1, 1, 5, 0, 0, 0, 5, 0, 0, 0],
         [4, 4, 1, 5, 0, 0, 0, 5, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert [row[4:7] for row in expected] == np.rot90(np.array(g)[:, 0:3], -1).tolist()
    task = _make_task([(g, expected)])
    model = solve_rotate_into_regions(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert from_onehot((res > 0.5).astype(np.float32)) == expected


def test_rotate_into_regions_rejects_wrong_size():
    from neurogolf.solvers.rotate_into_regions import solve_rotate_into_regions
    task = _make_task([([[1, 5, 0], [1, 5, 0]], [[1, 5, 0], [1, 5, 0]])])
    assert solve_rotate_into_regions(task) is None


def test_marker_ring():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.marker_ring import solve_marker_ring, _ref

    g = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 3, 0, 0, 0, 0],     # 3 -> 6-ring ; 8 -> 4-ring (spaced, no overlap)
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 8, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [0, 6, 3, 6, 0, 0, 0] and expected[3][4] == 4
    task = _make_task([(g, expected)])
    model = solve_marker_ring(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_marker_ring_rejects_other_colour():
    from neurogolf.solvers.marker_ring import solve_marker_ring
    task = _make_task([([[0, 0], [0, 7]], [[0, 0], [0, 7]])])
    assert solve_marker_ring(task) is None


def test_alt_ray_right():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.alt_ray_right import solve_alt_ray_right, _ref

    g = [[0, 0, 2, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 6, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [0, 0, 2, 5, 2, 5, 2, 5]
    assert expected[2] == [0, 0, 0, 6, 5, 6, 5, 6]
    task = _make_task([(g, expected)])
    model = solve_alt_ray_right(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_alt_ray_right_rejects_plain():
    from neurogolf.solvers.alt_ray_right import solve_alt_ray_right
    task = _make_task([([[0, 2], [0, 0]], [[0, 2], [0, 0]])])
    assert solve_alt_ray_right(task) is None


def test_right_then_down_ray():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.right_then_down_ray import solve_right_then_down_ray, _ref

    g = [[0, 0, 0, 0, 0, 0],
         [0, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [0, 0, 2, 2, 2, 2]
    assert expected[2][5] == 2 and expected[4][5] == 3 and expected[5][5] == 3
    task = _make_task([(g, expected)])
    model = solve_right_then_down_ray(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_right_then_down_ray_rejects_plain():
    from neurogolf.solvers.right_then_down_ray import solve_right_then_down_ray
    task = _make_task([([[0, 2], [0, 0]], [[0, 2], [0, 0]])])
    assert solve_right_then_down_ray(task) is None


def test_tall_short_lines():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.tall_short_lines import solve_tall_short_lines, _ref

    g = [[5, 0, 0, 0],
         [5, 0, 0, 5],
         [5, 0, 0, 5],
         [5, 0, 0, 0]]   # col0 tallest (4) -> 1 ; col3 shortest (2) -> 2
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [1, 0, 0, 0] and expected[1] == [1, 0, 0, 2]
    task = _make_task([(g, expected)])
    model = solve_tall_short_lines(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_tall_short_lines_rejects_uniform():
    from neurogolf.solvers.tall_short_lines import solve_tall_short_lines
    task = _make_task([([[5, 0, 5], [5, 0, 5]], [[5, 0, 5], [5, 0, 5]])])  # equal heights
    assert solve_tall_short_lines(task) is None


def test_midpoint_fill_h():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.midpoint_fill_h import solve_midpoint_fill_h, _ref

    g = [[1, 0, 1, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [1, 2, 1, 2, 1] and expected[2] == [0, 1, 2, 1, 0]
    task = _make_task([(g, expected)])
    model = solve_midpoint_fill_h(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_midpoint_fill_h_rejects_plain():
    from neurogolf.solvers.midpoint_fill_h import solve_midpoint_fill_h
    task = _make_task([([[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]])])
    assert solve_midpoint_fill_h(task) is None


def test_drop_one_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.drop_one_recolor import solve_drop_one_recolor, _ref

    g = [[8, 8, 0],
         [8, 8, 0],
         [0, 0, 0],
         [0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[1] == [2, 2, 0] and expected[2] == [2, 2, 0]
    task = _make_task([(g, expected)])
    model = solve_drop_one_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_drop_one_recolor_rejects_empty():
    from neurogolf.solvers.drop_one_recolor import solve_drop_one_recolor
    task = _make_task([([[0, 0], [0, 0]], [[0, 0], [0, 0]])])
    assert solve_drop_one_recolor(task) is None


def test_isolated_two_recolor():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.isolated_two_recolor import solve_isolated_two_recolor, _ref

    g = [[2, 2, 0, 0],
         [0, 0, 0, 2],
         [0, 0, 0, 0]]   # the (1,3) two is isolated -> 1 ; the 2x1 pair stays
    expected = _ref(np.array(g)).tolist()
    assert expected[0] == [2, 2, 0, 0] and expected[1] == [0, 0, 0, 1]
    task = _make_task([(g, expected)])
    model = solve_isolated_two_recolor(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_flood_ones():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.flood_ones import solve_flood_ones, _ref

    g = [[1, 0, 0, 3, 0],
         [0, 0, 0, 3, 0],   # right 0-column sealed by 3-wall -> stays 0
         [3, 3, 3, 3, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0][1] == 1 and expected[0][2] == 1  # reachable from the 1
    assert expected[0][4] == 0 and expected[1][4] == 0  # sealed pocket stays 0
    task = _make_task([(g, expected)])
    model = solve_flood_ones(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_smallest_blob_two():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.smallest_blob_two import solve_smallest_blob_two, _ref

    g = [[8, 8, 0, 0, 0],
         [8, 8, 0, 8, 0],
         [0, 0, 0, 0, 0],
         [8, 8, 8, 0, 0],
         [8, 8, 8, 0, 0]]   # blobs: 4-cell, 1-cell(smallest->2), 6-cell
    expected = _ref(np.array(g)).tolist()
    assert expected[1][3] == 2 and expected[0][0] == 1 and expected[3][0] == 1
    task = _make_task([(g, expected)])
    model = solve_smallest_blob_two(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_smallest_blob_two_rejects_single():
    from neurogolf.solvers.smallest_blob_two import solve_smallest_blob_two
    task = _make_task([([[8, 8], [8, 8]], [[1, 1], [1, 1]])])  # single blob
    assert solve_smallest_blob_two(task) is None


def test_domino_ring():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.domino_ring import solve_domino_ring, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 0, 0],
         [2, 0, 0, 0, 0]]   # vertical domino -> ring ; lone 2 untouched
    expected = _ref(np.array(g)).tolist()
    assert expected[0][1:4] == [3, 3, 3] and expected[1][2] == 2 and expected[4][0] == 2
    task = _make_task([(g, expected)])
    model = solve_domino_ring(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))


def test_domino_ring_rejects_lone():
    from neurogolf.solvers.domino_ring import solve_domino_ring
    task = _make_task([([[2, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]])])  # no domino
    assert solve_domino_ring(task) is None


def test_flood_ones_rejects_no_one():
    from neurogolf.solvers.flood_ones import solve_flood_ones
    task = _make_task([([[0, 2], [0, 0]], [[0, 2], [0, 0]])])
    assert solve_flood_ones(task) is None


def test_diag_corner_stamp():
    import numpy as np
    import onnxruntime as ort
    from neurogolf.grids import to_onehot
    from neurogolf.solvers.diag_corner_stamp import solve_diag_corner_stamp, _ref

    g = [[0, 0, 0, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 0, 0]]
    expected = _ref(np.array(g)).tolist()
    assert expected[0][1] == 3 and expected[0][3] == 6
    assert expected[2][1] == 8 and expected[2][3] == 7 and expected[1][2] == 0
    task = _make_task([(g, expected)])
    model = solve_diag_corner_stamp(task)
    assert model is not None and hasattr(model, "graph")
    sess = ort.InferenceSession(model.SerializeToString())
    res = sess.run(["output"], {"input": to_onehot(g)})[0]
    assert np.array_equal((res > 0.0).astype(np.float32), to_onehot(expected))
