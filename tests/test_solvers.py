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
