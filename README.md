<div align="center">

# NeuroGolf 2026

**Tiny ONNX networks that solve ARC-AGI puzzles · 用极小的 ONNX 网络求解 ARC-AGI 谜题**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.21-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.26-FF6F00)](https://onnxruntime.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-NeuroGolf%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/neurogolf-2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/zikuanqi/NeuroGolf)](https://github.com/zikuanqi/NeuroGolf/commits/main)
[![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)](tests/)
[![Tasks Solved](https://img.shields.io/badge/tasks_solved-119%2F400-blue)](networks/)
[![Local Score](https://img.shields.io/badge/local_score-1673.95-success)](networks/build_summary.json)
[![Public Score](https://img.shields.io/badge/public_score-1673.95-blue)](https://www.kaggle.com/competitions/neurogolf-2026)

</div>

---

## Table of Contents · 目录

1. [Overview · 项目概览](#overview)
2. [Current standing · 当前成绩](#standing)
3. [How it works · 工作原理](#how)
4. [Quick start · 快速开始](#quickstart)
5. [Solvers by family · 求解器分类](#solvers)
6. [Scoring & constraints · 评分与约束](#scoring)
7. [Repository layout · 项目结构](#layout)
8. [Adding a solver · 新增求解器](#adding)
9. [Tests · 单元测试](#tests)
10. [Submission history · 提交历史](#history)
11. [Roadmap · 后续计划](#roadmap)
12. [License · 许可证](#license)

---

<a id="overview"></a>

## 📖 Overview · 项目概览

> **EN** — Solution framework for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) on Kaggle. The competition asks for a small **ONNX network per ARC-AGI task**. A task that reproduces **every** train / test / arc-gen example earns `max(1, 25 − ln(memory_bytes + params))` points; any single wrong example scores zero. The goal is therefore to solve as many tasks as possible **and** keep each network tiny.
>
> Rather than train models, this repo is a library of **hand-built, pattern-specific solvers**. Each one recognises a family of ARC transformations (recolour, crop, gravity, flood-fill, connected-component relabelling, …) and emits an exact ONNX graph for it.

> **中文** —— Kaggle [2026 NeuroGolf 锦标赛](https://www.kaggle.com/competitions/neurogolf-2026) 的求解框架。比赛要求为每个 ARC-AGI 任务提交一个 **ONNX 网络**：在通过**全部** train / test / arc-gen 样例的前提下，得分为 `max(1, 25 − ln(内存字节 + 参数))`，任一样例错误即零分。目标是尽量多解题，并让每个网络尽量小。
>
> 本仓库不训练模型，而是一组**手工编写、按模式分类的求解器**：每个识别一类 ARC 变换并直接生成精确的 ONNX 图。

---

<a id="standing"></a>

## 📊 Current standing · 当前成绩

| Metric · 指标 | Value · 数值 |
|---|---|
| **Tasks solved · 通过任务** | **119 / 400** |
| **Local score · 本地总分** | **1673.95** — clean-room scorer over `build_summary.json` · 独立评分器统计 |
| **Public score · 公开分数 (Kaggle)** | **1673.95** — leaderboard-confirmed (119/400), matches the local score to the penny · 排行榜确认，与本地分数完全一致 |
| Solvers · 求解器 | 98, in 9 families · 共 98 个，分 9 类 |
| Unit tests · 单元测试 | 132 passing · 132 个全部通过 |
| Networks · 网络文件 | 119 × `networks/taskNNN.onnx` (one per solved task) · 每个解出任务一个 |

> Local development and the Kaggle leaderboard agree at **1673.95 / 119 tasks**. The local clean-room scorer has matched the official score to two decimals on every confirmed submission. The v1 → v46 progression lives in [Submission history](#history).
> 本地开发与 Kaggle 排行榜一致，均为 **1673.95 / 119 解**。本地独立评分器在每次已确认提交中均与官方分数小数点后两位一致。v1 → v46 进展见 [提交历史](#history)。

---

<a id="how"></a>

## ⚙️ How it works · 工作原理

```
ARC task (JSON)
      │
      │  grids.py: to_onehot()           every grid → float tensor (1, 10, 30, 30)
      ▼                                  channel = colour 0‑9; real grid top‑left, rest 0‑padded
 ┌────────────────────────────────────────────────────────────────────────┐
 │  pipeline.build_one(task)                                              │
 │    for solver in ALL_SOLVERS:          ~98 pattern‑specific solvers    │
 │        model = solver(task)            None if the pattern doesn't fit │
 │        score = verify(model, task)     clean‑room official scorer      │
 │    keep the highest‑scoring model that passes EVERY example            │
 └────────────────────────────────────────────────────────────────────────┘
      │
      ▼
 networks/taskNNN.onnx   +   networks/build_summary.json
```

**1. Representation.** `grids.py` maps each grid to a one-hot `(1, 10, 30, 30)` `float32` tensor: channel `k` is 1 where the cell has colour `k`. The real grid sits in the top-left; cells outside it are all-zero padding. `from_onehot` thresholds at `> 0` and trims padding to reverse the mapping.

**2. Solver contract.** Every solver is a callable `solve_xxx(task: dict) -> Optional[onnx.ModelProto]`. It returns `None` if the task doesn't match its pattern, otherwise a hand-built ONNX graph with a single `"input"` → `"output"` tensor of shape `(1, 10, 30, 30)`. Most solvers first **detect** parameters from the examples (wall colours, scale factor, a size→colour map, a 3×3 stamp, …) and **bake them in** to the graph.

**3. Verification.** `verify.py` is a clean-room re-implementation of the official scorer. It loads the candidate, runs it through ONNX Runtime on every train / test / arc-gen example, thresholds the output at `> 0`, requires an **exact** match against the expected one-hot, and computes `(memory_bytes, params)` to derive the per-task points.

**4. Selection.** `pipeline.build_one` runs **all** solvers on a task and keeps the highest-scoring one that passes; `scripts/build_all.py` does this across all 400 tasks and writes the `.onnx` files plus `build_summary.json`. Because selection is by score, a cheaper solver automatically wins over a more expensive one on the same task.

**中文摘要** —— 网格先转成 `(1,10,30,30)` 独热张量；`build_one` 对每个任务跑遍全部求解器，用独立实现的评分器 `verify.py` 验证（输出按 `>0` 阈值化并要求逐样例精确匹配），保留通过且得分最高的网络；`build_all.py` 对 400 个任务批量执行。多数求解器会先从样例中**检测参数**再烘焙进图。

---

<a id="quickstart"></a>

## 🚀 Quick start · 快速开始

```bash
# 1. Set up the environment · 配置环境
#    A) conda — reproducible, recommended · conda（可复现，推荐）
conda env create -f environment.yml    # builds the `neurogolf` env (Python 3.13)
conda activate neurogolf               # project installed editable → `import neurogolf`
#    B) pip / venv · pip / venv
pip install -e .                        # editable install (or: pip install -r requirements.txt)

# 2. Download competition data · 下载竞赛数据
python scripts/download.py

# 3. Build networks for all 400 tasks · 为全部 400 个任务构建网络
python scripts/build_all.py            # writes networks/*.onnx + build_summary.json
#   python scripts/build_all.py --from 40 --to 40   # rebuild a single task

# 4. Package the .onnx files · 打包 .onnx 文件
python scripts/package_submission.py   # → submissions/submission.zip

# 5. Submit to Kaggle · 提交到 Kaggle
python scripts/submit.py submissions/submission.zip "describe your run"
```

> Kaggle requires the submission file be named **exactly** `submission.zip`.
> Kaggle 要求提交文件必须命名为 `submission.zip`。

---

<a id="solvers"></a>

## 🧩 Solvers by family · 求解器分类

98 solvers in 9 families. Each is verified against the official scorer; `Params` is the parameter count of a representative network (it can vary slightly per task because detected constants differ). A few solvers are registered as capabilities but currently win no task (dominated or non-matching on the present set); these are marked `—†`.

98 个求解器分为 9 类。`Params` 列为代表性网络的参数量（不同任务因检测出的常量不同会略有差异）。少数求解器已登记但当前未中标（被更省的求解器击败或不匹配现有任务），以 `—†` 标记。

### 1 · Rigid moves — identity, flip, rotate, transpose, shift · 刚性变换

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_identity` | output ≡ input · 输出等于输入 | `Identity` | 0 |
| `solve_zero` | output is an empty grid · 输出为空网格 | `Sub` | 0 |
| `solve_transpose` | output is the input transposed · 输出为输入转置 | `Transpose` | 0 |
| `solve_flip_h_aware` / `solve_flip_v_aware` | content-aware horizontal / vertical flip · 形状感知水平 / 垂直翻转 | `ReduceSum`+`ReduceMax`+`Mul`(arange)+`Sub`+`Clip`+`Cast`+`Gather`+`Mul` | ~37 |
| `solve_rot180_aware` | two stacked shape-aware flips · 双轴形状感知翻转 | flip-h ∘ flip-v | ~74 |
| `solve_rot90_ccw_aware` / `solve_rot90_cw_aware` | shape-aware 90° rotation · 形状感知 90° 旋转 | `Transpose` ∘ flip-v | ~37 |
| `solve_shift` | constant-shape translation with colour-0 fill · 定形平移并以色 0 填充 | `Slice`+`Concat`(init fill)+`Pad` | ~50 |

### 2 · Crop & extract — output is a sub-region · 裁剪与提取

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_static_crop` | fixed-offset, fixed-size subrect · 定长定位裁剪 | `Slice`+`Pad` | ~14 |
| `solve_marker_crop` | crop a fixed window around a unique marker pixel · 围绕唯一标记的定长裁剪 | `Slice`→`ReduceSum`→`ArgMax`→`Slice`→`Pad` | ~27 |
| `solve_bbox_strip` | crop to the bbox of non-background cells · 非背景外接矩形提取 | `ReduceSum`+`Sub`+row/col `ArgMax`+`Gather`+`Less`+`Mul` | ~76 |
| `solve_largest_comp_crop` | crop to the largest 4-connected component (a solid blob amid single-cell noise), holes kept as bg (task 36) · 裁剪到最大连通块 | iterated label max-propagation + all-pairs size + largest mask + bg-fill + bbox `Gather` crop | ~1k |
| `solve_bbox_color_extract` | crop to the bbox of the majority / rarest colour · 按主色/稀有色外接矩形裁剪 | `ReduceSum`+`ArgMax`/`ArgMin`+`OneHot`+`Mul`+`Cast`+`Gather`+`Mod`+`Less` | ~66 |

### 3 · Scale, tile & symmetry — repeat or mirror the grid · 缩放、平铺与镜像

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_kron_scale` | N×N pixel expansion, constant shape · 同尺寸像素块复制 | 2× `Gather`+`Pad` | ~40 |
| `solve_resize_scale` | N× nearest upscale, variable shape · 变尺寸 N 倍最近邻放大 | `Slice`+`Resize` | ~24 |
| `solve_downscale_majority` | block-majority **down**scale: each k×k block → its majority colour (task 130, 9×9→3×3) · 块多数色降采样 | `AveragePool`(kernel=stride=block)+`ArgMax`+`OneHot`+`Pad` | ~26 |
| `solve_scale_detector` | N× nearest upscale **or** downscale · N 倍最近邻放大或缩小 | `Slice`+`Resize` | ~24 |
| `solve_variable_kron` | scale by N = `count(non-zero)` / `count(distinct)` · 变 N 倍（N 来自输入特征） | `ReduceSum`+`Cast`+`Div`+`Min`+`Gather`×2+`Less`+`Mul` | ~138 |
| `solve_tile_h` | horizontal tile-N, variable width · 变宽 N 倍水平复刻 | `ReduceSum`+`ReduceMax`+`Mod`+`Gather`+`Less`+`Mul` | ~67 |
| `solve_period_extend_h` | continue the column period (p=1/2/3) to 2× width: `out[r][c]=in[r][c%p]` (task 231) · 按列周期延展到 2 倍宽 | per-period fixed `Gather` candidates + smallest-valid-period select (`Sub`/`Mul`/`ReduceSum`/`Less`) + 2W `Less` mask | ~123 |
| `solve_untile_half` | inverse of a 2× tile — input is one panel repeated (L/R or T/B), output is the single panel (task 188) · 还原 2× 平铺，输出单块 | content extent + index-shifted half compare + `Less` half-mask select (H-axis first) | ~65 |
| `solve_palindrome_h` / `solve_palindrome_v` | mirror-concat to the right / bottom edge · 镜像拼接 | shape-aware `Where`+`Gather`+`Less` mask | ~68 |
| `solve_palindrome_2d` | four-quadrant 2D mirror · 四象限二维镜像 | palindrome-h ∘ palindrome-v | ~133 |
| `solve_axis_gather` | constant-shape row / column gather — any fixed line shuffle (e.g. mirror-stack `[flip_v(in); in]`); subsumes constant-shape palindromes · 定形行/列重排（含镜像堆叠），并接管定形回文 | single `Gather` (baked index) | 30 |
| `solve_rot_tile` | N×N → 2N×2N as four rotations (I/90/180/270), constant N · 四象限旋转拼接（定长 N） | `Slice`+`Transpose`+`Gather`×4+`Concat`×3+`Pad` | ~23 |
| `solve_rot_tile_aware` | same rotational tiling but **shape-aware** — N varies across examples (task 106) · 同上但形状感知，N 随样例变化 | shape-aware `_flip_along` rotations + `_shift_along` (content-extent `Gather`) quadrant placement + `Add` | ~262 |
| `solve_self_fractal` | N×N → N²×N² self-similar fractal keyed by a selector colour · 自相似分形放大 | `Slice`+`Tile`+`Gather`×2 (Kron mask)+`ArgMax`+`Mul`+`Pad`+`Concat` | ~57 |
| `solve_block_mask` | N×N → N²×N² masked tiling + channel-0 recovery · 块掩码平铺 | `Slice`+`ReduceSum`+`Less`+`Tile`+`Mul`+`Add` | ~78 |
| `solve_repeat_top_rows` | runtime period detection (P=2/3/4) + weighted candidate mix · 顶部行周期复刻 | period detection + `Gather` + weighted blend | ~29 |
| `solve_periodic_fill` | restore a periodic tiling from a partially-erased grid · 从残缺网格恢复周期平铺 | per-period `Pad`+`Slice`+`Max` (log-doubling) + `Where` select + bbox clip | ~9k |
| `solve_diag_tile` | tile by anti-diagonal residue `(i+j)%n`, one colour per residue class (n=3) · 按反对角线 (i+j)%n 循环着色 | `Slice`+per-residue `Mul`+`ReduceSum`+`Greater`+`Cast`+`Add`+bbox `Min` | ~3.6k |

### 4 · Recolour — per-cell / per-object / whole-grid colour rules · 重新着色

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_remap` | per-pixel colour lookup · 像素级颜色映射 | 1×1 `Conv` | 100 |
| `solve_single_color` | uniform fill, same shape · 同尺寸纯色填充 | 1×1 `Conv` | 100 |
| `solve_framed_regions` | recolour a fixed two-region frame template by its two marker colours (task 28) · 用两标记色重涂固定框模板 | `Slice` marker rows + `ReduceMax` colour + two baked masks + background fill | ~907 |
| `solve_plus_panels` | fill the central plus of an 8-line panel grid with fixed colours, corners stay blank (task 55) · 八线面板网格的中央十字按固定色填充 | content-aware divider detection + `CumSum` band index + outer-product regions | ~32 |
| `solve_zero_color` | erase all cells of one fixed colour (→ 0) · 抹除某固定颜色 | channel-mask `Mul` | 10 |
| `solve_majority_fill` | constant-shape rect filled with the majority non-bg colour · 常尺寸主色填充 | `ReduceSum`+`TopK`+`Greater`+`And`+`Where`+`OneHot`+`Mul`+`Pad` | ~32 |
| `solve_largest_bbox_fill` | constant fill with the colour whose input bbox is largest · 以最大外接矩形的颜色纯色填充 | `ReduceMax`+`ArgMax`(bbox)+area+`ArgMax`(chan)+`OneHot`+`Pad` | ~94 |
| `solve_row_uniform_indicator` | per row: colour X if the input row is uniform, else Y · 行均匀则填 X 否则 Y | `ReduceSum`+`Greater`+`ReduceMax`+`Less` in-grid mask+`OneHot`+`Mul` | ~79 |
| `solve_column_label` | label colour-5 columns 1,2,3… by top-most marker row · 按最上标记行给列编号 | `Slice`+`ReduceSum`+`ArgMax`+pairwise `Less`/`Equal`/`And`/`Or` ranking+`Mul`+`Concat` | ~80 |
| `solve_keep_majority` | keep the most-frequent colour; recolour all other markers to 5 · 保留最多色，其余→5 | `ReduceSum`(count)+`ReduceMax`+`Equal`(argmax)+`Mul`/`Add` | ~31 |
| `solve_blob_recolor` | two non-bg colours: repaint the majority "blob" with the rarer "key" colour, clear the rest · 用稀有色重涂多数团块 | `ReduceSum` counts+`Equal`+`Greater`+`Sub`/`Mul`/`Add` | ~21 |
| `solve_recolor_fives` | repaint every colour-5 cell with its row's marker colour · 颜色-5 取所在行标记色 | `Gather`(ch5)+`ReduceMax`(row marker)+`Sub`/`Mul`/`Add` | ~21 |
| `solve_filled_rect` | replace a monochromatic filled rectangle with a new colour (or crop it) · 纯色矩形替换/裁剪 | constant output via `Sub`+`Add` | ~9k |

### 5 · Lines, gaps & gravity — draw, extend, fill, slide · 连线、填隙与重力

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_connect_dots` | fill the span between the first and last same-colour dot per row/column · 连接每行/列首尾同色点 | `Slice`+`CumSum`+`Gather`(reverse)+`Greater`+`And`+`Concat` | ~37 |
| `solve_connect_fill` | fill the gap between aligned same-colour dots with one fixed colour · 用固定色连接同色点 | `CumSum`×4 (prefix/suffix)+`Greater`+`And`+`Or`+`ReduceMax`+`Concat` | ~64 |
| `solve_diag_connect` | connect each same-colour pair with a diagonal line (task 37) · 用对角线连接同色点对 | 9-channel 4-way diagonal log-doubling cumulative-`Max` + main/anti `Mul` intersect + `Concat` | ~85 |
| `solve_hspan_fill` | fill each bg cell flanked left & right by a wall colour with a fixed colour · 填充被同色墙左右夹住的格 | `Gather`+exclusive/reverse `CumSum`+`Mul`/`Sub`/`Add` | ~24 |
| `solve_endpoint_bridge` | bridge each row's two dots: nearer-dot colour each side, colour 5 at the midpoint · 连接两点，中点为 5 | `ReduceMax` column ramps+`Greater`/`Less` side masks+`Mul`/`Add` | ~117 |
| `solve_color_lines` | colour-2 markers fill their column; other colours fill their row (drawn on top) · 颜色2竖线，其余横线覆盖 | `ReduceMax`(col-has-2 / row colour)+`Sub`/`Mul`/`Add` priority blend | ~32 |
| `solve_stripe_seeds` | two edge seeds → periodic alternating stripes (vertical if on top/bottom rows, horizontal if on left/right cols), period = seed gap (task 13) · 两个边缘种子生成周期交替条纹 | colour-value projection + seed-extent `ReduceMin`/`Max` + dynamic `Mod` stripe + edge-flag orientation select + `OneHot` | ~82 |
| `solve_cross_laser` | each marker fires a full row+column "plus"; differing-colour crossings → 2 · 十字射线，异色交叉为 2 | `ReduceMax`(row/col colour)+`ReduceSum`(same-colour test)+`Sub`/`Mul`/`Add` | ~31 |
| `solve_ray_down` | carry each marker's colour straight down its column · 颜色沿列向下填充 | per-colour cumulative-max down (log-doubling `Pad`+`Slice`+`Max`)+`Equal`/`Greater` | ~95 |
| `solve_diag_ray` | each non-zero cell emits a down-right diagonal ray across a 2N×2N output (input top-left) · 每个标记沿右下对角线发射射线（2N×2N） | channel-1‑9 `Slice`+diagonal log-doubling `Pad`+`Slice`+`Max`+`ReduceSum`+`Greater`+`Sub`(bg)+`Concat`+`Pad` | ~74 |
| `solve_diag_block_slide` | a 2×2 block (colour C + colour-2 corner markers) slides diagonally per marked corner, leaving a 2-wide trail of C (task 34) · 2×2 块按角标记沿对角线滑移留痕 | bbox corner flags + per-diagonal log-doubling shift-`Max` gated by flag + grid mask + `OneHot`-style colour fill | ~70 |
| `solve_drop_into_wall` | colour-1 cells fall into a full colour-5 wall row in their column · 颜色1落入颜色5的墙 | `Gather`+`ReduceSum`/`ReduceMax`+`Mul`(outer)+`Relu`+`Concat` | ~12 |
| `solve_nearest_wall` | recolour each marker with the colour of the nearer of two facing walls · 重涂为较近一堵墙的颜色 | `ReduceSum`/`ReduceMax`+`Gather`(far wall)+`Less`/`Greater` half-masks+`Mul`/`Add` | ~75 |
| `solve_flood_fill_enclosure` | fill colour-0 cells fully enclosed by a single source colour · 填充被单色完全包围的色0 | `ArgMax`+`Equal`+58×(`Pad`+`Conv`+`Greater`+`Max`) border BFS+`Gather`+`Sub`+`Slice`+`Concat` | ~1.8k |
| `solve_gravity_right` | each row's cells slide right until blocked · 重力向右 | `ReduceSum`+`CumSum`+`Where`+`Mul` | ~94 |
| `solve_gravity_down` | each column's cells fall to the bottom edge · 重力向下 | `ReduceSum`+`ReduceMax`(height)+`Less`+`Mul`+`Slice`+`Concat` | ~66 |
| `solve_gravity_up` | each column's cells rise to the top, preserving order · 重力向上保序 | `ReduceSum`+sort-key+`TopK`+`Tile`+`GatherElements` | ~42 |
| `solve_slide_to_wall` | a colour-2 object slides as a rigid block toward a colour-8 wall until adjacent (task 8) · 物体滑向墙壁直至相邻 | bbox `ReduceMin`/`Max` of both objects + signed gap shift + index-shifted `Gather` (clip + validity mask) + `Concat` | ~88 |
| `solve_slide_to_line` | each marker slides to the full row/column of its own colour, landing adjacent; no-line markers removed (task 25) · 标记滑向同色整行/列直至相邻 | per-channel line-detect (`ReduceSum==W`) + `CumSum` above/below split + `ReduceMax` project + ±1-shifted outer product; transpose for the vertical case | ~66 |
| `solve_project_to_block` | border markers each fire a ray into a colour-8 block, recolouring the first block cell hit (task 35) · 边缘标记射入色块，染中其首格 | block bbox + per-edge `ReduceMax` marker projection placed on the edge + `Where` merge + `OneHot` | ~95 |
| `solve_gravity_right_diag` | per-channel diagonal slide toward the centre of mass · 按质心方向对角滑移 | `ReduceSum`+`ReduceMax`+`ArgMax`+`Mul`+`Slice`+`Concat`+`Min`+`Sub` | 150 |

### 6 · Morphology & shape — dilate, erode, outline, stamp · 形态学与形状

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_dilate_ones` | expand every marker into a filled 3×3 block of colour 1 · 膨胀为颜色1的实心3×3 | 3×3 ones `Conv`+`Greater`+`Cast`+`Mul`/`Sub`/`Add` | ~31 |
| `solve_halo` | keep each marker, ring its background neighbours with colour 1 · 在标记四周围一圈颜色1 | 3×3 ones `Conv`(dilation)+`Greater`+`Cast`+`Sub`/`Mul`/`Add` | ~32 |
| `solve_outline` | keep each shape's perimeter, erase its interior · 保留边缘，擦除内部 | 3×3 cross `Conv`(4-neighbour count)+`Greater`+`Mul`/`Sub`/`Add` | ~21 |
| `solve_denoise` | remove isolated single cells (no same-colour 8-neighbour) · 删除孤立点 | depthwise 3×3 `Conv`(hollow)+`Greater`+`Mul`+`Concat` | ~81 |
| `solve_color_bbox_fill` | replace each colour's scattered cells with its solid bounding box · 用实心外接矩形填充散点 | `ReduceMax`(span)+`CumSum`×4+`Greater`+`And`+`Mul`(outer)+`Concat` | ~80 |
| `solve_mirror_complete` | restore an erased half as the shape-aware mirror of the present half · 镜像补全被擦除的另一半 | shape-aware flip (`ReduceMax`+`Sub`+`Clip`+`Gather`)+`Mul`+`Add` | ~42 |
| `solve_rot180_repair` | restore a rot180-symmetric image hidden under one occluder colour (task 287) · 用180°对称还原被遮挡区域 | reversed-index `Gather`×2 (baked rot180) + `Pad` + channel-mask select | ~37 |
| `solve_stamp` | replace each marker with a fixed 3×3 colour motif detected from the task · 用检测出的3×3图案替换标记 | per-colour 3×3 `Conv` (180-flipped stamp kernel)+`Greater`+`Mul`/`Sub`/`Add` | ~51 |
| `solve_stamp_top_row` | stamp the top-row pattern (in colour 2) at rows flagged by a right-edge marker (task 43) · 在右缘标记行复刻顶行图案 | row-0 pattern `Slice` × marked-rows `ReduceMax` outer product, masked to channel 0 | ~16 |

### 7 · Connected components — label, count, rank · 连通分量

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_isolate_recolor` | recolour each cell by whether it is isolated (no same-colour 4-neighbour) or connected, via a per-task map · 按孤立/相连重新着色 | depthwise 3×3 cross `Conv`+`Less`/`Greater`+two 1×1 `Conv` remaps | ~291 |
| `solve_cc_size_recolor` | recolour every same-colour component by its **cell count**, via a per-task size→colour map · 按分量大小着色 | iterated same-colour max-propagation labelling (26×) + all-pairs label match (counts) + size→colour map | ~1k |
| `solve_cc_rank_recolor` | recolour every component by its **size-rank** (largest = 0), via a per-task rank→colour map · 按分量大小名次着色 | cc-size labelling + second all-pairs (size-compare × reps) for ranks + rank→colour map | ~1k |

### 8 · Counting, logic & learned filters · 计数、逻辑与学习滤波

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_count_bar` | output a 1×N bar of the marker colour, N = number of marker cells · 输出长度为计数的条形 | `ReduceSum`(count + colour)+`Less`(column ramp < N)+`Mul` | ~72 |
| `solve_block_count_bar` | count 2×2 solid blocks of a colour → fixed-width 1×K unary bar (task 38) · 数 2×2 实心块 → 定宽计数条 | 2×2 ones `Conv`+`Greater`+`ReduceSum`(N)+`Less`(ramp<N) bar+`Concat` (ch0 background) | ~947 |
| `solve_lattice_count` | count a lattice's R row-bands × C column-bands → solid R×C block in the majority colour (task 21) · 数格子带数→定色实心块 | per-row/col mono-line detect (`ReduceMax`==content) + `ReduceSum` counts + `Less`(idx<R/C) outer product + arg-max fill | ~62 |
| `solve_split_and` | split along a colour-5 separator and AND the two halves · 沿色5分隔做 AND | `Slice`+`Pad`+`Sub`+`Mul`+`Less`+`Cast`+`And` | ~8.1k |
| `solve_split_logic` | split into two halves (L-R or T-B, ± divider) and combine with and/or/xor/nor/nand · 两半布尔运算 | `Slice`+`ReduceSum`+`Mul`/`Max`/`Sub`+`Pad`+`Concat` | ~20 |
| `solve_odd_panel` | four panels (2×2 layout ± blank divider), three identical → output the unique fourth (task 207) · 四面板中输出唯一不同的一个 | per-panel `Slice`+pairwise `Sub`/`Abs`/`ReduceSum`/`Less` equality+`Less`(agree=0) select+`Mul`/`Add`+`Pad` | ~45 |
| `solve_odd_panel_aware` | same odd-one-out but **shape-aware** — grid size varies (5×5/7×7/11×11, task 65) · 同上但形状感知，网格尺寸可变 | runtime `n` from content extent + index-shifted `Gather` panel re-alignment + masked pairwise-equality select | ~66 |
| `solve_conv3x3` | least-squares fit of a 3×3 conv (no bias) · 无偏置3×3卷积拟合 | 3×3 `Conv` | 900 |
| `solve_conv1x1_masked` / `solve_conv3x3_masked` / `solve_conv5x5_masked` | K×K conv + bias, masked to non-padding cells · 带偏置的 K×K 卷积+掩码 | `Conv`+`ReduceSum`+`Mul` | 100 / 910 / 2510 |
| `solve_learned_conv` | learn a K×K conv kernel (no bias) by `lstsq` over im2col patches; a learned fallback · im2col+最小二乘拟合 K×K 卷积核 | `Conv` (learned `W`) | —† |

### 9 · Classification & feature hashing — small grid → fixed / 1×1 answer · 分类与特征哈希

These tasks have a tiny (often 3×3) grid and an answer that depends on a **global property** — symmetry, shape, pixel / colour counts, scatter, marker position — rather than a per-cell transform. Each detects the property and bakes a feature → output lookup, so params run high (≈18k–81k) for ~11–14 points.

这些任务输入极小（常为 3×3），答案取决于**全局属性**（对称性、形状、像素/颜色计数、分散度、标记位置）而非逐格变换。每个求解器检测属性并烘焙「特征 → 输出」查表，因此参数量较大（约 18k–81k），得分约 11–14。

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_symmetry_classify` | 1×1 output: colour 1 if the grid has both H&V symmetry, else 7 · H&V 对称输出 1，否则 7 | content `Slice`+`Gather` flips+`Sub`+`Abs`+`ReduceMax`+`Clip` | ~18k |
| `solve_shape_classify` | hash a small grid's colour-invariant ink layout → output colour · 颜色无关的形状哈希→颜色 | `Slice`+`ReduceSum`(ink)+weighted-sum hash+`Equal`+`Cast`+`Mul`+`Add` | ~45k |
| `solve_count_pattern` | map the number of marker cells to a fixed output motif · 标记计数→固定图案 | `ReduceSum`(count)+`Equal`+`Mul`+`Add` | ~36k |
| `solve_colorcount_pattern` | distinct non-bg colours → fixed 5-pattern (1→top row, 2→diagonal, 3→anti-diagonal) · distinct 色数→图案 | distinct-colour count+`Equal`+`Mul`+`Add` | ~27k |
| `solve_position_color` | map a marker's position to an output colour · 标记位置→颜色 | position detect+`Equal`/`Gather`+`Mul`+`Add` | ~81k |
| `solve_scattered_color` | of two non-bg colours, output the more "scattered" one (fewest pixels per component) · 输出更分散的颜色 | 4-neighbour `Conv`+`ReduceSum`+`ArgMax` | —† |
| `solve_spatial_classify` | hash which positions of a small grid are non-zero → 1×1 output colour · 像素布局哈希→单像素颜色 | `Slice`+`ReduceSum`+weighted-sum hash+`Equal`+`Cast`+`Mul`+`Add` | —† |

> **†** Registered as a capability but currently wins no task on the present set (dominated by a cheaper solver, or no current task matches), so it has no representative built network.
> **†** 已登记但当前未中标（被更省的求解器击败，或暂无任务匹配），故没有代表性网络。

---

<a id="scoring"></a>

## 📏 Scoring & constraints · 评分与约束

**Per-task score · 单任务得分**

```
points = max(1, 25 − ln(memory_bytes + params))   # if every example passes · 全样例通过时
       = 0                                         # otherwise · 否则
```

Smaller is better: a near-zero-memory `Transpose` (0 params) scores the full **25**, while a multi-megabyte connected-component network scores ~**9**.

**Constraints · 约束条件**

| Aspect · 项目 | Requirement · 要求 |
|---|---|
| Input / output tensor · 输入输出张量 | `(1, 10, 30, 30)` float32, one-hot · 独热编码 |
| Tensor names · 张量名 | `"input"` / `"output"` |
| Output decoding · 输出解码 | thresholded at `> 0.0` · 以 `> 0.0` 阈值化 |
| File size · 文件大小 | ≤ 1.44 MB per `.onnx` · 单文件不超过 1.44 MB |
| Banned ops · 禁用算子 | `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`, `COMPRESS`, any `Sequence*`, graph-typed attributes |
| Shapes · 形状 | statically inferable; declare `value_info` when shape inference can't see through dynamic ops · 必须静态可推断 |
| ARC-gen grids · arc-gen 样例 | grids larger than 30×30 are skipped by the scorer · 超过 30×30 的样例自动跳过 |

The banned ops are the reason iterative work (flood-fill, connected-component labelling) is done with a **fixed, unrolled** number of `Pad`/`Conv`/`Max` rounds rather than a loop.

---

<a id="layout"></a>

## 🗂️ Repository layout · 项目结构

```
NeuroGolf/
├── src/neurogolf/
│   ├── grids.py          # one-hot ⇄ grid conversions     · 独热编码转换
│   ├── onnx_ops.py       # ONNX graph helpers             · ONNX 图构建辅助
│   ├── verify.py         # clean-room official scorer      · 评分器独立实现
│   ├── pipeline.py       # run all solvers, keep the best  · 求解器调度
│   └── solvers/          # 98 pattern-specific solvers     · 各类求解器
│       ├── __init__.py   #   ALL_SOLVERS registry          · 求解器登记表
│       └── *.py          #   one module per solver family
├── scripts/
│   ├── download.py             # pull data via Kaggle CLI  · 下载比赛数据
│   ├── build_all.py            # build networks for tasks  · 跑全量任务
│   ├── package_submission.py   # zip the .onnx files       · 打包提交
│   └── submit.py               # post to Kaggle            · 上传 Kaggle
├── networks/             # generated taskNNN.onnx + build_summary.json
├── submissions/          # packaged submission.zip
└── tests/                # pytest cases                    · 单元测试
```

---

<a id="adding"></a>

## 🛠️ Adding a solver · 新增求解器

1. **Create** `src/neurogolf/solvers/my_rule.py` exposing `solve_my_rule(task) -> Optional[onnx.ModelProto]`. The common shape is:
   - `_transform(grid)` — a pure-Python reference implementation of the rule;
   - `_detect(task)` / `_params(task)` — confirm the rule holds on **every** example (return `None` otherwise), detecting any baked constants;
   - `_build(...)` — assemble the ONNX graph (helpers in `onnx_ops.py`).
2. **Register** it in `solvers/__init__.py`: add the import and append it to `ALL_SOLVERS`.
3. **Add a test** in `tests/test_solvers.py` — a positive case (build, run via ONNX Runtime, assert `from_onehot(output) == expected`) and a negative case (assert the solver declines).
4. **Build & verify** the target task: `python scripts/build_all.py --from N --to N`, then check `build_summary.json`.
5. **Update** this README (the relevant family table + the score badge).

**Gotchas · 注意事项**
- Every intermediate tensor must be **statically shaped** — the scorer's memory pass fails on any dynamic dim. Declare `value_info` for outputs that shape inference can't resolve (e.g. `Gather` by a computed index).
- Output must be a clean one-hot: exactly **one channel `> 0`** per real cell, all-zero on padding (otherwise `from_onehot` returns the "ambiguous" colour and the example fails).
- No loops — unroll iterative algorithms to a fixed round count sized for the worst case.

---

<a id="tests"></a>

## 🔬 Tests · 单元测试

```bash
python -m pytest tests/ -q        # 132 passing
```

The suite covers the one-hot round-trip contract and, for each solver family, a **positive** case (the built network reproduces the expected grid through ONNX Runtime) plus a **negative** case (the solver declines a task outside its pattern).

测试覆盖独热编码往返一致性，以及每个求解器的**正例**（生成的网络经 ONNX Runtime 复现期望输出）与**反例**（对不匹配的任务返回 `None`）。

---

<a id="history"></a>

## 📈 Submission history · 提交历史

<details>
<summary><b>v1 → v46 progression (click to expand) · 点击展开</b></summary>

| Submission | Solvers added · 新增求解器 | Tasks | Score |
|---|---|---|---|
| v1 | identity, zero, single-color, remap | 4 | **81.57** |
| v2 | + transpose | 6 | **131.57** |
| v3 | + marker-crop (opset 11) | 7 | **149.93** |
| v4 | + static-crop, kron-scale, resize-scale, shape-aware flip/rot180 | 15 | **276.86** |
| v5 | + rot90-ccw (transpose ∘ flip-v) | 16 | **290.62** |
| v6 | + bbox-strip | 17 | **303.98** |
| v7 | + shift + tile-h | 19 | ~337 |
| v8 | + palindrome H / V / 2D | 26 | ~435 |
| v9 | + majority-fill | 27 | ~453 |
| v10 | + variable-kron | 29 | **480.56** |
| v11 | + conv 1×1 / 3×3 / 5×5 masked | 31 | ~509 |
| v12 | + bbox-color-extract, split-and, scale-detector, variable-shift, gravity-right | 35 | **567.92** |
| v13 | + flood-fill-enclosure | 36 | **578.92** |
| v14 | + column-label | 37 | **592.92** |
| v15 | + repeat-top-rows | 39 | ~605 |
| v16 | + block-mask | 40 | ~616 |
| v17 | + gravity-right-diag | 41 | ~628 |
| v18 | + gravity-down | 45 | **697.88** |
| v19 | + self-fractal | 47 | **726.38** |
| v20 | + rot-tile | 48 | **742.85** |
| v21 | + gravity-up | 49 | **755.96** |
| v22 | + periodic-fill | 53 | **786.58** |
| v23 | + split-logic | 62 | ~924 |
| v24 | + connect-dots | 64 | ~949 |
| v25 | + mirror-complete | 66 | **975.27** |
| v26 | + denoise | 67 | ~988 |
| v27 | + connect-fill | 70 | ~1024 |
| v28 | + drop-into-wall | 71 | ~1037 |
| v29 | + color-bbox-fill | 72 | ~1051 |
| v30 | + nearest-wall | 73 | ~1064 |
| v31 | + cross-laser | 74 | **1076.16** |
| v32 | + halo | 75 | ~1089 |
| v33 | + colour-lines | 76 | ~1102 |
| v34 | + endpoint-bridge | 77 | **1113.94** |
| v35 | + keep-majority | 78 | ~1127 |
| v36 | + blob-recolour | 79 | ~1140 |
| v37 | + recolour-fives | 80 | ~1153 |
| v38 | + dilate-ones | 81 | ~1167 |
| v39 | + count-bar | 82 | ~1182 |
| v40 | + hspan-fill | 83 | **1195.21** |
| v41 | + stamp | 84 | ~1208 |
| v42 | + outline | 85 | ~1222 |
| v43 | + ray-down | 86 | **1232.92** |
| v44 | + isolate-recolour (147, 272) | 88 | ~1258 |
| v45 | + cc-size-recolour (169, 196, 330) | 91 | **1285.81** |
| v46 | + cc-rank-recolour (374) | 92 | **1294.40** |
| v47 | + classification & feature-hash family (56 / 103 / 167 / 186 / 262), remap (276), rot180 (140), diag-tile (7) | 98 | **1368.83** |
| v48 | + axis-gather (116; ↑ 164 / 210 / 311), diag-ray (327), rot-tile-aware (106), block-count-bar (38), odd-panel (207) | 103 | **1470.54** |
| v49 | + odd-panel-aware (65) | 104 | **1481.67** |
| v50 | + period-extend-h (231) | 105 | **1493.34** |
| v51 | + stripe-seeds (13) | 106 | **1506.55** |
| v52 | + slide-to-wall (8) | 107 | **1520.83** |
| v53 | + downscale-majority (130) | 108 | 1537.34 |
| v54 | + untile-half (188) | 109 | 1549.63 |
| v55 | + slide-to-line (25) | 110 | 1561.03 |
| v56 | + largest-comp-crop (36) | 111 | **1570.45** |
| v57 | + diag-block-slide (34) | 112 | 1582.57 |
| v58 | + project-to-block (35) | 113 | 1595.54 |
| v59 | + framed-regions (28) | 114 | 1608.56 |
| v60 | + diag-connect (37) | 115 | 1618.87 |
| v61 | + stamp-top-row (43) | 116 | **1633.03** |
| v62 | + plus-panels (55) | 117 | **1645.19** |
| v63 | + rot180-repair (287) | 118 | **1658.31** |
| v64 | + lattice-count (21) | 119 | **1673.95** |

Bold = score confirmed on the Kaggle leaderboard; `~` = local estimate from `build_summary.json` (the local clean-room scorer matches the official score to two decimals).

**Post-v46 (all Kaggle-confirmed).** v47 built out the **classification & feature-hash family** (family 9 — symmetry / shape / count / colour-count / position). v48–v52 then added a run of geometric & object solvers — shape-aware rotational tiling, diagonal rays, horizontal period extension, odd-one-out panels, two-seed stripes and object-slide — lifting the leaderboard score from **1294.40** to **1520.83 (107 / 400)**. v53+ (`downscale-majority`, `untile-half`, `slide-to-line`, `largest-comp-crop`, `diag-block-slide`, `project-to-block`, `framed-regions`, `diag-connect`, `stamp-top-row`, `plus-panels`, `rot180-repair`, `lattice-count`, …) are on `main` and Kaggle-confirmed at **1673.95 / 119**. Per-solver details live in [Solvers by family](#solvers).

**v46 之后（均已在排行榜确认）。** v47 补全**分类与特征哈希家族**（第 9 类 —— 对称/形状/计数/色数/位置）；v48–v52 再加入一批几何与物体类求解器 —— 形状感知旋转拼接、对角射线、水平周期延展、四面板择异、双种子条纹、物体滑移 —— 将排行榜分数从 **1294.40** 提升到 **1520.83（107 / 400）**。v53 起（`downscale-majority`、`untile-half`、`slide-to-line`、`largest-comp-crop`、`diag-block-slide`、`project-to-block`、`framed-regions`、`diag-connect`、`stamp-top-row`、`plus-panels`、`rot180-repair`、`lattice-count` 等）已合入 `main` 并经 Kaggle 确认，分数 **1673.95 / 119**。各求解器详见 [求解器分类](#solvers)。

</details>

---

<a id="roadmap"></a>

## 🛣️ Roadmap · 后续计划

The pattern families above cover every ARC transformation that reduces to a **detectable, statically-expressible ONNX graph** — including the connected-component frontier (labelling + counting + ranking). The remaining unsolved tasks need a different class of reasoning:

上表已覆盖所有"可检测、可用静态 ONNX 图表达"的变换（含连通分量的标注/计数/排名）。剩余任务需要另一类推理：

- [x] **Small-grid classification** — family 9 (symmetry / shape / count / colour-count / position) now solves tasks 56 / 103 / 167 / 186 / 262. The remaining 1×1-answer tasks **48 / 291 / 346 / 355** still resist a clean, statically-expressible feature hash. · 小网格分类（已部分完成，余 48/291/346/355）
- [x] **Object motion & periodic generation** — `slide_to_wall` (8) slides an object to a wall; `stripe_seeds` (13) and `period_extend_h` (231) generate periodic patterns from seeds. · 物体移动与周期生成（已部分完成）
- [ ] **Object / template matching & copying** — locate a shape and stamp it at a *found* location (beyond the fixed `stamp` solver). · 物体匹配与复制
- [ ] **Multi-step composition / search** — chain several primitive operations. · 多步组合
- [ ] **Memory trimming** — fuse the `(1,10,30,30)` intermediates in the connected-component solvers to lift their ~9-point scores. · 削减中间张量内存

---

<a id="license"></a>

## 📜 License · 许可证

MIT © 2026 [zikuanqi](https://github.com/zikuanqi) — see [LICENSE](LICENSE) · 详见 [LICENSE](LICENSE)
