<div align="center">

# NeuroGolf 2026

**Tiny ONNX networks that solve ARC-AGI puzzles · 用极小的 ONNX 网络求解 ARC-AGI 谜题**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.21-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.26-FF6F00)](https://onnxruntime.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-NeuroGolf%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/neurogolf-2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/zikuanqi/NeuroGolf)](https://github.com/zikuanqi/NeuroGolf/commits/main)
[![Tests](https://img.shields.io/badge/tests-340%20passing-brightgreen)](tests/)
[![Tasks Solved](https://img.shields.io/badge/tasks_solved-225%2F400-blue)](networks/)
[![Local Score](https://img.shields.io/badge/local_score-3026.18-success)](networks/build_summary.json)
[![Public Score](https://img.shields.io/badge/public_score-2883.25-blue)](https://www.kaggle.com/competitions/neurogolf-2026)

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
| **Tasks solved · 通过任务** | **225 / 400** |
| **Local score · 本地总分** | **3026.18** — clean-room scorer over `build_summary.json` · 独立评分器统计 |
| **Public score · 公开分数 (Kaggle)** | **2704.10** — leaderboard-confirmed at 200/400 (exact match to local); tasks 201+ queued for next daily reset · 已确认 200/400（与本地完全一致）；201 及之后待额度重置后提交 |
| Solvers · 求解器 | in 9 families · 分 9 类 |
| Unit tests · 单元测试 | 340 passing · 340 个全部通过 |
| Networks · 网络文件 | 225 × `networks/taskNNN.onnx` (one per solved task) · 每个解出任务一个 |

> Local score is **2884.40 / 214 tasks**; the Kaggle leaderboard is confirmed at **2883.25 / 214** (exact match to local) — **past the halfway mark**. The v1 → v46 progression lives in [Submission history](#history).
> 本地分数 **2884.40 / 214 解**；Kaggle 排行榜确认 **2883.25 / 214**（与本地完全一致）—— **已过半**。v1 → v46 进展见 [提交历史](#history)。

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
 │    for solver in ALL_SOLVERS:          ~157 pattern‑specific solvers   │
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

157 solvers in 9 families. Each is verified against the official scorer; `Params` is the parameter count of a representative network (it can vary slightly per task because detected constants differ). A few solvers are registered as capabilities but currently win no task (dominated or non-matching on the present set); these are marked `—†`.

157 个求解器分为 9 类。`Params` 列为代表性网络的参数量（不同任务因检测出的常量不同会略有差异）。少数求解器已登记但当前未中标（被更省的求解器击败或不匹配现有任务），以 `—†` 标记。

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
| `solve_bbox_strip_zero` | crop to the bbox of the non-majority colour, mapping that bg colour → 0 (task 259) · 裁剪非多数色外接矩形并将背景置 0 | `bbox_strip` dynamic crop + final bg-channel → channel-0 remap | 88 |
| `solve_quadrant_crop` | output the top-left h×h quadrant of the lone 2h×2h content block (task 39) · 取内容外接框的左上四分之一 | bbox top-left via `CumSum` + reversed-offset `Gather`×2 + `Pad` | ~27 |
| `solve_largest_comp_crop` | crop to the largest 4-connected component (a solid blob amid single-cell noise), holes kept as bg (task 36) · 裁剪到最大连通块 | iterated label max-propagation + all-pairs size + largest mask + bg-fill + bbox `Gather` crop | ~1k |
| `solve_bbox_color_extract` | crop to the bbox of the majority / rarest colour · 按主色/稀有色外接矩形裁剪 | `ReduceSum`+`ArgMax`/`ArgMin`+`OneHot`+`Mul`+`Cast`+`Gather`+`Mod`+`Less` | ~66 |

### 3 · Scale, tile & symmetry — repeat or mirror the grid · 缩放、平铺与镜像

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_kron_scale` | N×N pixel expansion, constant shape · 同尺寸像素块复制 | 2× `Gather`+`Pad` | ~40 |
| `solve_resize_scale` | N× nearest upscale, variable shape · 变尺寸 N 倍最近邻放大 | `Slice`+`Resize` | ~24 |
| `solve_downscale_majority` | block-majority **down**scale: each k×k block → its majority colour (task 130, 9×9→3×3) · 块多数色降采样 | `AveragePool`(kernel=stride=block)+`ArgMax`+`OneHot`+`Pad` | ~26 |
| `solve_panel_summary` | MxN grid of blank-separated colour panels → the MxN grid of each panel's dominant colour (task 184) · 面板阵列→各面板主色阵列 | `CumSum` band indices + one-hot bands + double `MatMul` per-channel binning + `ArgMax` | ~50 |
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
| `solve_fractal_blocks` | 3×3 block grid (packed) → 9×9 self-fractal of its 3×3 occupancy meta (task 195) · 块阵列降采样后自相似分形 | bbox third-index binning (`MatMul`) → 3×3 meta + reshape-broadcast Kronecker square | ~70 |
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
| `solve_recolor_in_block` | recolour every source-colour cell that falls inside a region colour's bounding box (tasks 70, 166) · 在区域外接框内改色 | channel `Slice` + bbox span via log-doubling shift-`Max` flood + `e_D−e_S` add | ~252 |
| `solve_interior_recolor` | recolour the one-cell-eroded interior of each solid shape to 8, borders kept (task 120) · 把实心形状内部改为 8 | 4-way `Pad`+`Slice` neighbour shift + `ReduceSum`(input·neighbour) same-colour test, product = interior mask | ~69 |
| `solve_interior_recolor_aware` | same erosion, but the interior fill colour is detected from the examples (task 294 → 2) · 同上但填充色自动检测 | `interior_recolor` with a parameterised target channel | ~69 |
| `solve_rect_interior_rank` | two stacked rectangles: repaint each eroded interior by size rank — smaller→1, larger→2 (task 156) · 两叠放矩形按大小给内部上色 | 4-neighbour erosion + `CumSum` over row 0→1 transitions to label top/bottom bands + masked-`ReduceSum` areas | 93 |
| `solve_ring_recolor` | recolour solid rectangles by orthogonal-neighbour count — corner→1, edge→4, interior→2 (task 283) · 按邻居数给矩形描边上色 | 4-way `Pad`+`Slice` neighbour-sum + equality masks at 2/3/4 | ~80 |
| `solve_line_cross_swap` | where a horizontal and vertical line cross, swap which one is on top (task 293) · 交叉处互换上层线条 | runtime colour detect by column span + band intersection + `(e_B−e_A)` swap | ~30 |
| `solve_box_stretch` | stretch a hollow box until its edge reaches an aligned marker, redrawing frame+interior (task 281) · 把空心框拉伸到标记 | most-common-colour border + bbox + inside/outside marker split + 4-way direction select + index-mask frame redraw | ~80 |
| `solve_gap_fill` | fill the gap between two rectangles with 8 over their interior intersection (task 341) · 用 8 填两矩形之间的缝隙 | per-channel profile 1-D erosion + `ReduceProd` interior intersection + prefix/suffix-`Max` gap detection | ~190 |
| `solve_merge_pair` | a colour-3 cell 4-adjacent to a colour-2 → 8, the 2 erased (task 344) · 相邻 3 与 2 合并为 8 | 4-way `Pad`+`Slice` neighbour shift + `Max` + `e_8−e_3` / `e_0−e_2` | ~30 |
| `solve_cross_move` | a full colour cross moves down-left by the colour-5 marker count (task 362) · 十字按标记数左下平移 | full row/col colour detect + `ReduceSum`(ch5)=k + `row==cr+k`/`col==cc−k` index masks | ~62 |
| `solve_row_checker` | two uniform rows → diagonal checkerboard of the two colours (task 373) · 两行纯色→对角棋盘 | `ReduceSum` content mask + baked `(r+c)%2` parity + two channel `Slice` colours | 920 |
| `solve_five_isolate` | keep the colour-5 scaffold, repaint it the one other colour, clear the rest (task 389) · 保留 5 骨架并改为另一色 | channel-5 `Slice` mask + histogram `Greater` for the colour + `e_C`·mask + background fill | 33 |
| `solve_color_sort_column` | list the colours ordered by cell-count as a K×1 column (task 393) · 按格数排序输出颜色列 | per-channel histogram + 10×10 pairwise `Greater` rank + row-index placement | 79 |
| `solve_plus_panels` | fill the central plus of an 8-line panel grid with fixed colours, corners stay blank (task 55) · 八线面板网格的中央十字按固定色填充 | content-aware divider detection + `CumSum` band index + outer-product regions | ~32 |
| `solve_zero_color` | erase all cells of one fixed colour (→ 0) · 抹除某固定颜色 | channel-mask `Mul` | 10 |
| `solve_majority_fill` | constant-shape rect filled with the majority non-bg colour · 常尺寸主色填充 | `ReduceSum`+`TopK`+`Greater`+`And`+`Where`+`OneHot`+`Mul`+`Pad` | ~32 |
| `solve_largest_bbox_fill` | constant fill with the colour whose input bbox is largest · 以最大外接矩形的颜色纯色填充 | `ReduceMax`+`ArgMax`(bbox)+area+`ArgMax`(chan)+`OneHot`+`Pad` | ~94 |
| `solve_row_uniform_indicator` | per row: colour X if the input row is uniform, else Y · 行均匀则填 X 否则 Y | `ReduceSum`+`Greater`+`ReduceMax`+`Less` in-grid mask+`OneHot`+`Mul` | ~79 |
| `solve_column_label` | label colour-5 columns 1,2,3… by top-most marker row · 按最上标记行给列编号 | `Slice`+`ReduceSum`+`ArgMax`+pairwise `Less`/`Equal`/`And`/`Or` ranking+`Mul`+`Concat` | ~80 |
| `solve_keep_majority` | keep the most-frequent colour; recolour all other markers to 5 · 保留最多色，其余→5 | `ReduceSum`(count)+`ReduceMax`+`Equal`(argmax)+`Mul`/`Add` | ~31 |
| `solve_band_majority` | denoise striped bands: each row/col → its majority colour, band axis auto-detected (task 359) · 条带去噪，各行/列取多数色 | per-line `ArgMax` majority (both axes) + line-uniformity strength flag + blend + grid mask | ~21 |
| `solve_column_template` | fill every row from the template row's repeated-column pattern, recoloured by each row's seed (task 197) · 按模板列模式填充各行 | fullest-row template + pairwise column-equality `MatMul` + masked-min first-occurrence col + `Gather` | ~52 |
| `solve_odd_col_recolor` | recolour the non-background cells in odd columns to 4 (task 252) · 奇数列的非背景格改为 4 | baked odd-column mask × non-bg mask + `e_4` paint | ~22 |
| `solve_triangle_diag` | left-anchored 2-segment → diagonal triangle, 3 above / 2 on / 1 below (task 256) · 2 段→对角三角，上 3 下 1 | index mask `col<=c1+r0-row` + row-vs-r0 colour split | ~58 |
| `solve_pocket_drop` | downward-opening ∪ staple drops a 4 to the grid floor in the gap column (task 126) · 朝下的 ∪ 形在缺口列底部落一个 4 | Pad+Slice pocket detect (bg with non-bg up/left/right) → per-column flag × last-real-row one-hot | ~76 |
| `solve_square_complete` | fill the missing 4th corner of an L-tromino 2×2 square with 1 (task 81) · 补全 L 形三格为 2×2 方块的缺角为 1 | four diagonal Pad+Slice shifts → OR of 2×2-corner conjunctions → paint `e_1` | ~14 |
| `solve_midpoint_plus` | draw a 3-plus at the midpoint of two aligned 1-markers (task 371) · 在两个对齐标记的中点画 3 的十字 | centroid `Σidx/count` + index masks for the plus arms → paint `e_3` | ~12 |
| `solve_elbow_connect` | L-path of 8 from a 2-marker to a 3-marker (along the 2's row, then the 3's column) (task 246) · 2 与 3 之间画 8 的折线 | centroids → row/col index masks for the two segments → paint `e_8` (endpoints kept) | ~10 |
| `solve_mirror_quad` | reflect a 2-shape across the H/V axes through a 3-block → 4 mirrored copies (task 112) · 以 3 块为中心把 2 形镜像成四份 | 3-centroid axes → `Gather` on `round(2c−i)` reflection indices (bounds-masked) → union paint `e_2` | ~10 |
| `solve_arrow_ray` | a solid triangle + base marker shoots a ray of the marker's colour from the apex to the edge (task 51) · 三角形从尖端朝指向方向射出标记色的射线 | colour-histogram finds the unique count-1 marker; shape-centroid vs marker fixes direction; half-line painted the runtime marker colour | ~6 |
| `solve_diag_shoot` | satellite cells off a 2×2 block shoot diagonal rays away from it to the edge (task 190) · 方块旁的卫星格朝外射出对角射线 | isolated-cell + diagonal-neighbour split into 4 directions → log-step doubling-scan propagation → paint runtime colour | ~3 |
| `solve_ring_reverse` | reverse the colour order of concentric square rings (task 203) · 同心方环颜色顺序反转 | ring depth `d`, diagonal palette `in[k,k]` → `Gather` by mirror depth `D−d` | ~3 |
| `solve_corner_burst` | a 2-marker bursts into four fixed-colour diagonal corners (UL 3 / UR 6 / DL 8 / DR 7) (task 266) · 2 标记炸成四个对角定色角 | four diagonal Pad+Slice shifts of the 2-mask → paint `e_3/e_6/e_8/e_7`, clear the 2 | ~5 |
| `solve_col3_recolor` | recolour colour-4 cells in every third column (col % 3 == 0) to 6 (task 292) · 每隔三列的 4 改为 6 | baked column mask `col%3==0` × channel-4 mask → paint `e_6` | ~6 |
| `solve_vperiod3` | extend a vertically period-3 texture to fill the grid (task 215) · 将竖直周期 3 的纹理铺满整格 | OR non-bg mask over all multiples of 3 via doubling-scan vertical shifts → paint runtime colour | ~3 |
| `solve_key_cycle` | cycle row-0 key colours into solid bands below the separator: row r → key[(r-2) mod W] (task 297) · 键行颜色循环成实心条带 | transpose row 0 to a colour column + runtime-modulo `Gather` over rows | ~5 |
| `solve_laser_cross` | an 8-segment fills its column, a 2-segment fills its row, 4 at the crossing (task 299) · 8 段填满整列、2 段填满整行，交点为 4 | `ReduceMax` column/row flags → paint `e_8` / `e_2` / `e_4` (overlap) | ~5 |
| `solve_enclosure_recolor` | recolour a shape to 8 iff its component encloses a hole (task 279) · 围出空洞的形状改为 8（首个连通域求解器） | **connected components** via two iterative flood-fills (border-reachability of bg, then component spread) | ~94 |
| `solve_key_flood` | flood each 5-block with the row-0 key colour of the column it spans (task 354) · 5 块染成所在列的键颜色 | read key value off row 0 → seed on the block's key column → **scalar-value max-dilation flood** through the component | ~10 |
| `solve_hole_size_fill` | fill each enclosed square hole with colour 5 + side (task 302) · 方形空洞按边长填 5+边长 | border-flood enclosure + masked run-length (square hole → run = side) → paint `5+side` | ~5 |
| `solve_hole_parity_fill` | fill each enclosed square hole 2 (even side) / 7 (odd) (task 204) · 方形空洞按边长奇偶填 2/7 | shares enclosure + run-length; `fill = 2 + 5·(side mod 2)` | ~6 |
| `solve_blob_size_color` | recolour each small off-field blob by size: 1→3, 2→2, 3→1 (task 369) · 小斑块按大小染色 | `colour = 3 − maxdeg`, where maxdeg = max-flood of 4-neighbour degree over the component | ~4 |
| `solve_bbox_fill` | recolour structure cells inside each colour-2 component's bounding box to 4 (task 77) · 把 2 连通域外接框内的结构格染成 4 | **per-component bbox rasterization** via a co-evolving bound-flood (growing inbox carries min/max row/col; BIG-padded shifts pick inbox neighbours) | ~30 |
| `solve_cross_center` | full-grid cross of 6 through each hollow box's bbox centre, over background (task 94) · 过每个空心框中心画 6 的全幅十字 | min/max floods give each shape cell its bbox; centre row/col flags broadcast to a cross, painted over bg | ~25 |
| `solve_fold_mirror` | reflect a shape across a 2-marker fold line; background → 3 (task 62) · 沿 2 标记折叠镜像，背景变 3 | centroids choose fold orientation; `Gather` on `round(2A+off−i)` reflects (V/H selected), painted runtime colour | ~8 |
| `solve_bar_half` | recolour the bottom floor(height/2) cells of each vertical 2-bar to 8 (task 320) · 每根 2 竖条下半段（向下取整）变 8 | per-bar minr/maxr via min/max floods; cell → 8 iff `maxr−r < ⌊(h)/2⌋` | ~6 |
| `solve_corner_rect_fill` | fill the interior of each 4-corner-marker rectangle with 2 (task 273) · 把 4 角标记的矩形内部填 2 | bg cell is interior iff a 4 exists in each diagonal quadrant — four exclusive `CumSum` passes | ~7 |
| `solve_blob_recolor` | two non-bg colours: repaint the majority "blob" with the rarer "key" colour, clear the rest · 用稀有色重涂多数团块 | `ReduceSum` counts+`Equal`+`Greater`+`Sub`/`Mul`/`Add` | ~21 |
| `solve_recolor_fives` | repaint every colour-5 cell with its row's marker colour · 颜色-5 取所在行标记色 | `Gather`(ch5)+`ReduceMax`(row marker)+`Sub`/`Mul`/`Add` | ~21 |
| `solve_filled_rect` | replace a monochromatic filled rectangle with a new colour (or crop it) · 纯色矩形替换/裁剪 | constant output via `Sub`+`Add` | ~9k |

### 5 · Lines, gaps & gravity — draw, extend, fill, slide · 连线、填隙与重力

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_connect_dots` | fill the span between the first and last same-colour dot per row/column · 连接每行/列首尾同色点 | `Slice`+`CumSum`+`Gather`(reverse)+`Greater`+`And`+`Concat` | ~37 |
| `solve_l_connect` | L-path of colour 4 from an 8 to a 2: down the 8's column to the 2's row, then across (task 335) · 8→2 的 L 形连线 | marker positions via index ramps + `Min`/`Max` span + `col==c8`/`row==r2` index masks | ~57 |
| `solve_connect_pairs` | connect each colour's two aligned markers with a line; vertical on top at crossings (task 92) · 连接每色两标记成线，交叉处竖线在上 | per-channel forward/reverse `CumSum` prefix∧suffix (both axes) + vertical-priority blend | ~21 |
| `solve_move_toward` | step the 3-marker one cell toward the 4-marker, 4 fixed (task 353) · 3 向 4 走一步 | marker positions + `Sign` of offset + index mask at the new cell | ~92 |
| `solve_connect_fill` | fill the gap between aligned same-colour dots with one fixed colour · 用固定色连接同色点 | `CumSum`×4 (prefix/suffix)+`Greater`+`And`+`Or`+`ReduceMax`+`Concat` | ~64 |
| `solve_diag_connect` | connect each same-colour pair with a diagonal line (task 37) · 用对角线连接同色点对 | 9-channel 4-way diagonal log-doubling cumulative-`Max` + main/anti `Mul` intersect + `Concat` | ~85 |
| `solve_hspan_fill` | fill each bg cell flanked left & right by a wall colour with a fixed colour · 填充被同色墙左右夹住的格 | `Gather`+exclusive/reverse `CumSum`+`Mul`/`Sub`/`Add` | ~24 |
| `solve_endpoint_bridge` | bridge each row's two dots: nearer-dot colour each side, colour 5 at the midpoint · 连接两点，中点为 5 | `ReduceMax` column ramps+`Greater`/`Less` side masks+`Mul`/`Add` | ~117 |
| `solve_color_lines` | colour-2 markers fill their column; other colours fill their row (drawn on top) · 颜色2竖线，其余横线覆盖 | `ReduceMax`(col-has-2 / row colour)+`Sub`/`Mul`/`Add` priority blend | ~32 |
| `solve_stripe_seeds` | two edge seeds → periodic alternating stripes (vertical if on top/bottom rows, horizontal if on left/right cols), period = seed gap (task 13) · 两个边缘种子生成周期交替条纹 | colour-value projection + seed-extent `ReduceMin`/`Max` + dynamic `Mod` stripe + edge-flag orientation select + `OneHot` | ~82 |
| `solve_cross_laser` | each marker fires a full row+column "plus"; differing-colour crossings → 2 · 十字射线，异色交叉为 2 | `ReduceMax`(row/col colour)+`ReduceSum`(same-colour test)+`Sub`/`Mul`/`Add` | ~31 |
| `solve_corner_rays` | a colour line down column 0 sprouts an anti-diagonal of 2 and a bottom row of 4 (task 84) · 列0色线生成反对角线与底行 | grid side H = `ReduceSum`(row_has) + `row+col == H−1` / `row == H−1` index masks | ~82 |
| `solve_diag_x` | draw the full diagonal X through a single marker in its colour (task 141) · 过单个标记画对角十字 | marker colour/position via weighted `ReduceSum` + `abs(\|row−mr\|−\|col−mc\|)<0.5` mask | ~30 |
| `solve_float_up` | float each colour bar up by its own height (task 128) · 每个色条上浮自身高度 | per-channel bar height + batched `MatMul` with shift matrices M[c,i,j]=(j==i+h_c) + background rebuild | ~963 |
| `solve_divider_fold` | fold a shape across a full divider cross into a 4-fold-symmetric grid, recolour to the line colour, drop the divider (task 109) · 按分隔十字四向折叠并改色去线 | runtime L/S colour + divider detect, 4× reversed-index `Gather` reflections (colour-only), `Max` merge, `e_L−e_S` recolour, row/col-skip `Gather` resize | ~159 |
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
| `solve_connect_box_markers` | join each row/column-aligned marker to a solid box with a line in its own colour (task 64) · 把对齐标记用同色线连到方块 | solidity box detect (count==bbox area) + 4-way log-doubling shift-`Max` colour flood + gap `Mul` | ~524 |
| `solve_gravity_right_diag` | per-channel diagonal slide toward the centre of mass · 按质心方向对角滑移 | `ReduceSum`+`ReduceMax`+`ArgMax`+`Mul`+`Slice`+`Concat`+`Min`+`Sub` | 150 |

### 6 · Morphology & shape — dilate, erode, outline, stamp · 形态学与形状

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_dilate_ones` | expand every marker into a filled 3×3 block of colour 1 · 膨胀为颜色1的实心3×3 | 3×3 ones `Conv`+`Greater`+`Cast`+`Mul`/`Sub`/`Add` | ~31 |
| `solve_explode_corners` | explode a 2×2 colour block into four point-reflected 2×2 stamps, block kept (task 225) · 2×2 块向四角点反射爆炸 | neighbour-classify cells TL/TR/BL/BR + dilate-then-translate by fixed offsets + grid-mask | ~206 |
| `solve_block_quadrant` | fill a 2×2 block with the four scattered markers, one per quadrant (task 342) · 按象限把四标记填入 2×2 块 | colour-8 centroid + quadrant masks × marker channels + block-cell index masks | ~80 |
| `solve_diagonal_markers` | mark each 2×2 block's four diagonal corners 1/2/3/4 (task 230) · 在 2×2 块四角对角放 1/2/3/4 | neighbour-classify cells TL/TR/BL/BR + one-cell diagonal `Pad`+`Slice` shift + paint | ~70 |
| `solve_halo` | keep each marker, ring its background neighbours with colour 1 · 在标记四周围一圈颜色1 | 3×3 ones `Conv`(dilation)+`Greater`+`Cast`+`Sub`/`Mul`/`Add` | ~32 |
| `solve_outline` | keep each shape's perimeter, erase its interior · 保留边缘，擦除内部 | 3×3 cross `Conv`(4-neighbour count)+`Greater`+`Mul`/`Sub`/`Add` | ~21 |
| `solve_cut_diagonals` | erase both diagonals of a solid NxN square, leaving an X-hole (task 375) · 抹掉正方形两条对角线 | `row==col` / `row+col==N−1` index masks (N−1 = max content row) | ~9 |
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
| `solve_band_sort` | read stacked colour bands into an ordered colour strip, row or column per orientation (task 115) · 读取色带顺序输出色条 | per-colour centroid + orientation from spread + 10×10 pairwise-`Less` rank + placement-matrix `Reshape`/`Pad` | ~102 |
| `solve_staircase` | grow a 1×W run of K cells into a (W/2)×W staircase, row r having K+r cells (task 295) · 单行色段长成阶梯 | detect colour/count/width + `col < K+row` index mask + background rebuild | ~30 |
| `solve_split_and` | split along a colour-5 separator and AND the two halves · 沿色5分隔做 AND | `Slice`+`Pad`+`Sub`+`Mul`+`Less`+`Cast`+`And` | ~8.1k |
| `solve_split_logic` | split into two halves (L-R or T-B, ± divider) and combine with and/or/xor/nor/nand · 两半布尔运算 | `Slice`+`ReduceSum`+`Mul`/`Max`/`Sub`+`Pad`+`Concat` | ~20 |
| `solve_odd_panel` | four panels (2×2 layout ± blank divider), three identical → output the unique fourth (task 207) · 四面板中输出唯一不同的一个 | per-panel `Slice`+pairwise `Sub`/`Abs`/`ReduceSum`/`Less` equality+`Less`(agree=0) select+`Mul`/`Add`+`Pad` | ~45 |
| `solve_odd_panel_aware` | same odd-one-out but **shape-aware** — grid size varies (5×5/7×7/11×11, task 65) · 同上但形状感知，网格尺寸可变 | runtime `n` from content extent + index-shifted `Gather` panel re-alignment + masked pairwise-equality select | ~66 |
| `solve_odd_panel_shape` | strip of 3×3 panels (vertical or horizontal) → output the odd-*shaped* one (task 263) · 输出形状不同的 3×3 面板 | 5D panel reshape both orientations + per-cell majority shape + differ-from-majority select + orientation blend | ~10 |
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
│   └── solvers/          # 157 pattern-specific solvers    · 各类求解器
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
python -m pytest tests/ -q        # 244 passing
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
| v65 | + quadrant-crop (39) | 120 | 1688.33 |
| v66 | + connect-box-markers (64) | 121 | **1698.41** |
| v67 | + recolor-in-block (70, 166) | 123 | **1726.44** |
| v68 | + corner-rays (84) | 124 | **1739.58** |
| v69 | + divider-fold (109) | 125 | 1751.08 |
| v70 | + band-sort (115) | 126 | 1763.78 |
| v71 | + interior-recolor (120) | 127 | 1775.56 |
| v72 | + float-up (128) | 128 | 1787.88 |
| v73 | + diag-x (141) | 129 | 1801.02 |
| v74 | + staircase (295) | 130 | **1814.51** |
| v75 | + box-stretch (281) | 131 | **1826.86** |
| v76 | + gap-fill (341) | 132 | 1840.15 |
| v77 | + merge-pair (344) | 133 | 1853.93 |
| v78 | + cross-move (362) | 134 | 1866.03 |
| v79 | + checkerboard / five-isolate / colour-sort (373/389/393) | 137 | 1909.67 |
| v80 | + rect-interior-rank (156) | 138 | **1921.30** |
| v81 | + 8 bespoke solvers (259/283/294/293/225/335/342/353) | 146 | 2019.60 |
| v82 | + cut-diagonals (375) | 147 | 2032.96 |
| v83 | + odd-panel-shape / band-majority / connect-pairs (263/359/92) | 150 | 2073.23 |
| v84 | + panel-summary (184) | 151 | 2086.49 |
| v85 | + column-template (197) | 152 | 2100.52 |
| v86 | + fractal-blocks (195, 217) | 154 | **2128.66** |
| v87 | + diagonal-markers (230) | 155 | _2140.73 local · folded into v89 batch_ |
| v88 | + odd-col-recolor (252) | 156 | _2154.04 local · folded into v89 batch_ |
| v89 | + triangle-diag (256) | 157 | **2166.58** |
| v90 | + pocket-drop (126) | 158 | _2179.86 local · folded into v96 batch_ |
| v91 | + square-complete (81) | 159 | _2192.75 local · folded into v96 batch_ |
| v92 | + midpoint-plus (371) | 160 | _2206.19 local · folded into v96 batch_ |
| v93 | + elbow-connect (246) | 161 | _2219.40 local · folded into v96 batch_ |
| v94 | + mirror-quad (112) | 162 | _2232.64 local · folded into v96 batch_ |
| v95 | + arrow-ray (51) | 163 | _2245.26 local · folded into v96 batch_ |
| v96 | + diag-shoot (190) | 164 | **2257.25** |
| v97 | + ring-reverse (203) | 165 | **2270.70** |
| v98 | + corner-burst (266) | 166 | **2282.65** |
| v99 | + col3-recolor (292) | 167 | **2296.32** |
| v100 | + vperiod3 (215) | 168 | _2309.19 local · folded into v107 batch_ |
| v101 | + key-cycle (297) | 169 | _2322.81 local · folded into v107 batch_ |
| v102 | + laser-cross (299) | 170 | _2335.35 local · folded into v107 batch_ |
| v103 | + enclosure-recolor (279, first CC) | 171 | _2345.85 local · folded into v107 batch_ |
| v104 | + key-flood (354, CC) | 172 | _2357.10 local · folded into v107 batch_ |
| v105 | + hole-size-fill (302, CC) | 173 | _2367.65 local · folded into v107 batch_ |
| v106 | + hole-parity-fill (204, CC) | 174 | _2378.20 local · folded into v107 batch_ |
| v107 | + blob-size-color (369, CC) | 175 | **2389.93** |
| v108 | + bbox-fill (77, per-component rasterization) | 176 | _2399.81 local · submit pending_ |
| v109 | + cross-center (94, bbox rasterization) | 177 | _2409.59 local · submit pending_ |
| v110 | + fold-mirror (62) | 178 | _2422.32 local · submit pending_ |
| v111 | + bar-half (320) | 179 | _2432.79 local · submit pending_ |
| v112 | + corner-rect-fill (273) | 180 | _2445.96 local · submit pending_ |
| v113 | + neighbor-halo (15) | 181 | **2458.66** ✓ confirmed (incl. 176–181) |
| v114 | + align-to-anchor (30) | 182 | _2471.39 local_ |
| v115 | + panel-complete (33) | 183 | **2484.64** ✓ confirmed (incl. 182–183) |
| v116 | + crop-tile-h (57) | 184 | **2497.58** ✓ confirmed |
| v117 | + panel-max-fill (59) | 185 | **2510.03** ✓ confirmed |
| v118 | + stamp-at-markers (75); +bbox-recolor-ones opt (70) | 186 | **2521.58** ✓ confirmed |
| v119 | + left-third (67) | 187 | _2538.31 local · queued_ |
| v120 | + marker-box-interior (88) | 188 | _2551.21 local · queued_ |
| v121 | + stack-to-band (93) | 189 | **2563.36** ✓ confirmed (incl. 187–189) |
| v122 | + edge-frame (114) | 190 | **2576.32** ✓ confirmed |
| v123 | + eight-center-crop (121) | 191 | **2589.36** ✓ confirmed |
| v124 | + diag-ray-pair (136) | 192 | **2602.00** ✓ confirmed |
| v125 | + blob-box-fill (139) | 193 | **2613.48** ✓ confirmed |
| v126 | + bar-echo (148) | 194 | _2625.84 local · queued_ |
| v127 | + panel-pair-flag (149) | 195 | _2639.51 local · queued_ |
| v128 | + cross-ring (151) | 196 | **2652.93** ✓ confirmed (incl. 194–196) |
| v129 | + edge-pair-lines (161) | 197 | **2665.68** ✓ confirmed |
| v130 | + key-meta-mask (170) | 198 | _2679.33 local · submitting_ |
| v131 | + symmetric-shape-crop (174) | 199 | **2690.93** ✓ confirmed (incl. 198) |
| v132 | + crop-flip-h (177) | **200 — halfway!** | **2704.10** ✓ confirmed |
| v133 | + reflect-marker-dir (181), quadrant-corner-map (183) | 202 | _2729.48 local · queued_ |
| v134 | + open-2x2 (193) | 203 | _2742.36 local · queued_ |
| v135 | + maze-enclose (187, hard) | 204 | _2754.28 local · queued_ |
| v136 | + band-drill (202, hard) | 205 | _2765.39 local · queued_ |
| v137 | + stamp-template-at-five (206) | 206 | _2777.65 local · queued_ |
| v138 | + divider-rays (212) | 207 | _2790.45 local · queued_ |
| v139 | + mirror-tile-3x2 (211) | 208 | _2804.26 local · queued_ |
| v140 | + rotate-into-regions (214) | 209 | _2818.06 local · queued_ |
| v141 | + marker-ring (220) | 210 | _2830.46 local · queued_ |
| v142 | + alt-ray-right (232) | 211 | _2843.27 local · queued_ |
| v143 | + right-then-down-ray (237) | 212 | _2856.03 local · queued_ |
| v144 | + tall-short-lines (254) | 213 | _2868.98 local · queued_ |
| v145 | + drop-one-recolor (261, new) + midpoint-fill-h (258 opt) | 214 | **2883.25** ✓ confirmed |
| v146 | perf: cheaper 272 (+1.14) & 266 re-solves | 214 | _2884.40 local · queued_ |
| v147 | + flood-ones (243) | 215 | _2896.26 local · queued_ |
| v148 | + smallest-blob-two (277) | 216 | _2905.95 local · queued_ |
| v149 | + domino-ring (278) | 217 | _2919.49 local · queued_ |
| v150 | + empty-line-fill (303) | 218 | _2933.58 local · queued_ |
| v151 | + hollow-color-pick (291) | 219 | _2948.90 local · queued_ |
| v152 | + crop-swap-pair (290) | 220 | _2961.70 local · queued_ |
| v153 | + line-triangle-expand (348) | 221 | _2974.61 local · queued_ |
| v154 | + two-halo-ones (352) | 222 | _2988.78 local · queued_ |
| v155 | + overlay-mirror-halves (360) | 223 | _3001.43 local · queued_ |
| v156 | + line-pierce-box (379) + box-fill-gap-ray (336) | 225 | _3026.18 local · queued_ |

Bold = score confirmed on the Kaggle leaderboard; `~` = local estimate from `build_summary.json` (the local clean-room scorer matches the official score to two decimals).

**Post-v46 (all Kaggle-confirmed).** v47 built out the **classification & feature-hash family** (family 9 — symmetry / shape / count / colour-count / position). v48–v52 then added a run of geometric & object solvers — shape-aware rotational tiling, diagonal rays, horizontal period extension, odd-one-out panels, two-seed stripes and object-slide — lifting the leaderboard score from **1294.40** to **1520.83 (107 / 400)**. v53+ (`downscale-majority`, `untile-half`, `slide-to-line`, `largest-comp-crop`, `diag-block-slide`, `project-to-block`, `framed-regions`, `diag-connect`, `stamp-top-row`, `plus-panels`, `rot180-repair`, `lattice-count`, `quadrant-crop`, `connect-box-markers`, `recolor-in-block`, `corner-rays`, `divider-fold`, `band-sort`, `interior-recolor`, `float-up`, `diag-x`, `staircase`, `box-stretch`, `gap-fill`, `merge-pair`, `cross-move`, `row-checker`, `five-isolate`, `colour-sort`, `rect-interior-rank`, `bbox-strip-zero`, `ring-recolor`, `interior-recolor-aware`, `line-cross-swap`, `explode-corners`, `l-connect`, `block-quadrant`, `move-toward`, `cut-diagonals`, `odd-panel-shape`, `band-majority`, `connect-pairs`, `panel-summary`, `column-template`, `fractal-blocks`, …) are on `main` and Kaggle-confirmed; the `diagonal-markers`, `odd-col-recolor`, `triangle-diag`, `pocket-drop`, `square-complete`, `midpoint-plus`, `elbow-connect`, `mirror-quad`, `arrow-ray`, `diag-shoot`, `ring-reverse`, `corner-burst` & `col3-recolor` batch (155–167) lifts the leaderboard to **2296.32 / 167** (matches local 2296.33 to ~0.01); the `vperiod3`, `key-cycle`, `laser-cross`, `enclosure-recolor`, `key-flood`, `hole-size-fill`, `hole-parity-fill` & `blob-size-color` batch (168–175; the last five are **connected-components** solvers via iterative flood-fill) lifts the leaderboard to **2389.93 / 175** (exact match to local); `bbox-fill`, `cross-center` (per-component **bounding-box rasterization**), `fold-mirror`, `bar-half` & `corner-rect-fill` (176–180, **2445.96 local**) plus `neighbor-halo` (15 — a fixed plus/X neighbour stamp: a `1` paints `7` on its 4 orthogonal neighbours, a `2` paints `4` on its 4 diagonals; 181) is Kaggle-confirmed at **2458.66 / 181**. Then `align-to-anchor` (30 — every colour block slides vertically so its top meets the anchor `1`-block, via a runtime `MatMul` shift matrix) and `panel-complete` (33 — a 17×17 grid of 3×3 panels; each panel is completed to the union template, the missing cells painted with the divider colour) bring **183 / 400, 2484.65 local** — Kaggle-confirmed **2484.64**. Then `crop-tile-h` (57 — crop the shape's bounding box and repeat it side-by-side once, `H×2W`, via the bbox-strip Gather crop plus a `MatMul` column-shift) brings **184 / 400, 2497.58 local** (Kaggle-confirmed **2497.58**). Then `panel-max-fill` (59 — an 11×11 / 3×3-panel grid; count markers per panel with a stride-4 ones `Conv`, fill the max-count panel(s) solid via a matching `ConvTranspose`) brings **185 / 400, 2510.04 local** (Kaggle-confirmed **2510.03**). Then `stamp-at-markers` (75 — stamp the top-left 3×3 template centred on each `1`-marker, as nine colour-weighted shifted marker masks) brings **186 / 400, 2521.58 local** (Kaggle-confirmed **2521.58**). Then `left-third` (67 — keep the leftmost `W/3` columns of a 3×-wide tiling; recover `W` from the non-padding columns and mask the rest; 16.7 pts at 31 params) brings **187 / 400, 2538.31 local** (queued for the next reset). Then `marker-box-interior` (88 — crop the interior of a 4-corner marker box (= the non-bg bbox) and repaint its inner shape in the marker colour, gathered from the box's top-left corner) brings **188 / 400, 2551.21 local** (queued). Then `stack-to-band` (93 — markers gravitate onto a solid 5-band and stack against its edge as 5s; band orientation detected within the grid width, horizontal/vertical results blended by a gate) brings **189 / 400, 2563.37 local** (Kaggle-confirmed **2563.36**). Then `edge-frame` (114 — grow the grid to `(H+2)×(W+2)` by edge-replication via a clamped two-axis `Gather`, with the four corners blanked) brings **190 / 400, 2576.33 local** (Kaggle-confirmed **2576.32**). Then `eight-center-crop` (121 — locate the lone `8` by channel argmax, clamp-`Gather` its 3×3 neighbourhood, and recolour the 8 to the shape's only other colour) brings **191 / 400, 2589.36 local** (Kaggle-confirmed **2589.36**). Then `diag-ray-pair` (136 — a colour-1 and colour-2 2×2 block each emit a fixed diagonal ray to the edge, drawn as constant-`r−c` diagonal masks past the block corner) brings **192 / 400, 2602.00 local** (Kaggle-confirmed **2602.00**). Then `blob-box-fill` (139 — fill each 4-blob's 3×3 bounding box holes with 7; min-floods the row/col indices through the 8-connected blob via `-MaxPool(-V)` steps, marks the box centre, and dilates it once) brings **193 / 400, 2613.48 local** (Kaggle-confirmed **2613.48**). Then `bar-echo` (148 — a marked vertical bar fills 8s from itself to each marker (marker→4) and the partner bar echoes the same rows, at the same relative offsets, fully with 8; closed-form via per-row `ArgMax` columns and a 30×30 offset-comparison matrix) brings **194 / 400, 2625.84 local** (queued). Then `panel-pair-flag` (149 — an 11×11 / 3×3-panel grid: output 3×3 marks each panel holding ≥2 sixes, via one stride-4 ones `Conv` and a threshold; 48 params, 13.67 pts) brings **195 / 400, 2639.51 local** (queued). Then `cross-ring` (151 — a full row and a full column cross; the intersection's eight neighbours are painted 4, located via per-line non-bg counts vs grid width/height) brings **196 / 400, 2652.93 local** (Kaggle-confirmed **2652.93**). Then `edge-pair-lines` (161 — markers at opposite border ends connect into full rows/columns and all noise clears; the marker colour is the one with no cells besides its endpoints, `count == 2×lines`, computed wholly on the channel axis; 27 params) brings **197 / 400, 2665.69 local** (Kaggle-confirmed **2665.68**). Then `key-meta-mask` (170 — an N×N colour key masked by the arrangement of solid blocks; the block colour is isolated by **erode-then-dilate** of its mask so key-internal block-colour cells don't confuse the bboxes, then block centres are sampled at runtime-computed stride) brings **198 / 400, 2679.33 local** (submitting). Then `symmetric-shape-crop` (174 — crop the one shape that is mirror-symmetric in its bbox; all nine colours tested at once with **batched per-channel MatMul shift matrices** for the mirror-alignment and the crop) brings **199 / 400, 2690.93 local** (Kaggle-confirmed **2690.93**). Then `crop-flip-h` (177 — bbox crop mirrored left-right via a runtime reflection `MatMul`, `S[j,c]=1 iff j+c==w−1`) reaches the **halfway milestone: 200 / 400, 2704.10 local** (Kaggle-confirmed **2704.10**). Then `reflect-marker-dir` (181 — mirror the 8-shape horizontally, the left/right direction read off the 4-marker's displaced top arm, via a runtime reflection `MatMul`) and `quadrant-corner-map` (183 — recolour an inner 8-pattern by the four grid-corner keys, split into quadrants) bring **202 / 400, 2729.48 local** (queued). Then `open-2x2` (193 — keep only cells inside a solid 2×2+ block, a morphological opening built from `MaxPool` erosion/dilation; removed colour cells revert to background) brings **203 / 400, 2742.36 local** (queued). Then `maze-enclose` (187 — a maze of one wall colour over background; a 4-connected `MaxPool` flood from the 30×30 frame through background-or-padding paints border-reachable background `3` and sealed-off pockets `2`) brings **204 / 400, 2754.28 local** (queued). Then `band-drill` (202 — solid colour stripes where each 0-hole drills a full 0-line across its band; a band-gated vertical flood (`MatMul` row-shifts gated by `bandVec[r]·bandVec[r±1]`) handles horizontal stripes, the transpose handles vertical, blended by an orientation gate) brings **205 / 400, 2765.39 local** (queued). Then `stamp-template-at-five` (206 — copy a multi-colour template, centred, onto the lone `5`; the bbox-centre-to-marker offset drives a runtime 2D `MatMul` translation) brings **206 / 400, 2777.65 local** (queued). Then `divider-rays` (212 — across a full 5-row, each `2` marker grows a line toward the divider and each `1` marker grows away to the edge; four directional `CumSum` passes masked to each half) brings **207 / 400, 2790.45 local** (queued). Then `mirror-tile-3x2` (211 — every 3×2 input reflect-tiles to a fixed 9×4; a constant index map `output[r,c]=input[rowmap[r],colmap[c]]` = two `Gather`s + a mask; 13.8 pts) brings **208 / 400, 2804.26 local** (queued). Then `rotate-into-regions` (214 — fill two 5-divided regions with the 3×3 key rotated 90°cw and 180°; a fixed 3×11 cell permutation done as a flatten + `Gather` reflection-map) brings **209 / 400, 2818.06 local** (queued). Then `marker-ring` (220 — surround each marker with a 3×3 ring whose colour is fixed by the marker colour (`2→1, 3→6, 8→4`), via per-colour `MaxPool` dilation masked to background) brings **210 / 400, 2830.46 local** (queued). Then `alt-ray-right` (232 — each isolated marker shoots a rightward ray to the grid edge alternating its own colour and `5`; the active region is `CumSum(markers) > 0`, a second `CumSum` gives the per-cell offset whose parity picks the marker colour or `5`, all confined to the grid by the occupancy mask) brings **211 / 400, 2843.27 local** (queued). Then `right-then-down-ray` (237 — each marker draws a solid ray right to the edge, then turns down the rightmost grid column to the bottom, lower markers overwriting; the horizontal leg is a `CumSum` flood, the vertical leg a sample-and-hold "fill-down" of the per-row marker colour via Hillis-Steele doubling shifts, the right column located as the last grid column) brings **212 / 400, 2856.03 local** (queued). Then `tall-short-lines` (254 — among several vertical line segments, recolour the tallest column `1` and the shortest non-empty column `2`, erasing the rest; per-column counts via a row `ReduceSum`, then a max and a non-zero min select the survivors) brings **213 / 400, 2868.98 local** (queued). Then `drop-one-recolor` (261 — the coloured block falls one row and is repainted `2`, a single `Pad`/`Slice` downward shift re-confined to the grid) brings **214 / 400**, and `midpoint-fill-h` (258 — a background cell flanked by `1`s left and right becomes `2`, two single-column shifts of the `1` mask intersected with the background; a +0.76 cheaper re-solve of an already-covered task) lifts the local total to **2883.25** (Kaggle-confirmed **2883.25**, exact). Then two lighter re-solves of already-covered tasks — `isolated-two-recolor` (272 — a `2` with no orthogonal `2`-neighbour becomes `1`, four single-step shifts of the `2` mask; **+1.14**) and `diag-corner-stamp` (266 — each `2` is erased and its four diagonal neighbours painted `3/6/8/7` via diagonal shifts; +0.01) — lift the local total to **2884.40**. Then `flood-ones` (243 — a noisy grid already holds some `1` cells; every background `0` reachable from a `1` through orthogonal background steps becomes `1`, an iterative 4-connected `MaxPool` flood masked to the background) brings **215 / 400, 2896.26 local** (queued). Then `smallest-blob-two` (277 — every 8-connected blob is recoloured `1` except the one with the fewest cells, which becomes `2`; blobs are labelled by iterated 3×3 `MaxPool` max-propagation of a unique per-cell id, sized by an all-pairs label match, and the global minimum size selects the `2` blob) brings **216 / 400, 2905.95 local** (queued). Then `domino-ring` (278 — every `2` that has an orthogonally-adjacent `2` gets its 3×3 surround painted `3`; isolated 2s are left alone, via an orthogonal-neighbour test then a 3×3 `MaxPool` dilation of the pair, kept on background cells) brings **217 / 400, 2919.49 local** (queued). Then `empty-line-fill` (303 — any grid row or column with no foreground cell is filled solid `2`; emptiness is the grid mask minus the per-row/column `ReduceMax` of the foreground) brings **218 / 400, 2933.58 local** (queued). Then `hollow-color-pick` (291 — among several solid rectangular blocks, exactly one is a hollow frame; output its colour as a single cell, found per-colour via `count < bbox-area` using spatial `ReduceSum` counts and `ReduceMax`/`ReduceMin` row/column extents) brings **219 / 400, 2948.90 local** (queued). Then `crop-swap-pair` (290 — crop the shape's bounding box to the top-left and exchange its two colours, the swap done by a runtime 10×10 permutation `MatMul` over channels and the crop by the standard clamped two-axis `Gather`) brings **220 / 400, 2961.70 local** (queued). Then `line-triangle-expand` (348 — a vertical `7`-line of length `L` becomes a downward triangle: row `r` spans `c0 ± (L-1-r)`, painted `7` where the offset from the line is even and `8` where odd, selected by `r + |c-c0| ≤ L-1`) brings **221 / 400, 2974.61 local** (queued). Then `two-halo-ones` (352 — every `2` paints its eight background neighbours `1` (overlapping halos merge), other colours untouched, via a 3×3 `MaxPool` dilation of the `2` mask kept on background) brings **222 / 400, 2988.78 local** (queued). Then `overlay-mirror-halves` (360 — a full `5` column splits the grid into equal halves; the output is the left half with the horizontally-mirrored right half laid on top (they never collide), the mirror done by a column `Gather`) brings **223 / 400, 3001.43 local** — **past 3000** (queued). Then `line-pierce-box` (379 — each `2` shoots a ray to the nearest full `8`-line on each side, painting the ray `2`; every pierced line bulges into a 3×3 box of `8` with the pierce cell itself `2`; the ray is a prefix-line-count (`CumSum`) band gated by whether a line exists on that side, both orientations via transpose) and `box-fill-gap-ray` (336 — a `5` frame with one missing border cell: its interior fills `8` and an `8` ray escapes through the gap to the edge) bring **225 / 400, 3026.18 local** (queued). Per-solver details live in [Solvers by family](#solvers).

**v46 之后（均已在排行榜确认）。** v47 补全**分类与特征哈希家族**（第 9 类 —— 对称/形状/计数/色数/位置）；v48–v52 再加入一批几何与物体类求解器 —— 形状感知旋转拼接、对角射线、水平周期延展、四面板择异、双种子条纹、物体滑移 —— 将排行榜分数从 **1294.40** 提升到 **1520.83（107 / 400）**。v53 起（`downscale-majority`、`untile-half`、`slide-to-line`、`largest-comp-crop`、`diag-block-slide`、`project-to-block`、`framed-regions`、`diag-connect`、`stamp-top-row`、`plus-panels`、`rot180-repair`、`lattice-count`、`quadrant-crop`、`connect-box-markers`、`recolor-in-block`、`corner-rays`、`divider-fold`、`band-sort`、`interior-recolor`、`float-up`、`diag-x`、`staircase`、`box-stretch`、`gap-fill`、`merge-pair`、`cross-move`、`row-checker`、`five-isolate`、`colour-sort`、`rect-interior-rank`、`bbox-strip-zero`、`ring-recolor`、`interior-recolor-aware`、`line-cross-swap`、`explode-corners`、`l-connect`、`block-quadrant`、`move-toward`、`cut-diagonals`、`odd-panel-shape`、`band-majority`、`connect-pairs`、`panel-summary`、`column-template`、`fractal-blocks` 等）已合入 `main` 并经 Kaggle 确认；`diagonal-markers`、`odd-col-recolor`、`triangle-diag`、`pocket-drop`、`square-complete`、`midpoint-plus`、`elbow-connect`、`mirror-quad`、`arrow-ray`、`diag-shoot`、`ring-reverse`、`corner-burst`、`col3-recolor`（155–167）批次将排行榜提升到 **2296.32 / 167**（与本地 2296.33 一致）；`vperiod3`、`key-cycle`、`laser-cross`、`enclosure-recolor`、`key-flood`、`hole-size-fill`、`hole-parity-fill`、`blob-size-color`（168–175；后五个是**连通域**求解器，用迭代洪泛填充）批次将排行榜提升到 **2389.93 / 175**（与本地完全一致）；`bbox-fill`、`cross-center`（逐连通域**外接框光栅化**）、`fold-mirror`、`bar-half`、`corner-rect-fill`（176–180，**本地 2445.96**）以及 `neighbor-halo`（15 —— 固定十字/叉号邻域印章：`1` 在 4 个正交邻格涂 `7`，`2` 在 4 个对角邻格涂 `4`；181）已经 Kaggle 确认 **2458.66 / 181**。随后 `align-to-anchor`（30 —— 每个色块垂直滑动，使顶部对齐锚点 `1` 块，用运行时 `MatMul` 移位矩阵实现）与 `panel-complete`（33 —— 17×17 的 3×3 面板网格，每个面板补全到并集模板，缺失格用分隔线颜色填充）将进度带到 **183 / 400，本地 2484.65** —— Kaggle 确认 **2484.64**。随后 `crop-tile-h`（57 —— 裁出形状外接框并水平复制一次，`H×2W`，复用 bbox-strip 的 Gather 裁剪加 `MatMul` 列移位）将进度带到 **184 / 400，本地 2497.58**（Kaggle 确认 **2497.58**）。随后 `panel-max-fill`（59 —— 11×11 的 3×3 面板网格，用步长 4 的全一 `Conv` 统计每个面板的标记数，再用对应的 `ConvTranspose` 把标记最多的面板填满）将进度带到 **185 / 400，本地 2510.04**（Kaggle 确认 **2510.03**）。随后 `stamp-at-markers`（75 —— 把左上角 3×3 模板以每个 `1` 标记为中心盖章，用 9 个按颜色加权的位移标记掩码实现）将进度带到 **186 / 400，本地 2521.58**（Kaggle 确认 **2521.58**）。随后 `left-third`（67 —— 保留 3× 宽平铺的最左 `W/3` 列，从非填充列恢复 `W` 并把其余列掩零；31 参数得 16.7 分）将进度带到 **187 / 400，本地 2538.31**（待额度重置提交）。随后 `marker-box-interior`（88 —— 裁出四角标记框（即非背景外接框）的内部，并把内部形状重涂成标记色，标记色取自框左上角）将进度带到 **188 / 400，本地 2551.21**（待提交）。随后 `stack-to-band`（93 —— 散布的标记向实心 5-带靠拢并贴边堆叠成 5；在网格宽度内判定带的方向，横/竖结果用门控混合）将进度带到 **189 / 400，本地 2563.37**（Kaggle 确认 **2563.36**）。随后 `edge-frame`（114 —— 用钳位的双轴 `Gather` 做边缘复制，把网格扩成 `(H+2)×(W+2)`，四角清零）将进度带到 **190 / 400，本地 2576.33**（Kaggle 确认 **2576.32**）。随后 `eight-center-crop`（121 —— 用通道 argmax 定位唯一的 `8`，钳位 `Gather` 取其 3×3 邻域，把 8 重涂成形状里唯一的另一种颜色）将进度带到 **191 / 400，本地 2589.36**（Kaggle 确认 **2589.36**）。随后 `diag-ray-pair`（136 —— 颜色 1 与颜色 2 的两个 2×2 块各向边缘发射一条固定方向的对角射线，用常数 `r−c` 对角掩码从块角向外绘制）将进度带到 **192 / 400，本地 2602.00**（Kaggle 确认 **2602.00**）。随后 `blob-box-fill`（139 —— 把每个 4 色块 3×3 外接框内的空洞填成 7；用 `-MaxPool(-V)` 步进在 8 连通块内做行/列下标的最小值洪泛，标出框中心再膨胀一次）将进度带到 **193 / 400，本地 2613.48**（Kaggle 确认 **2613.48**）。随后 `bar-echo`（148 —— 带标记的竖条向每个 8 标记填充 8（标记变 4），另一根竖条在相同相对行整行回声填 8；用逐行 `ArgMax` 列定位加 30×30 偏移比对矩阵闭式实现）将进度带到 **194 / 400，本地 2625.84**（待提交）。随后 `panel-pair-flag`（149 —— 11×11 的 3×3 面板网格：输出 3×3 标出含 ≥2 个 6 的面板，用一次步长 4 的全一 `Conv` 加阈值实现；48 参数得 13.67 分）将进度带到 **195 / 400，本地 2639.51**（待提交）。随后 `cross-ring`（151 —— 一条整行线与一条整列线交叉，交点的 8 个邻格涂成 4；用逐行/逐列非背景计数对比网格宽高定位）将进度带到 **196 / 400，本地 2652.93**（Kaggle 确认 **2652.93**）。随后 `edge-pair-lines`（161 —— 位于边界两端的成对标记连成整行/整列，其余噪声全部清除；标记色的判据是除端点外没有别的格子，即 `count == 2×lines`，完全在通道维上计算；仅 27 参数）将进度带到 **197 / 400，本地 2665.69**（Kaggle 确认 **2665.68**）。随后 `key-meta-mask`（170 —— 用大块排列图案掩码 N×N 颜色钥匙；先对块色掩码做**先腐蚀后膨胀**，避免钥匙内部的块色格子干扰外接框，再按运行时步长采样块中心）将进度带到 **198 / 400，本地 2679.33**（提交中）。随后 `symmetric-shape-crop`（174 —— 裁出唯一一个在自身外接框内左右镜像对称的形状；9 种颜色用**逐通道批量 MatMul 移位矩阵**同时做镜像对齐与裁剪）将进度带到 **199 / 400，本地 2690.93**（Kaggle 确认 **2690.93**）。随后 `crop-flip-h`（177 —— 外接框裁剪并左右镜像，用运行时反射 `MatMul`（`S[j,c]=1 当且仅当 j+c==w−1`）实现）达成**过半里程碑：200 / 400，本地 2704.10**（Kaggle 确认 **2704.10**）。随后 `reflect-marker-dir`（181 —— 把 8 形状水平镜像，左右方向由 4 标记错位的顶端臂决定，用运行时反射 `MatMul` 实现）与 `quadrant-corner-map`（183 —— 按四个网格角的钥匙分象限重涂内部 8 图案）将进度带到 **202 / 400，本地 2729.48**（待提交）。随后 `open-2x2`（193 —— 只保留位于实心 2×2 及以上块内的格子，用 `MaxPool` 腐蚀/膨胀构成的形态学开运算实现，被去除的颜色格子还原为背景）将进度带到 **203 / 400，本地 2742.36**（待提交）。随后 `maze-enclose`（187 —— 单一墙色在背景上画出的迷宫；从 30×30 边框出发、经背景或填充区做 4 连通 `MaxPool` 洪泛，把从边界可达的背景涂 `3`、被封死的口袋涂 `2`）将进度带到 **204 / 400，本地 2754.28**（待提交）。随后 `band-drill`（202 —— 实心颜色条纹，每个 0 洞沿其所在条带钻出一整条 0 线；用受条带颜色门控的竖向洪泛（`MatMul` 行移位，门控 `bandVec[r]·bandVec[r±1]`）处理横条纹，转置处理竖条纹，再按朝向门控混合）将进度带到 **205 / 400，本地 2765.39**（待提交）。随后 `stamp-template-at-five`（206 —— 把多色模板以中心对齐复制到唯一的 `5` 上；外接框中心到标记的偏移驱动运行时 2D `MatMul` 平移）将进度带到 **206 / 400，本地 2777.65**（待提交）。随后 `divider-rays`（212 —— 跨一整行 5 分隔线，每个 `2` 标记朝分隔线方向长出一条线、每个 `1` 标记背离分隔线长到边缘；用 4 次按半区掩码的方向性 `CumSum`）将进度带到 **207 / 400，本地 2790.45**（待提交）。随后 `mirror-tile-3x2`（211 —— 每个 3×2 输入反射平铺成固定 9×4；常数索引映射 `output[r,c]=input[rowmap[r],colmap[c]]` = 两次 `Gather` 加一个掩码；13.8 分）将进度带到 **208 / 400，本地 2804.26**（待提交）。随后 `rotate-into-regions`（214 —— 把两个被 5 分隔的区域填上 3×3 钥匙顺时针旋转 90° 和旋转 180° 的结果；固定 3×11 的格子置换，用 flatten + `Gather` 反射映射实现）将进度带到 **209 / 400，本地 2818.06**（待提交）。随后 `marker-ring`（220 —— 给每个标记套一圈 3×3 环，环色由标记色固定决定（`2→1、3→6、8→4`），用逐色 `MaxPool` 膨胀并掩到背景）将进度带到 **210 / 400，本地 2830.46**（待提交）。随后 `alt-ray-right`（232 —— 每个孤立标记向网格右边缘发出一条射线，交替填自身颜色与 `5`；活动区为 `CumSum(标记) > 0`，再做一次 `CumSum` 得到每格偏移，其奇偶决定填标记色还是 `5`，并用占据掩码把射线限制在网格内）将进度带到 **211 / 400，本地 2843.27**（待提交）。随后 `right-then-down-ray`（237 —— 每个标记先向右画一整条射线到边缘，再沿最右一列向下拐到底，较下方的标记覆盖较上方的；横段用 `CumSum` 洪泛，竖段用 Hillis-Steele 倍增移位对逐行标记色做采样保持式"向下填充"，最右列由"最后一个网格列"定位）将进度带到 **212 / 400，本地 2856.03**（待提交）。随后 `tall-short-lines`（254 —— 在若干竖向线段中，把最高的一列重涂为 `1`、最矮的非空列重涂为 `2`，其余清除；逐列计数用逐行 `ReduceSum`，再用最大值与非零最小值选出幸存列）将进度带到 **213 / 400，本地 2868.98**（待提交）。随后 `drop-one-recolor`（261 —— 彩色块下落一行并重涂为 `2`，用一次 `Pad`/`Slice` 下移再用占据掩码限制在网格内）带到 **214 / 400**；再加 `midpoint-fill-h`（258 —— 左右被 `1` 夹住的背景格变 `2`，把 `1` 掩码左右各移一列后相交背景；这是对一个已覆盖任务的 +0.76 更省实现）把本地总分提升到 **2883.25**（Kaggle 确认 **2883.25**，完全一致）。随后两个对已覆盖任务的更省实现 —— `isolated-two-recolor`（272 —— 没有正交 `2` 邻居的 `2` 变成 `1`，用 `2` 掩码的四次单步移位；**+1.14**）与 `diag-corner-stamp`（266 —— 每个 `2` 被擦除、其四个对角邻格涂成 `3/6/8/7`，用对角移位实现；+0.01）—— 把本地总分提升到 **2884.40**。随后 `flood-ones`（243 —— 含噪网格里已有若干 `1`；每个能经正交背景步从某个 `1` 到达的背景 `0` 都变成 `1`，用迭代 4 连通 `MaxPool` 洪泛并掩到背景）将进度带到 **215 / 400，本地 2896.26**（待提交）。随后 `smallest-blob-two`（277 —— 每个 8 连通块重涂为 `1`，唯独格数最少的那个变成 `2`；用唯一逐格编号经迭代 3×3 `MaxPool` 取最大传播来标记连通块，用全配对标签匹配求每块大小，再用全局最小尺寸选出 `2` 块）将进度带到 **216 / 400，本地 2905.95**（待提交）。随后 `domino-ring`（278 —— 每个有正交相邻 `2` 的 `2`，其 3×3 周围涂成 `3`；孤立的 `2` 保持不变；用正交邻居判定加 3×3 `MaxPool` 膨胀，只保留落在背景上的格子）将进度带到 **217 / 400，本地 2919.49**（待提交）。随后 `empty-line-fill`（303 —— 任何没有前景格的整行或整列填成实心 `2`；空判据为网格掩码减去逐行/逐列前景的 `ReduceMax`）将进度带到 **218 / 400，本地 2933.58**（待提交）。随后 `hollow-color-pick`（291 —— 在若干实心矩形块中恰有一个是空心框；输出它的颜色为单格，逐色用 `count < 外接框面积` 判定，计数用空间 `ReduceSum`、外接框范围用逐行/逐列 `ReduceMax`/`ReduceMin`）将进度带到 **219 / 400，本地 2948.90**（待提交）。随后 `crop-swap-pair`（290 —— 把形状外接框裁到左上角并交换其两种颜色；交换用运行时 10×10 置换矩阵在通道维 `MatMul` 实现，裁剪用标准的钳位双轴 `Gather`）将进度带到 **220 / 400，本地 2961.70**（待提交）。随后 `line-triangle-expand`（348 —— 长度为 `L` 的竖直 `7` 线展开成向下的三角形：第 `r` 行覆盖 `c0 ± (L-1-r)`，离线偏移为偶数处涂 `7`、奇数处涂 `8`，由 `r + |c-c0| ≤ L-1` 选定）将进度带到 **221 / 400，本地 2974.61**（待提交）。随后 `two-halo-ones`（352 —— 每个 `2` 把它的八个背景邻格涂成 `1`（重叠的光环合并），其它颜色不变；用 `2` 掩码的 3×3 `MaxPool` 膨胀并保留落在背景上的格子）将进度带到 **222 / 400，本地 2988.78**（待提交）。随后 `overlay-mirror-halves`（360 —— 一整列 `5` 把网格分成相等两半；输出为左半叠加水平镜像后的右半（两者从不冲突），镜像用列 `Gather` 实现）将进度带到 **223 / 400，本地 3001.43** —— **突破 3000**（待提交）。随后 `line-pierce-box`（379 —— 每个 `2` 向两侧最近的整条 `8` 线各射出一条 `2` 射线；每条被刺穿的线鼓成一个 3×3 的 `8` 盒，刺穿点本身为 `2`；射线用前缀线计数 `CumSum` 的同带掩码并按该侧是否有线门控，两种朝向用转置处理）与 `box-fill-gap-ray`（336 —— 缺一个边格的 `5` 框：内部填 `8`，并从缺口射出一条 `8` 线到边缘）将进度带到 **225 / 400，本地 3026.18**（待提交）。各求解器详见 [求解器分类](#solvers)。

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
