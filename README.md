<div align="center">

# NeuroGolf 2026

**Tiny ONNX networks that solve ARC-AGI puzzles · 用极小的 ONNX 网络求解 ARC-AGI 谜题**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.21-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.26-FF6F00)](https://onnxruntime.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-NeuroGolf%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/neurogolf-2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/zikuanqi/NeuroGolf)](https://github.com/zikuanqi/NeuroGolf/commits/main)
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen)](tests/)
[![Tasks Solved](https://img.shields.io/badge/tasks_solved-92%2F400-blue)](networks/)
[![Public Score](https://img.shields.io/badge/public_score-1294.40-success)](https://www.kaggle.com/competitions/neurogolf-2026)

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
| **Public score · 公开分数** | **1294.40** |
| **Tasks solved · 通过任务** | **92 / 400** |
| Solvers · 求解器 | 68, in 8 families · 共 68 个，分 8 类 |
| Unit tests · 单元测试 | 88 passing · 88 个全部通过 |
| Networks · 网络文件 | one `networks/taskNNN.onnx` per solved task |

> The full submission-by-submission progression (v1 → v46) lives in [Submission history](#history).
> 逐次提交的成绩进展（v1 → v46）见 [提交历史](#history)。

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
 │    for solver in ALL_SOLVERS:          ~48 pattern‑specific solvers    │
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
# 1. Install deps · 安装依赖
pip install -r requirements.txt

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

68 solvers in 8 families. Each is verified against the official scorer; `Params` is the parameter count of a representative network (it can vary slightly per task because detected constants differ).

68 个求解器分为 8 类。`Params` 列为代表性网络的参数量（不同任务因检测出的常量不同会略有差异）。

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
| `solve_variable_shift` | fixed-offset shift, variable input shape · 定偏移量平移，变输入尺寸 | `Slice`+`Pad`+`Concat` | ~50 |

### 2 · Crop & extract — output is a sub-region · 裁剪与提取

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_static_crop` | fixed-offset, fixed-size subrect · 定长定位裁剪 | `Slice`+`Pad` | ~14 |
| `solve_marker_crop` | crop a fixed window around a unique marker pixel · 围绕唯一标记的定长裁剪 | `Slice`→`ReduceSum`→`ArgMax`→`Slice`→`Pad` | ~27 |
| `solve_bbox_strip` | crop to the bbox of non-background cells · 非背景外接矩形提取 | `ReduceSum`+`Sub`+row/col `ArgMax`+`Gather`+`Less`+`Mul` | ~76 |
| `solve_bbox_color_extract` | crop to the bbox of the majority / rarest colour · 按主色/稀有色外接矩形裁剪 | `ReduceSum`+`ArgMax`/`ArgMin`+`OneHot`+`Mul`+`Cast`+`Gather`+`Mod`+`Less` | ~66 |

### 3 · Scale, tile & symmetry — repeat or mirror the grid · 缩放、平铺与镜像

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_kron_scale` | N×N pixel expansion, constant shape · 同尺寸像素块复制 | 2× `Gather`+`Pad` | ~40 |
| `solve_resize_scale` | N× nearest upscale, variable shape · 变尺寸 N 倍最近邻放大 | `Slice`+`Resize` | ~24 |
| `solve_scale_detector` | N× nearest upscale **or** downscale · N 倍最近邻放大或缩小 | `Slice`+`Resize` | ~24 |
| `solve_variable_kron` | scale by N = `count(non-zero)` / `count(distinct)` · 变 N 倍（N 来自输入特征） | `ReduceSum`+`Cast`+`Div`+`Min`+`Gather`×2+`Less`+`Mul` | ~138 |
| `solve_tile_h` | horizontal tile-N, variable width · 变宽 N 倍水平复刻 | `ReduceSum`+`ReduceMax`+`Mod`+`Gather`+`Less`+`Mul` | ~67 |
| `solve_palindrome_h` / `solve_palindrome_v` | mirror-concat to the right / bottom edge · 镜像拼接 | shape-aware `Where`+`Gather`+`Less` mask | ~68 |
| `solve_palindrome_2d` | four-quadrant 2D mirror · 四象限二维镜像 | palindrome-h ∘ palindrome-v | ~133 |
| `solve_rot_tile` | N×N → 2N×2N as four rotations (I/90/180/270) · 四象限旋转拼接 | `Slice`+`Transpose`+`Gather`×4+`Concat`×3+`Pad` | ~23 |
| `solve_self_fractal` | N×N → N²×N² self-similar fractal keyed by a selector colour · 自相似分形放大 | `Slice`+`Tile`+`Gather`×2 (Kron mask)+`ArgMax`+`Mul`+`Pad`+`Concat` | ~57 |
| `solve_block_mask` | N×N → N²×N² masked tiling + channel-0 recovery · 块掩码平铺 | `Slice`+`ReduceSum`+`Less`+`Tile`+`Mul`+`Add` | ~78 |
| `solve_repeat_top_rows` | runtime period detection (P=2/3/4) + weighted candidate mix · 顶部行周期复刻 | period detection + `Gather` + weighted blend | ~29 |
| `solve_periodic_fill` | restore a periodic tiling from a partially-erased grid · 从残缺网格恢复周期平铺 | per-period `Pad`+`Slice`+`Max` (log-doubling) + `Where` select + bbox clip | ~9k |

### 4 · Recolour — per-cell / per-object / whole-grid colour rules · 重新着色

| Solver | Pattern · 识别模式 | Key ONNX ops | Params |
|---|---|---|---|
| `solve_remap` | per-pixel colour lookup · 像素级颜色映射 | 1×1 `Conv` | 100 |
| `solve_single_color` | uniform fill, same shape · 同尺寸纯色填充 | 1×1 `Conv` | 100 |
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
| `solve_hspan_fill` | fill each bg cell flanked left & right by a wall colour with a fixed colour · 填充被同色墙左右夹住的格 | `Gather`+exclusive/reverse `CumSum`+`Mul`/`Sub`/`Add` | ~24 |
| `solve_endpoint_bridge` | bridge each row's two dots: nearer-dot colour each side, colour 5 at the midpoint · 连接两点，中点为 5 | `ReduceMax` column ramps+`Greater`/`Less` side masks+`Mul`/`Add` | ~117 |
| `solve_color_lines` | colour-2 markers fill their column; other colours fill their row (drawn on top) · 颜色2竖线，其余横线覆盖 | `ReduceMax`(col-has-2 / row colour)+`Sub`/`Mul`/`Add` priority blend | ~32 |
| `solve_cross_laser` | each marker fires a full row+column "plus"; differing-colour crossings → 2 · 十字射线，异色交叉为 2 | `ReduceMax`(row/col colour)+`ReduceSum`(same-colour test)+`Sub`/`Mul`/`Add` | ~31 |
| `solve_ray_down` | carry each marker's colour straight down its column · 颜色沿列向下填充 | per-colour cumulative-max down (log-doubling `Pad`+`Slice`+`Max`)+`Equal`/`Greater` | ~95 |
| `solve_drop_into_wall` | colour-1 cells fall into a full colour-5 wall row in their column · 颜色1落入颜色5的墙 | `Gather`+`ReduceSum`/`ReduceMax`+`Mul`(outer)+`Relu`+`Concat` | ~12 |
| `solve_nearest_wall` | recolour each marker with the colour of the nearer of two facing walls · 重涂为较近一堵墙的颜色 | `ReduceSum`/`ReduceMax`+`Gather`(far wall)+`Less`/`Greater` half-masks+`Mul`/`Add` | ~75 |
| `solve_flood_fill_enclosure` | fill colour-0 cells fully enclosed by a single source colour · 填充被单色完全包围的色0 | `ArgMax`+`Equal`+58×(`Pad`+`Conv`+`Greater`+`Max`) border BFS+`Gather`+`Sub`+`Slice`+`Concat` | ~1.8k |
| `solve_gravity_right` | each row's cells slide right until blocked · 重力向右 | `ReduceSum`+`CumSum`+`Where`+`Mul` | ~94 |
| `solve_gravity_down` | each column's cells fall to the bottom edge · 重力向下 | `ReduceSum`+`ReduceMax`(height)+`Less`+`Mul`+`Slice`+`Concat` | ~66 |
| `solve_gravity_up` | each column's cells rise to the top, preserving order · 重力向上保序 | `ReduceSum`+sort-key+`TopK`+`Tile`+`GatherElements` | ~42 |
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
| `solve_stamp` | replace each marker with a fixed 3×3 colour motif detected from the task · 用检测出的3×3图案替换标记 | per-colour 3×3 `Conv` (180-flipped stamp kernel)+`Greater`+`Mul`/`Sub`/`Add` | ~51 |

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
| `solve_split_and` | split along a colour-5 separator and AND the two halves · 沿色5分隔做 AND | `Slice`+`Pad`+`Sub`+`Mul`+`Less`+`Cast`+`And` | ~8.1k |
| `solve_split_logic` | split into two halves (L-R or T-B, ± divider) and combine with and/or/xor/nor/nand · 两半布尔运算 | `Slice`+`ReduceSum`+`Mul`/`Max`/`Sub`+`Pad`+`Concat` | ~20 |
| `solve_conv3x3` | least-squares fit of a 3×3 conv (no bias) · 无偏置3×3卷积拟合 | 3×3 `Conv` | 900 |
| `solve_conv1x1_masked` / `solve_conv3x3_masked` / `solve_conv5x5_masked` | K×K conv + bias, masked to non-padding cells · 带偏置的 K×K 卷积+掩码 | `Conv`+`ReduceSum`+`Mul` | 100 / 910 / 2510 |

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
│   └── solvers/          # 68 pattern-specific solvers     · 各类求解器
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
python -m pytest tests/ -q        # 88 passing
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

Bold = score confirmed on the Kaggle leaderboard; `~` = local estimate from `build_summary.json` (the local clean-room scorer matches the official score to two decimals).

</details>

---

<a id="roadmap"></a>

## 🛣️ Roadmap · 后续计划

The pattern families above cover every ARC transformation that reduces to a **detectable, statically-expressible ONNX graph** — including the connected-component frontier (labelling + counting + ranking). The remaining unsolved tasks need a different class of reasoning:

上表已覆盖所有"可检测、可用静态 ONNX 图表达"的变换（含连通分量的标注/计数/排名）。剩余任务需要另一类推理：

- [ ] **Spatial classification** for the 1×1-answer tasks (e.g. tasks 48 / 56 / 103 / 291 / 346 / 355) — the answer depends on shape / symmetry, not on colour counts. · 形状/对称性分类
- [ ] **Object / template matching & copying** — locate a shape and stamp it elsewhere. · 物体匹配与复制
- [ ] **Multi-step composition / search** — chain several primitive operations. · 多步组合
- [ ] **Memory trimming** — fuse the `(1,10,30,30)` intermediates in the connected-component solvers to lift their ~9-point scores. · 削减中间张量内存

---

<a id="license"></a>

## 📜 License · 许可证

MIT © 2026 [zikuanqi](https://github.com/zikuanqi) — see [LICENSE](LICENSE) · 详见 [LICENSE](LICENSE)
