<div align="center">

# NeuroGolf 2026

**Tiny ONNX networks that solve ARC-AGI puzzles · 用极小的 ONNX 网络求解 ARC-AGI 谜题**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.21-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.26-FF6F00)](https://onnxruntime.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-NeuroGolf%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/neurogolf-2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/zikuanqi/NeuroGolf)](https://github.com/zikuanqi/NeuroGolf/commits/main)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![Tasks Solved](https://img.shields.io/badge/tasks_solved-74%2F400-blue)](networks/)

</div>

---

## 📖 Overview · 项目概览

> **EN** — Solution framework for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) on Kaggle. The competition asks for ONNX networks that solve ARC-AGI tasks while minimizing `memory_bytes + parameter_count`. A task that passes **every** train / test / arc-gen example earns `max(1, 25 − ln(memory + params))` points; one that fails any example scores zero.

> **中文** —— Kaggle [2026 NeuroGolf 锦标赛](https://www.kaggle.com/competitions/neurogolf-2026) 的求解框架。比赛要求为每个 ARC-AGI 任务提交一个 ONNX 网络，在通过**全部** train / test / arc-gen 样例的前提下，最小化 `内存字节数 + 参数数量`。单任务得分为 `max(1, 25 − ln(memory + params))`，任一样例失败即得零分。

---

## 📊 Results · 当前成绩

| Submission · 提交 | Solvers added · 新增求解器 | Tasks pass · 通过任务 | Score · 公开分数 |
|---|---|---|---|
| v1 | identity, zero, single-color, remap | 4 | **81.57** |
| v2 | + transpose | 6 | **131.57** |
| v3 | + marker-crop (opset 11) | 7 | **149.93** |
| v4 | + static-crop, kron-scale, resize-scale, shape-aware flip/rot180 | 15 | **276.86** |
| v5 | + rot90-ccw (transpose ∘ flip-v) | 16 | **290.62** |
| v6 | + bbox-strip (Slice + Sub + ReduceMax/ArgMax + Gather + Less + Mul) | 17 | **303.98** |
| v7 | + shift (Slice+Concat+Pad) + tile-h (shape-aware Mod-Gather) | 19 | ~337 (pending) |
| v8 | + palindrome H / V / 2D (Where-based mirror-concat) | 26 | ~435 (pending) |
| v9 | + majority-fill (ReduceSum→TopK→Where→OneHot) | 27 | ~453 (pending) |
| v10 | + variable-kron (Div+Min+Mul masked Resize-by-N) | 29 | **480.56** |
| v11 | + conv 1×1 / 3×3 / 5×5 masked (least-squares + bias) | 31 | ~509 (pending) |
| v12 | + bbox-color-extract, split-and, scale-detector, variable-shift, gravity-right | 35 | **567.92** |
| v13 | + flood-fill-enclosure (border BFS unrolled with Pad+Conv+Max) | 36 | **578.92** |
| v14 | + column-label (ArgMax pairwise ranking for per-column top-row labeling) | 37 | **592.92** |
| v15 | + repeat-top-rows (runtime period detection P=2/3/4 with weighted candidate mix) | 39 | ~605 (pending) |
| v16 | + block-mask (N×N→N²×N² masked tiling + channel-0 recovery with Tile×inv_mask) | 40 | ~616 (pending) |
| v17 | + gravity-right-diag (per-channel center-of-mass diagonal shift with stay/shift mask) | 41 | ~628 (pending) |
| v18 | + gravity-down (size-aware per-channel column sink via content-mask height + ramp threshold) | 45 | **697.88** |
| v19 | + self-fractal (Tile × Kron-upscaled selector mask; fixed / most-frequent colour) | 47 | **726.38** |
| v20 | + rot-tile (four-quadrant rotational tiling I / rot90 / rot180 / rot270) | 48 | **742.85** |
| v21 | + gravity-up (stable column rise via TopK sort-key + GatherElements, multi-colour columns) | 49 | **755.96** |
| v22 | + periodic-fill (period in-painting; static period enumeration + log-doubling shift-Max propagation) | 53 | **786.58** |
| v23 | + split-logic (two-half boolean combine: and/or/xor/nor over L-R or T-B split, ± divider) | 62 | ~924 (pending) |
| v24 | + connect-dots (fill the gap between the first and last same-colour dot per row/column) | 64 | ~949 (pending) |
| v25 | + mirror-complete (restore an erased half as the shape-aware mirror of the present half) | 66 | **975.27** |
| v26 | + denoise (remove isolated single cells via depthwise 3×3 same-colour neighbour count) | 67 | ~988 (pending) |
| v27 | + connect-fill (fill the gap between aligned same-colour dots with a fixed colour) | 70 | ~1024 (pending) |
| v28 | + drop-into-wall (colour-1 cells fall into a full colour-5 wall row in their column) | 71 | ~1037 (pending) |
| v29 | + color-bbox-fill (replace each colour's scattered cells with its solid bounding box) | 72 | ~1051 (pending) |
| v30 | + nearest-wall (recolour each marker with the colour of the nearer of two facing walls) | 73 | ~1064 (pending) |
| v31 | + cross-laser (each marker fires a full row + column "plus"; differing-colour crossings become 2) | 74 | **1076.16** |

---

## 🚀 Quick Start · 快速开始

```bash
# 1. Install deps · 安装依赖
pip install -r requirements.txt

# 2. Download competition data · 下载竞赛数据
python scripts/download.py

# 3. Build networks for all 400 tasks · 为全部 400 个任务构建网络
python scripts/build_all.py

# 4. Package the .onnx files · 打包 .onnx 文件
python scripts/package_submission.py

# 5. Submit to Kaggle · 提交到 Kaggle
python scripts/submit.py submissions/submission.zip "describe your run"
```

> Kaggle requires the submission file be named **exactly** `submission.zip`.
> Kaggle 要求提交文件必须命名为 `submission.zip`。

---

## 🧩 Solvers · 求解器一览

Each solver is a callable `(task: dict) → Optional[onnx.ModelProto]`. The pipeline runs every solver, verifies each candidate against the official scorer, and keeps the highest-scoring network.

每个求解器是一个 `(task: dict) → Optional[onnx.ModelProto]` 调用，pipeline 会运行所有求解器，用官方评分器验证每个候选网络，保留得分最高的网络。

| Solver · 求解器 | Pattern · 识别模式 | ONNX ops · 算子 | Params · 参数 |
|---|---|---|---|
| `solve_identity` | output ≡ input · 输出等于输入 | `Identity` | 0 |
| `solve_remap` | per-pixel color lookup · 像素级颜色映射 | 1×1 `Conv` | 100 |
| `solve_single_color` | uniform fill, same shape · 同尺寸纯色填充 | 1×1 `Conv` | 100 |
| `solve_transpose` | output is input transposed · 输出为输入的转置 | `Transpose` | 0 |
| `solve_static_crop` | fixed-offset, fixed-size subrect of input · 定长定位裁剪 | `Slice` + `Pad` | ~14 |
| `solve_kron_scale` | N×N pixel expansion, constant input shape · 同尺寸像素块复制 | 2× `Gather` + `Pad` | ~40 |
| `solve_resize_scale` | N× nearest upscale, variable input shape · 变尺寸 N 倍最近邻放大 | `Slice` + `Resize` | ~24 |
| `solve_marker_crop` | crop a fixed window around a unique marker pixel · 围绕唯一标记像素的定长裁剪 | `Slice` → `ReduceSum` → `ArgMax` → `Slice` → `Pad` | ~27 |
| `solve_flip_h_aware` / `solve_flip_v_aware` | content-aware horizontal / vertical flip · 形状感知水平 / 垂直翻转 | `ReduceSum` + `ReduceMax` + `Mul`(arange) + `Sub` + `Clip` + `Cast` + `Gather` + `Mul` | ~37 |
| `solve_rot180_aware` | two stacked shape-aware flips · 双轴形状感知翻转 | flip-h ∘ flip-v | ~74 |
| `solve_rot90_ccw_aware` / `solve_rot90_cw_aware` | shape-aware 90° rotation · 形状感知 90° 旋转 | `Transpose` ∘ flip-v | ~37 |
| `solve_bbox_strip` | crop input to the bbox of non-background cells · 非背景外接矩形提取 | `ReduceSum` + `Sub` + row/col `ArgMax` + `Gather` + `Less` + `Mul` | ~76 |
| `solve_shift` | constant-shape translation with color-0 fill on the gap · 定形平移并以色 0 填充 | `Slice` + `Concat`(with init fill) + `Pad` | ~50 |
| `solve_tile_h` | horizontal tile-N with variable input width · 变宽 N 倍水平复刻 | `ReduceSum` + `ReduceMax` + `Mod` + `Gather` + `Less` + `Mul` | ~67 |
| `solve_palindrome_h` / `solve_palindrome_v` | input mirror-concatenated to its right / bottom edge · 镜像拼接 | shape-aware `Where` + `Gather` + `Less` mask | ~68 |
| `solve_palindrome_2d` | four-quadrant 2D mirror · 四象限二维镜像 | palindrome-h ∘ palindrome-v | ~133 |
| `solve_majority_fill` | constant-shape rectangle filled with the majority non-bg color · 常尺寸主色填充 | `ReduceSum` + `TopK` + `Greater` + `And` + `Where` + `OneHot` + `Mul` + `Pad` | ~32 |
| `solve_variable_kron` | scale by N where N is `count(non-zero)` or `count(distinct colors)` · 变 N 倍 Kronecker（N 来自输入特征）| `ReduceSum` + `Cast` + `Div` + `Min` (float) + `Gather` ×2 + `Less` + `Mul` | ~138 |
| `solve_bbox_color_extract` | crop input to bbox of majority/rarest color · 按主色/稀有色外接矩形裁剪 | `ReduceSum` + `ArgMax`/`ArgMin` + `OneHot` + `Mul` + `Cast` + `Gather` + `Mod` + `Less` | ~66 |
| `solve_split_and` | split input along a color-5 vertical separator, AND the two halves · 沿颜色5分隔列分割并对两半做AND | `Slice` + `Pad` + `Sub` + `Mul` + `Less` + `Cast` + `And` | ~8129 |
| `solve_split_logic` | split into two halves (L-R or T-B, ± divider) and combine with and/or/xor/nor/nand · 切两半并做布尔运算 | `Slice` + `ReduceSum` + `Mul`/`Max`/`Sub` (boolean) + `Pad` + `Concat` | ~20 |
| `solve_connect_dots` | fill the span between the first and last same-colour dot in each row/column · 连接每行/列首尾同色点之间的间隔 | `Slice` + `CumSum` + `Gather` (reverse) + `Greater` + `And` + `Concat` | ~37 |
| `solve_mirror_complete` | restore an erased half as the shape-aware mirror of the present half · 用形状感知镜像补全被擦除的另一半 | shape-aware flip (`ReduceMax` + `Sub` + `Clip` + `Gather`) + `Mul` + `Add` merge | ~42 |
| `solve_denoise` | remove isolated single cells (no same-colour 8-neighbour) · 删除无同色邻居的孤立点 | depthwise 3×3 `Conv` (hollow kernel) + `Greater` + `Mul` + `Concat` | ~81 |
| `solve_connect_fill` | fill the gap between aligned same-colour dots with one fixed colour · 用固定色连接同行/列的成对同色点 | `CumSum` ×4 (h/v prefix+suffix) + `Greater` + `And` + `Or` + `ReduceMax` + `Concat` | ~64 |
| `solve_drop_into_wall` | colour-1 cells fall into a full colour-5 wall row in their column · 颜色1沿列落入整行的颜色5墙 | `Gather` + `ReduceSum`/`ReduceMax` (wall + column detect) + `Mul` (outer product) + `Relu` + `Concat` | ~12 |
| `solve_color_bbox_fill` | replace each colour's scattered cells with its solid bounding box · 用实心外接矩形填充每个颜色的散点 | `ReduceMax` (row/col span) + `CumSum` ×4 + `Greater` + `And` + `Mul` (outer) + `Concat` | ~80 |
| `solve_nearest_wall` | recolour each marker with the colour of the nearer of two facing walls (L/R columns or T/B rows) · 将每个标记重涂为左右列或上下行两堵相对墙中较近一堵的颜色 | `ReduceSum`/`ReduceMax` (content, W/H, wall colour) + `Gather` (far wall by index) + `Less`/`Greater` half-masks + `Cast` + `Mul` + `Add` blend | ~75 |
| `solve_cross_laser` | each marker fires a full row + column "plus"; cells where one marker's row meets a different marker's column become colour 2 · 每个标记射出整行整列十字，不同颜色十字交叉处变为颜色 2 | `ReduceMax` (per-row/col colour) + `ReduceSum` (same-colour test) + `Sub`/`Mul`/`Add` mask blend | ~31 |
| `solve_scale_detector` | N× nearest-neighbor upscale or downscale · N 倍最近邻放大或缩小 | `Slice` + `Resize` | ~24 |
| `solve_variable_shift` | shift by fixed offset with zero-fill, variable input shape · 定偏移量平移零填充，变输入尺寸 | `Slice` + `Pad` + `Concat` | ~50 |
| `solve_gravity_right` | each row's cells slide right until blocked · 重力向右：每行格子右移直到被挡住 | `ReduceSum` + `CumSum` + `Where` + `Mul` | ~94 |
| `solve_gravity_down` | each column's cells fall to the bottom edge · 重力向下：每列格子下落堆积到底部 | `ReduceSum` + `ReduceMax` (content-mask height) + `Less` + `Mul` + `Slice` + `Concat` | ~66 |
| `solve_gravity_up` | each column's cells rise to the top, preserving order (multi-colour) · 重力向上：每列保序上浮，支持多色列 | `ReduceSum` + sort-key + `TopK` + `Tile` + `GatherElements` | ~42 |
| `solve_periodic_fill` | restore a periodic tiling from a partially-erased grid · 从被部分擦除的网格恢复周期平铺 | per-period `Pad`+`Slice`+`Max` (log-doubling) + `Where` period select + bbox clip | ~9k |
| `solve_self_fractal` | N×N input tiles into an N²×N² self-similar fractal, keyed by a selector colour · N×N 输入自相似放大为 N²×N²，按选择色决定哪些块保留 | `Slice` + `Tile` + `Gather` ×2 (Kron mask) + `ArgMax` + `Mul` + `Pad` + `Concat` | ~57 |
| `solve_rot_tile` | N×N square tiles into 2N×2N as four rotations (I / rot90 / rot180 / rot270) · N×N 方阵拼为 2N×2N 的四象限旋转 | `Slice` + `Transpose` + `Gather` ×4 (index-reverse) + `Concat` ×3 + `Pad` | ~23 |
| `solve_gravity_right_diag` | per-channel diagonal slide: compute center-of-mass, shift non-rightmost cells toward it · 按通道对角线滑移：计算质心，将非最右像素移向质心方向 | `ReduceSum` + `ReduceMax` + `ArgMax` + `Mul` + `Slice` + `Concat` + `Min` + `Sub` | 150 |
| `solve_conv3x3` | least-squares fit of a 3×3 conv (no bias) · 无偏置 3×3 卷积拟合 | 3×3 `Conv` | 900 |
| `solve_conv1x1_masked` / `solve_conv3x3_masked` / `solve_conv5x5_masked` | K×K conv + bias, masked to non-padding cells · K×K 卷积带偏置和非填充掩码 | `Conv` + `ReduceSum` + `Mul` | 100 / 910 / 2510 |
| `solve_zero` | output is empty grid · 输出为空网格 | `Sub` | 0 |
| `solve_flood_fill_enclosure` | fill color-0 cells fully enclosed by a single source color · 填充被单一颜色完全包围的色 0 区域 | `ArgMax` + `Equal` + 58×(`Pad`+`Conv`+`Greater`+`Max`) border flood-fill + `Gather` + `Sub` + `Slice` + `Concat` | ~1.8k |
| `solve_block_mask` | N×N→N²×N²: tile input row/col-wise, mask[0-block]→zero, channel-0 recovery via Tile×inv_mask · 块掩码平铺：按块复制输入，零块掩蔽，通道 0 恢复 | `Slice` + `ReduceSum` + `Less` + `Tile` + `Mul` + `Add` | 78 |

---

## 🗂️ Project Layout · 项目结构

```
NeuroGolf/
├── src/neurogolf/
│   ├── grids.py          # one-hot conversions      · 独热编码转换
│   ├── onnx_ops.py       # ONNX graph helpers       · ONNX 图构建辅助
│   ├── verify.py         # clean-room scorer        · 评分器独立实现
│   ├── pipeline.py       # solver orchestration     · 求解器调度
│   └── solvers/          # per-family solvers       · 各类求解器
├── scripts/
│   ├── download.py             # pull data via Kaggle CLI · 下载比赛数据
│   ├── build_all.py            # run pipeline over tasks  · 跑全量任务
│   ├── package_submission.py   # zip the .onnx files      · 打包提交
│   └── submit.py               # post to Kaggle           · 上传 Kaggle
├── networks/             # generated .onnx files    · 生成的网络
├── submissions/          # packaged submission.zip  · 打包的提交文件
└── tests/                # pytest cases             · 单元测试
```

---

## 📏 Scoring & Constraints · 评分与约束

**Per-task score · 单任务得分**

```
points = max(1, 25 − ln(memory_bytes + params))   # if all examples pass · 全样例通过时
       = 0                                         # otherwise · 否则
```

**Constraints · 约束条件**

| Aspect · 项目 | Requirement · 要求 |
|---|---|
| Input / output tensor · 输入输出张量 | `(1, 10, 30, 30)` float32, one-hot · 独热编码 |
| Tensor names · 张量名 | `"input"` / `"output"` |
| Output decoding · 输出解码 | thresholded at `> 0.0` · 以 `> 0.0` 阈值化 |
| File size · 文件大小 | ≤ 1.44 MB per `.onnx` · 单文件不超过 1.44 MB |
| Banned ops · 禁用算子 | `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`, `COMPRESS`, any `Sequence*`, graph-typed attributes · 任何 `Sequence*`、子图属性 |
| Shapes · 形状 | statically inferable; declare `value_info` when shape inference can't see through dynamic ops · 必须静态可推断；动态算子需显式声明 `value_info` |
| ARC-gen grids · arc-gen 样例 | grids larger than 30×30 are skipped by the scorer · 超过 30×30 的样例由评分器自动跳过 |

---

## 🔬 Tests · 单元测试

```bash
python -m pytest tests/ -q
```

20 cases cover the one-hot round-trip contract and per-family solver
shape contracts (Identity / Conv1×1 / Transpose / Slice+Pad / Gather+Pad /
Resize / shape-aware flip / rot / bbox / shift / tile-h).

20 个测试覆盖独热编码的往返一致性，以及各求解器输出图的形状契约
（Identity / Conv1×1 / Transpose / Slice+Pad / Gather+Pad / Resize /
形状感知翻转 / 旋转 / 外接矩形 / 平移 / 横向复刻）。

---

## 🛣️ Roadmap · 后续计划

- [x] Bounding-box extraction (ReduceSum + ArgMax row/col projections) · 外接矩形提取（行列投影 + ArgMax）
- [x] Shape-aware flip / rotate that handles top-left alignment · 处理左上角对齐的形状感知翻转 / 旋转
- [x] Scale-by-N tasks via static `Resize` or `Tile` · 用 `Resize` / `Tile` 实现 N 倍缩放任务
- [x] Constant-shape translation (shift with color-0 gap) · 定形平移并填补色 0 间隙
- [x] Shape-aware horizontal tile-N (Mod-Gather-mask pipeline) · 形状感知横向 N 倍复刻
- [ ] Generalize marker-crop to handle multi-pixel markers and per-color crop rules · 通用化标记裁剪以支持多像素标记
- [ ] Detect "input contains a filled rectangle of color C" tasks (57 candidates) · 识别"输入含纯色矩形"类任务
- [ ] Reduce double `(1,10,30,30)` intermediates in rot180 and flip via sparse / fused Gather · 借助稀疏或融合 `Gather` 降低 rot180 / flip 的双 `(1,10,30,30)` 中间张量内存
- [ ] Cache ONNX Runtime traces to speed up iteration · 缓存 ONNX Runtime trace 加速迭代

---

## 📜 License · 许可证

MIT © 2026 [zikuanqi](https://github.com/zikuanqi) — see [LICENSE](LICENSE) · 详见 [LICENSE](LICENSE)
