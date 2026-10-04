<div align="center">

# 动漫角色数据集构建器 · Anime Character Dataset Builder

**视频 → 抽帧 → 角色检测 → 裁剪 → 去重 → 质量过滤 → 特征 & 聚类 → 人工标注 → 分层划分 → ImageFolder 数据集**

纯 CPU 可跑通全流程 · 检测器/特征提取器/聚类器均为可替换插件 · 断点续跑 · 幂等

</div>

---

## ⚠️ 版权与合规声明（请先读这一段）

* 本项目**仅用于学习、研究与个人技术验证**。使用者须自行确保对所处理的视频素材拥有合法权利。
* 仓库**不包含任何模型权重**（YOLO / timm / CLIP 等一律不内置），也不包含任何动漫片段、截图或数据集。所有模型权重需使用者自行从官方渠道获取并遵守其各自许可证（见 [模型权重获取](#模型权重获取)）。
* **请勿**使用本项目构建、传播或商用任何侵犯著作权的内容；**请勿**将产出数据集公开分发，除非你拥有相应授权。
* 仓库自带的演示视频是**程序化绘制的合成画面**（`acdb demo-video`），不含任何第三方素材，可自由使用。
* 本项目以 MIT 许可证发布；第三方模型与库遵循各自许可证。

---

## 目录

- [它能做什么](#它能做什么)
- [三步上手](#三步上手)
- [硬性输出契约](#硬性输出契约)
- [模型权重获取](#模型权重获取)
- [配置说明](#配置说明)
- [全流程与单阶段运行](#全流程与单阶段运行)
- [中间产物一览](#中间产物一览)
- [可替换性：换检测器/特征提取器/聚类器](#可替换性换检测器特征提取器聚类器)
- [断点续跑、幂等与日志](#断点续跑幂等与日志)
- [测试](#测试)
- [复现演示（验收证据）](#复现演示验收证据)
- [设计决策](#设计决策)
- [目录结构](#目录结构)
- [常见问题](#常见问题)

---

## 它能做什么

```
data/input/**/*.mp4                （可嵌套子目录，任意数量）
        │
        ├─ 1 extract-frames   interval / scene / keyframe 三种抽帧策略，可组合
        ├─ 2 detect           YOLOv8 / YOLO11 / 动漫专用权重 / 无权重显著性检测器
        ├─ 3 crop             按比例外扩 + 方形补齐 + 背景保留开关 + pHash
        ├─ 4 dedup            pHash 汉明距离粗筛 + 特征余弦相似度精排
        ├─ 5 filter           Laplacian 方差 / 过曝欠曝 / 对比度 / 色彩度 → quality_score
        ├─ 6 cluster          特征提取 → 降维 → HDBSCAN/KMeans → 代表图导出待人工核对
        ├─ 7 apply-labels     合并 cluster_labels.yaml（或重命名的文件夹）→ anime/character
        ├─ 8 split            按类别分层 8:1:1，同视频同角色不跨 split（防泄漏）
        └─ 9 export           ImageFolder + metadata.csv + classes.json + 每类统计
                │
                ▼
data/dataset/{train,val,test}/{anime}__{character}/*.jpg
```

## 三步上手

```bash
# 0) 安装（Python 3.10+，3.12 已完整验证；无需 GPU）
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                  # 仅核心依赖，纯 CPU 可跑

# 1) 放入视频（可嵌套；默认读取 data/input）
cp -r /path/to/your/anime_clips/* data/input/

# 2) 跑流水线
acdb run-all -c configs/default.yaml
#    → 会在「人工标注」处暂停，并打印需要你做的事
# 3) 按提示标注后继续
acdb apply-labels -c configs/default.yaml && acdb split -c configs/default.yaml && acdb export -c configs/default.yaml
#    或直接：acdb run-all -c configs/default.yaml --resume
```

**想先看效果、又不想下载任何权重？** 用自带演示（合成视频，全程离线，CPU 约 30 秒）：

```bash
acdb demo-video -c configs/demo.yaml          # 渲染 3 部动画 × 3 集合成视频 + 角色参考图
acdb run-all   -c configs/demo.yaml --yes     # 跑到人工标注处暂停
acdb demo-annotate -c configs/demo.yaml       # 模拟人工标注（按参考图做特征匹配）
acdb run-all   -c configs/demo.yaml --resume --yes
python scripts/check_output_contract.py -c configs/demo.yaml   # 独立校验输出契约
```

完整日志与产出统计见 **[docs/demo_run.md](docs/demo_run.md)**。

### 在 VS Code 里运行 / 调试

仓库自带 `.vscode/` 配置，打开文件夹即可按 <kbd>F5</kbd> 选配置运行（`.vscode/launch.json`，16 条）：

| 配置 | 作用 |
|---|---|
| `acdb: run-all（demo 配置，离线）` | 零权重跑完整流水线，到人工标注处暂停 |
| `acdb: run-all --resume（继续上次中断）` | 断点续跑 |
| `acdb: 单阶段（选择阶段 + 配置）` | F5 时弹出下拉框选阶段与配置文件 |
| `acdb: cluster（在聚类/导出评审图处断点）` | 想单步跟踪聚类逻辑时用 |
| `demo: 渲染合成视频 / 模拟人工标注` | 演示用的两个辅助命令 |
| `校验: check_output_contract.py` | 独立校验输出契约 |
| `pytest: 全部用例 / 当前文件 / 光标所在测试` | 带断点跑测试 |

另有两个组合任务（<kbd>Ctrl+Shift+P</kbd> → `Tasks: Run Task`）：
`demo: 完整离线流程`（渲染 → 跑 → 标注 → 导出 → 校验，五步顺序执行）与 `check: ruff + pytest`。

`launch.json` 里已统一设好三件事，所以一般不需要改：`cwd=${workspaceFolder}`（配置里的相对路径都相对项目根解析）、
`module=acdb`（等价于 `python -m acdb ...`）、`PYTHONPATH=${workspaceFolder}/src`（即使没做 `pip install -e .` 也能导入）。
需要交互/彩色输出的配置都用 `integratedTerminal`，断点调试请用集成终端而不是调试控制台。

> `settings.json` 默认解释器指向 `./.venv/Scripts/python.exe`；若你的环境在别处，
> 用 `Python: Select Interpreter` 选一次即可，所有配置会自动跟随。

---

## 硬性输出契约

### `data/metadata/metadata.csv`

固定 16 列、固定顺序、UTF-8（无 BOM）、`\n` 换行、每张图一行：

```
image_id,anime,character,split,video_path,video_id,frame_idx,timestamp_ms,
bbox_x1,bbox_y1,bbox_x2,bbox_y2,det_conf,cluster_id,quality_score,file_path
```

| 列 | 含义 |
|---|---|
| `image_id` | 稳定主键，等于裁剪文件名主干：`{video_id}_f{frame:06d}_b{bbox:02d}` |
| `anime` / `character` | 作品名 / 角色名，来自聚类标签 |
| `split` | `train` / `val` / `test` |
| `video_path` | 源视频路径，相对项目根目录（POSIX 分隔符） |
| `video_id` | 见下方生成规则 |
| `frame_idx` / `timestamp_ms` | 帧号（源视频内）与时间戳 |
| `bbox_*` | **最终裁剪框**在源帧坐标系中的位置（含外扩与方形补齐，可能为负值表示补边） |
| `det_conf` | 检测置信度 |
| `cluster_id` | 聚类编号，`-1` 表示噪声（HDBSCAN 未归属） |
| `quality_score` | 综合质量分 ∈ [0,1] |
| `file_path` | 导出图片路径，相对项目根目录 |

> `bbox_*` 记录的是**裁剪框**而不是原始检测框：这样单凭 metadata.csv 就能还原每张图的来源。
> 原始检测框保存在 `data/metadata/detections.csv`。

### `data/metadata/classes.json`

```json
{"classes": [{"id": 0, "anime": "demo_show_a", "character": "hikari"}, ...]}
```

类别按 `anime__character` 字典序编号，从 0 开始连续。

### 命名规范

| 阶段 | 模板 |
|---|---|
| 抽帧 | `{video_id}_f{frame_idx:06d}.jpg` |
| 裁剪 | `{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg` |
| 最终 | `{anime}__{character}__{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg` |

**`video_id` 生成规则**：`"{slug}_{hash}"`

* `slug`：文件名主干做 Unicode NFKD → ASCII 折叠 → 转小写 → 非 `[a-z0-9]` 全部替换为 `_` → 连续下划线合并为一个 → 截断到 `naming.video_id_slug_maxlen`（默认 40）。因此 **slug 中永远不会出现 `__`**，不会与最终文件名的 `anime__character` 分隔符冲突；纯中文名会退化为 `video`。
* `hash`：**视频相对输入目录的路径**（POSIX 分隔符、小写）做 BLAKE2b 摘要，取前 `naming.video_id_hash_len`（默认 8）位十六进制。
* 效果：同名文件在不同子目录下 id 不同；整体搬迁输入目录 id 不变；可在 README 之外独立复现。
* 形如 `con`、`nul` 等 Windows 保留名会加 `_v` 后缀。
* `bbox_idx` 为 2 位数字，因此**单帧最多 99 个框**（`detect.filter.max_per_frame` 默认 10，正常不会触及）。

---

## 模型权重获取

仓库不内置权重。按需二选一：

**A. 用官方 COCO 权重（最省事，检测「人」，对动漫角色召回一般）**

```yaml
detect:
  type: yolo_v8            # 或 yolo11
  params:
    model: yolov8n.pt      # 首次运行由 ultralytics 自动下载
```

**B. 用动漫角色专用权重（推荐，效果最好）**

下载一个单类别（character）的 YOLO 检查点到本地（例如 `models/anime_char_yolov8s.pt`），然后：

```yaml
detect:
  type: anime_yolo
  params:
    weights: models/anime_char_yolov8s.pt
    class_names: [character]
```

* 用分割权重（`-seg`）时选 `type: anime_yolo_seg`，掩码可用于裁剪阶段。
* 训练自己的权重：任何 YOLO 格式数据集都能用 ultralytics 训练；本项目只要求推理接口返回 bbox(+mask)。
* 权重来源请遵守其许可证；仓库的 `.gitignore` 已排除 `*.pt / *.onnx / *.safetensors`。

**C. 特征提取权重**

```yaml
features:
  type: timm
  params: {model_name: resnet18, pretrained: true}     # 首次运行自动下载
  # 或本地/离线：
  # params: {model_name: /path/to/local/model.safetensors, pretrained: false}
  # 或 ONNX：
  # type: onnx
  # params: {onnx_path: models/anime_embedding.onnx}
  # 或 CLIP：
  # type: clip
  # params: {model_name: openai/clip-vit-base-patch32}
  # 或完全不需要权重：
  # type: color_hist
```

离线机器上可先用 `huggingface-cli download` / `torch.hub` 预取权重，再把 `model_name` 指向本地路径。

---

## 配置说明

所有路径、阈值、模型名、比例都在 `configs/*.yaml`；CLI 只做覆盖（`--set a.b=c`），不重复定义参数。
**未知键会直接报错**，拼错阈值不会静默失效。

* `configs/default.yaml` —— 真实流水线（YOLO + timm + HDBSCAN），需要 `pip install -e ".[all]"`。
* `configs/demo.yaml` —— 演示配置，`base: default.yaml` 深合并，换成无需权重、纯 CPU 的实现，全程离线。

配置文件支持一层继承：顶层 `base: default.yaml` 会与当前文件深合并（当前文件优先）。

主要配置段：

| 段 | 关键项 |
|---|---|
| `project` | `seed`、`device`（`auto/cpu/cuda/cuda:0/mps`）、`max_videos` |
| `paths` | 输入/输出目录（相对路径基于项目根，即 cwd 或 `--root`） |
| `naming` | 三个命名模板、slug 长度、hash 长度 |
| `video` | 扩展名、是否递归、是否跳过隐藏文件 |
| `extract_frames` | `strategies`、`interval_seconds/frames`、`scene_threshold`、`keyframe_backend`、`max_frames_per_video`、`skip_static` |
| `detect` | `type`、`batch_size`、`params`（模型相关）、`filter`（置信度/最小边/宽高比/面积比/每帧上限/跨帧 NMS） |
| `crop` | `expand_ratio`、`square`、`keep_background`、`pad_mode`、`output_size` |
| `dedup` | `phash_hamming`、`cosine_threshold`、`keep`、`same_video_only`、`max_group_size` |
| `features` | `type`、`normalize`、`batch_size`、`params` |
| `quality` | 模糊/曝光/色彩度阈值、`min_score`、四项权重 |
| `cluster` | `type`、`params`、`reduce_dim`/`pca_components`、每簇代表图数量 |
| `split` | `ratios`、`seed`、`stratify`、`prevent_leakage`、`leakage_fallback`、`min_images_per_class` |
| `export` | `copy_mode`（copy/hardlink/symlink）、`overwrite`、`drop_unknown`、`min_images_per_class` |
| `logging` | 控制台/文件级别、轮转、保留份数 |

示例：

```bash
acdb detect -c configs/default.yaml --set detect.type=anime_yolo \
            --set detect.params.weights=models/anime_char.pt --set project.device=cuda
```

---

## 全流程与单阶段运行

每个阶段都是一个独立子命令，只读写约定目录；上游产物存在且参数未变就跳过。

```bash
acdb extract-frames -c configs/default.yaml      # 抽帧
acdb detect         -c configs/default.yaml      # 检测
acdb crop           -c configs/default.yaml      # 裁剪
acdb dedup          -c configs/default.yaml      # 去重
acdb filter         -c configs/default.yaml      # 质量过滤
acdb cluster        -c configs/default.yaml      # 特征 + 聚类 + 导出待核对图
#   ↓ 人工环节：编辑 data/clusters/cluster_labels.yaml，或重命名 cluster_003/ →
#     cluster_003__show__hero/ （保留 cluster_<id> 标记）
acdb apply-labels   -c configs/default.yaml      # 合并标注
acdb split          -c configs/default.yaml      # 分层划分
acdb export         -c configs/default.yaml      # 导出数据集
```

整条流水线：

```bash
acdb run-all -c configs/default.yaml                 # 从第一个未完成阶段开始
acdb run-all -c configs/default.yaml --resume        # 断点续跑
acdb run-all -c configs/default.yaml --from crop --to cluster
acdb run-all -c configs/default.yaml --force         # 忽略状态全部重跑
acdb run-all -c configs/default.yaml --dry-run       # 只算不写
acdb status  -c configs/default.yaml                 # 各阶段状态 / 是否新鲜 / 下一步命令
```

通用参数：`-c/--config`、`--root`、`--set`、`--force`、`--dry-run`、`-y/--yes`、`-v/--verbose`。

**抽帧策略**（可任意组合，取并集去重）：

| 策略 | 说明 | 依赖 |
|---|---|---|
| `interval` | 每 N 秒或每 N 帧取一帧，单次顺序解码 | 无 |
| `scene` | 镜头切换处抽帧 | PySceneDetect；未安装时用内置 HSV 直方图检测器（近似） |
| `keyframe` | 容器 I 帧 | PyAV；不可用时回退到 `scene` 并打印警告 |

**人工标注两种方式**（二选一，或混用；YAML 显式值优先于文件夹名，冲突会告警）：

```yaml
# data/clusters/cluster_labels.yaml
clusters:
  0: demo_show_a__hikari
  1: {anime: demo_show_a, character: kaito}
  2: unknown          # 归入 unknown/
```

```text
data/clusters/cluster_003/                        → 未标注
data/clusters/cluster_003__demo_show_b__yuki/     → 已标注
data/clusters/demo_show_b__yuki__cluster_003/     → 同上（两种写法都支持）
```

未标注的簇与 `cluster_id == -1` 的噪声一律归入 `unknown/unknown`，并在 `export.drop_unknown: true` 时被排除出最终数据集。

---

## 中间产物一览

| 文件 | 生产者 | 内容 |
|---|---|---|
| `data/frames/{video_id}/*.jpg` | extract-frames | 抽出的原始帧 |
| `data/metadata/frames.csv` | extract-frames | 帧清单（帧号/时间戳/策略/尺寸） |
| `data/metadata/detections.csv` | detect | **原始检测框**、置信度、类别名 |
| `data/crops/{video_id}/*.jpg` | crop | 裁剪图 |
| `data/metadata/crops.csv` | crop | 裁剪清单 + pHash + 实际裁剪框 |
| `data/metadata/dedup.csv` | dedup | crops + `kept` / `dedup_group` |
| `data/metadata/quality.csv` | filter | dedup 存活项 + 质量指标 + `quality_score` |
| `data/rejected/…` | filter | 被质量门淘汰的图（`move_rejected: true`） |
| `data/metadata/features.npz` | dedup/cluster | 特征向量缓存（按图片内容 + 提取器签名） |
| `data/clusters/cluster_{id:03d}/` | cluster | 每簇代表图 + `index.csv`（含来源视频/帧/bbox） |
| `data/clusters/noise/` | cluster | 噪声点代表图 |
| `data/metadata/clusters.csv` | cluster | quality 存活项 + `cluster_id` |
| `data/metadata/labeled.csv` | apply-labels | + `anime` / `character` |
| `data/metadata/split.csv` | split | + `split` |
| `data/metadata/metadata.csv` | export | **交付契约（16 列）** |
| `data/metadata/classes.json` | export | **交付契约（类别映射）** |
| `data/metadata/class_stats.csv` | export | 每类 × 每 split 样本数 |
| `data/metadata/config_resolved.yaml` | runner | 本次运行实际生效的完整配置（可复现性） |

> 每个阶段的 CSV 都保留它「考虑过」的全部行，用 `kept` 列表达判定结果；下一个阶段只取 `kept=True` 的行。

---

## 可替换性：换检测器/特征提取器/聚类器

三个抽象基类：`BaseDetector`、`BaseFeatureExtractor`、`BaseClusterer`。**换实现只改配置，不改代码。**

```yaml
detect:    {type: anime_yolo_seg}    # yolo_v8 | yolo11 | anime_yolo | anime_yolo_seg | saliency
features:  {type: clip}              # timm | onnx | clip | color_hist
cluster:   {type: kmeans}            # hdbscan | kmeans | agglomerative
```

自己写一个只需要三步：

```python
from acdb.detectors.base import BaseDetector, Detection

class MyDetector(BaseDetector):
    name = "my_detector"
    def detect(self, image):                     # image: BGR uint8 (H,W,3)
        return [Detection(bbox=(x1, y1, x2, y2), confidence=0.9, class_name="character")]

from acdb.detectors import DETECTOR_TYPES
DETECTOR_TYPES["my_detector"] = DetectorSpec(MyDetector, {"threshold": 0.5}, "我的检测器")
```

之后 `detect.type: my_detector` 即可。特征提取器（`extract()` 返回 `(N, D)` float32）与聚类器（`fit_predict()` 返回 `ClusterResult`，`-1` 表示噪声）同理，分别在 `EXTRACTOR_TYPES` / `CLUSTERER_TYPES` 注册。

---

## 断点续跑、幂等与日志

* **状态文件**：`data/metadata/.state/{stage}.json`，记录 `signature`（本阶段参数 + 全部上游签名的链式哈希）、`items`（已处理输入清单）、计数与耗时。
* **参数变化**：任一阶段参数（或上游参数）变化 → 该阶段及其下游签名不匹配 → 运行时给出原因并询问是否重跑，`-y` 直接同意，`--force` 强制全跑。编辑 `cluster_labels.yaml` 或重命名核对文件夹同样会触发 `apply-labels` 及下游重跑（标签内容也参与哈希）。
* **崩溃与硬杀**：状态先写 `running` 再干活，输出先写临时文件再 `os.replace` 改名，因此**不会产生半成品文件**；已处理的输入会按批（每 25 项）落盘，`acdb run-all --resume` 从第一个未完成/已失效阶段继续，阶段内已完成的文件不会重复处理。
* **日志**：loguru 分级，控制台 + `logs/pipeline_{date}.log`（轮转 10MB，保留 14 份）。每阶段结束打印固定摘要行：

  ```
  [dedup] summary | processed=72 filtered=40 produced=32 elapsed=0.21s candidate_pairs=92 confirmed_pairs=90 groups=18
  ```

  即「处理数 / 过滤数 / 产出数 / 耗时」，后附阶段特有指标。

---

## 测试

```bash
pip install -e ".[dev]"
pytest                       # 87 个用例
pytest -m "not slow"         # 跳过端到端（端到端会现场渲染视频并跑完整流水线）
```

覆盖：命名与 `video_id` 规则、`metadata.csv` 契约与校验、classes.json、标签解析（YAML/文件夹/冲突优先级）、去重（哈希、成对距离、候选门、**贪婪分组不链式合并**）、划分（比例、每类覆盖、**同视频不跨 split**）、质量分与各道门、几何（外扩/方形/边界补齐）、配置加载与校验、状态机（签名链、失效、续跑点）、以及一个 9 阶段端到端冒烟测试（含中断后按文件续跑）。

---

## 复现演示（验收证据）

`docs/demo_run.md` 记录了在一台无 GPU 的 Windows 机器上、用**合成视频**跑完整流水线的全过程：命令、每阶段摘要、耗时、最终统计与契约校验结果。全部过程**不下载任何模型权重、不使用任何受版权保护的素材**：9 段 12 秒视频 → 108 帧 → 216 张裁剪 → 95 张去重与质检后存活 → 7 个簇 → 6 个角色 → **90 张图片的 8:1:1 数据集**，总耗时约 28 秒。

---

## 设计决策

规范未明确的地方，以下是选择与理由。

1. **包结构用 `src/acdb/{pipeline,detectors,extractors,clustering,io,utils}`**
   规范给的 `src/{pipeline,...}` 会把 `src` 变成包名，既无法 `pip install` 也容易与同名第三方包冲突。加一层 `acdb` 命名空间后仍是 src-layout，`pip install -e .` 与 `python -m acdb` 都可用。

2. **元数据用 CSV 表而不是内存对象串场**
   每个阶段读写一个可直接用 Excel/pandas 检查的 CSV，出问题时能一眼定位是哪一步错了；`feature.npz` 单独存放向量。所有写入都是「先写 `*.tmp` 再 `os.replace`」，硬杀不会留下半截文件。

3. **`metadata.csv` 的 `bbox_*` 记录最终裁剪框，而非原始检测框**
   数据集里的图就是裁剪后的图，记录裁剪框才能让每张图单凭 `metadata.csv` 复现。原始检测框完整保留在 `detections.csv`，两者互不丢失。

4. **`bbox_idx` 不进 `metadata.csv`**
   契约规定列顺序固定，不允许多一列。`bbox_idx` 保存在各中间表与文件名中，因此 `image_id`（= 裁剪文件名主干）足以还原它。约束：单帧框数 ≤ 99。

5. **路径以「相对项目根的 POSIX 路径」写入清单**
   绝对路径会让清单换台机器就失效；相对路径让数据集可整体搬迁，Windows/Linux 一致。

6. **去重用「贪婪代表分组」而不是连通分量**
   pHash 相似关系不满足传递性：A≈B、B≈C 但 A≁C。连通分量会把「同一角色连续几秒的姿势」串成一个巨组，一次性删掉整个角色的多样性。改为按质量排序选代表、只吸收**与代表本人**同时满足两道阈值的成员，组内直径就被阈值本身限住了。`dedup.max_group_size` 作为安全阀，超限时多余成员一律保留并告警，绝不因一次错误合并清空一个角色。

7. **两道门的顺序是「粗筛 pHash → 精排余弦」**
   pHash 只看结构不看颜色，两个配色不同但剪影相近的角色会被误判；因此 pHash 只用来把候选对从 O(N²) 降到少数几对，最终判定必须同时满足余弦阈值。汉明距离用 `bits @ (1-bits).T + (1-bits) @ bits.T` 的 BLAS 矩阵乘精确算出，按行分块控制内存。

8. **`quality_score` 是四项归一化分量的加权平均；`dedup.keep: best_quality` 复用它**
   各分量映射到 [0,1] 时使用固定参考值（`log1p(Laplacian 方差)/log1p(1000)`、`std/64`、色彩度/60、曝光取最差一项），因此分数跨视频可比。`dedup` 在阶段顺序上早于 `filter`，所以它按需现场计算该分数来决定保留谁，而不是把两个阶段耦合起来。

9. **聚类前默认 PCA 降维**
   余弦/欧氏距离在高维（512–1024 维 embedding）下区分度急剧下降。默认降到 32 维（自动裁剪到 `min(n-1, d)`），使用确定性 SVD，保证同数据同结果。`reduce_dim: none` 可关闭。

10. **HDBSCAN 优先，噪声显式保留**
    它自己决定簇数，且不会把离群裁剪硬塞进某个角色。`cluster_id == -1` 一路保留到 `apply-labels`，最终归入 `unknown/`，可被人工复核。`noise_as_unknown: false` 时改为并入最近簇质心（可选行为，非默认）。

11. **划分的分配单元是 `(类别, video_id)`，并为「一个角色只出现在一个视频里」提供显式兜底**
    严格防泄漏时，若某角色只来自一个视频，则整个类别只能落进一个 split —— val/test 会空掉，数据集不可用。默认 `split.leakage_fallback: image` 会把**这类**降级为按图分配并打印警告（泄漏范围明确可见）；想严格保守就设成 `train_only`。同一视频的同一角色默认绝不跨 split。

12. **每个类别独立按比例分配**
    全局贪心会让小类别整体被推给 train。按类别分别分配，并对含 ≥3 个单元的类别强制 val/test 各至少 1 个单元，保证每个类在每个 split 都有样本（比例仍尽量逼近 8:1:1）。

13. **`crop.keep_background: false` 用「裁剪框边界环显著性」近似抠背景**
    纯 bbox 检测器没有掩码，而把掩码落盘再读回会显著增加复杂度与磁盘占用。默认实现对任意检测器都可用，代价是近似而非精细抠图；需要精细分割时把分割模型接到 `BaseDetector`（返回 `mask`）并扩展裁剪阶段即可，接口已预留。

14. **`crop.square: true` 默认开启**
    方形裁剪让所有下游（timm/CLIP 的 resize、直方图特征的空间网格）行为一致，避免宽高比差异被当成身份差异。

15. **默认 `extract_frames.strategies: [interval, scene]`，`skip_static: true`**
    固定间隔保证时间覆盖，镜头切换保证不遗漏场景变化；内容完全相同的相邻帧直接丢掉，省下后面所有阶段的开销。

16. **PySceneDetect / PyAV / ultralytics / timm / onnxruntime 全部可选**
    缺哪个都有明确降级路径（场景切换→内置直方图检测器；I 帧→场景切换；YOLO/嵌入→无权重实现），并打印原因。这让「干净环境三步复现」成立，也让依赖体积可控。

17. **合成演示视频 + 模拟标注器**
    没有模型权重、也不该分发受版权保护的素材，所以演示用程序化绘制的六角色合成视频（`acdb demo-video`），并额外渲染每个角色的「全身/头部」参考图；`acdb demo-annotate` 用**与流水线相同的特征提取器**把每个簇的图片与参考图做余弦匹配来模拟人工标注。它只做人工会做的事（看图 → 认人 → 写标签），匹配不上的簇老实留 `unknown`。

18. **`run-all` 在人工标注处主动暂停**
    首次运行时所有簇都是 `unknown`，硬跑到底只会在 `export` 报错「没有东西可导出」。改为在 `apply-labels` 前检测「一个簇都没标注」并打印横幅后停下，`--resume` 继续，体验上等价于 `git rebase --continue`。

19. **单阶段命令不接受 `--resume`**
    每个单阶段命令本来就会读状态：参数未变且已完成 → 跳过；被中断 → 只处理未完成的输入。`--resume` 只在 `run-all` 上有意义（决定从哪个阶段开始）。

20. **配置未知键直接报错**
    阈值拼错却静默使用默认值，是数据管线里最难查的一类 bug。宁可启动即失败。

---

## 目录结构

```
anime-character-dataset-builder/
├── configs/
│   ├── default.yaml              # 真实流水线配置（YOLO + timm + HDBSCAN）
│   └── demo.yaml                 # 离线演示配置（继承 default.yaml）
├── src/acdb/
│   ├── cli.py                    # Typer 命令行（每阶段一个子命令）
│   ├── config.py                 # 类型化配置加载/校验/哈希
│   ├── errors.py
│   ├── demo.py                   # 合成视频生成 + 模拟人工标注（仅演示用）
│   ├── detectors/                # base / filters / yolo / saliency / 注册表
│   ├── extractors/               # base / timm / onnx / clip / color_hist / 缓存 / 注册表
│   ├── clustering/               # base / hdbscan / kmeans+agglomerative / 降维 / 注册表
│   ├── io/                       # tables（契约与读写）/ labels（人工标注）
│   ├── pipeline/
│   │   ├── runner.py             # 编排、续跑、失效提示、配置留档
│   │   ├── stage.py              # 阶段基类：状态、计时、摘要
│   │   ├── context.py            # 阶段上下文与崩溃安全进度
│   │   ├── scoring.py            # 质量评分（filter 与 dedup 共用）
│   │   └── stages/               # 九个阶段，一阶段一文件
│   └── utils/                    # naming / image / video / state / logging / device / atomic
├── scripts/
│   ├── make_demo_video.py        # 生成演示视频
│   ├── make_contact_sheet.py     # 每类样本联络表（人工快速复核）
│   └── check_output_contract.py  # 独立校验输出契约（可用于 CI）
├── .vscode/                      # F5 运行/调试配置、任务、推荐扩展
├── tests/                        # 87 个 pytest 用例
├── docs/demo_run.md              # 演示运行记录（验收证据）
├── pyproject.toml / requirements.txt
└── README.md
```

## 常见问题

**Q: `acdb detect` 报「no detection survived the filters」。**
权重类别不对（COCO 权重只会在 `classes: [0]` 上找人），或过滤太严。先放宽 `detect.params.conf_threshold` 与 `detect.filter.min_side/min_area_ratio`，或用 `--set detect.type=saliency` 验证链路本身是否通。

**Q: `acdb export` 报「nothing left to export」。**
所有簇仍是 `unknown`。编辑 `data/clusters/cluster_labels.yaml`（或重命名核对文件夹）后重跑 `apply-labels → split → export`。

**Q: val/test 是空的。**
你的每个角色只出自一个视频。看 `split.leakage_fallback`（默认 `image` 会降级分配并告警）；要严格防泄漏就准备同一角色出现在多个视频里的素材。

**Q: Windows 上路径含中文/日文读不出图。**
图像读写已走 `numpy.fromfile + cv2.imdecode`（支持任意路径）；视频解码受 OpenCV 限制，**建议视频路径为 ASCII**，非 ASCII 时请先重命名或用 `--set paths.input_dir` 指向纯 ASCII 目录。

**Q: 想只跑一小部分试试。**
`--set project.max_videos=1 --set extract_frames.max_frames_per_video=20`。

**Q: 磁盘不够。**
`data/frames` 是最大的一块，抽帧后可删（后续阶段只用 `data/crops`）；`export.copy_mode: hardlink` 可以不再复制一份 crops。
