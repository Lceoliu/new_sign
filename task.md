项目名称：Pose Tokenizer (VQ/RVQ) for Skeleton Sequences  
目标交付物：

1.  `encode()`：输入 pose 序列，输出离散 token 序列（含下采样后的长度 L、token ids、可选每层 RVQ code ids）
2.  `decode()`：输入 token 序列，输出重建 pose 序列
3.  训练脚本：无监督训练 tokenizer（只用 pose 数据，无需文本）
4.  评估脚本：重建误差、动力学误差、骨长误差、码本使用率、token 长度统计
5.  可复现实验：下采样率、码本大小、RVQ 层数、loss 权重的 ablation 入口
6.  开发者文档：数据格式、模块接口、运行命令、常见问题排查

---

一、代码仓库结构（必须按这个拆，便于测试与替换）

- `configs/`
  - `tokenizer_base.yaml`（默认可跑）
  - `experiments/`（ablation 用配置）

- `pose_tokenizer/`
  - `data/`（数据读写、预处理、collate）
  - `models/`（encoder/quantizer/decoder）
  - `losses/`
  - `train/`（trainer、优化器、日志）
  - `eval/`（评估指标、可视化）
  - `utils/`（seed、分布式、导出）

- `scripts/`
  - `prepare_dataset.py`
  - `train_tokenizer.py`
  - `eval_tokenizer.py`
  - `export_tokenizer.py`（导出 codebook 与模型权重）

- `tests/`
  - 单元测试、shape 测试、回归测试

- `README.md`（运行入口）
- `DEV_GUIDE.md`（开发规范、模块协议）

验收标准：按 README 三条命令能完成 数据准备→训练→评估，且评估输出包含码本使用率与多项误差。

---

二、数据规范与预处理（最容易踩坑的部分，写死协议）  
2.1 输入张量协议  
每个样本是一段序列：

- `pose`: shape = `[T, J, D]`
  - `T` 可变长
  - `J` 关节数
  - `D` 默认为 2 或 3

- 可选 `conf`: shape = `[T, J, 1]`，2D keypoint 置信度
- 可选 `meta`: fps、视频 id、帧时间戳等

统一内部格式：  
`x`: `[B, T, J, D]`  
`mask`: `[B, T]`，有效帧为 1，padding 为 0

2.2 数据文件格式（建议 npz，避免视频依赖）  
`*.npz` 至少包含：

- `pose`（float32, \[T,J,D\]）
- 可选 `conf`（float32, \[T,J,1\]）
- 可选 `fps`（int）

2.3 预处理流水线（每一步可开关，参数可配）  
必须实现以下模块，顺序固定：

(1) 缺失值处理

- NaN/Inf 替换为 0，并在日志里记录样本 id
- 可选：线性插值修补短缺口（`interp_max_gap` 可配）

(2) Root-relative（强烈建议默认开启）

- 选定 `root_joint`（如胸骨/骨盆）
- 每帧所有关节减去 root 坐标
- 可选：对 root 的全局轨迹另存为 `root_traj`（decoder 可重建时加回去）

(3) 尺度归一化（2D 特别需要）

- `scale_mode`: `shoulder_width` 或 `bone_length_mean`
- 每帧除以该尺度，避免“人离镜头远近”污染 token

(4) 可选平滑（2D 噪声大时打开）

- 一阶低通或 Savitzky-Golay
- 参数：`smooth_window`, `smooth_polyorder`

(5) 关节拓扑映射

- 统一成项目内部的关节顺序
- 提供 `skeleton_definition.json`，含边列表 `edges`、对称关节对 `mirror_pairs`、骨段列表 `bones`

验收标准：`scripts/prepare_dataset.py` 跑完后输出

- 样本数、平均 T、P95 T
- 坐标均值方差（归一化后应在合理范围）
- NaN 修复计数
- 2D 时 conf 分布统计

---

三、模型协议与张量形状（写死，便于 debug）  
定义符号：

- 输入长度 `T`
- 下采样率 `l`（例如 4）
- token 长度 `L = ceil(T / l)`
- latent 维度 `d_model`

3.1 Encoder（ST-GCN + Temporal Compression）  
输入：`x` `[B,T,J,D]`

(1) 输入嵌入层

- 将 `[J,D]` 映射到 `d_in`（线性层或 1x1 conv）  
  输出：`h0` `[B,T,J,d_in]`

(2) ST-GCN blocks（空间图卷积 + 时间卷积）

- `num_stgcn_blocks` 可配，默认 2 或 3
- 每个 block 保持 `[B,T,J,C]`，带 residual  
  输出：`h1` `[B,T,J,C]`

(3) Temporal Downsample（只在时间维降采样）  
强制使用“可学习下采样”，不做生硬切块平均：

- 两层 temporal conv：stride = 2 + 2，总 stride = 4
- kernel size、dilation 可配  
  输出：`z_e` `[B,L,J,C2]`

(4) 聚合成 token latent（两种模式可选）

- `token_latent_mode = "per_joint"`：保留每个关节，`z_e` reshape 为 `[B,L,J,C2]` 送入量化
- `token_latent_mode = "pooled"`：对 J 做 attention/mean pooling 得到 `[B,L,d_model]`

建议默认 `pooled`，原因是码本更稳，先跑通再做 per_joint 扩展。

最终 encoder 输出：`z_e` `[B,L,d_model]`

验收标准：任意 batch 输入，encoder 输出长度符合 `L = ceil(T/l)`，mask 同步生成 `mask_L` `[B,L]`。

3.2 Quantizer（RVQ，残差向量量化）  
输入：`z_e` `[B,L,d_model]`

参数：

- `rvq.num_codebooks`（默认 3）
- `rvq.codebook_size`（默认 1024）
- `rvq.code_dim`（默认与 d_model 相同）
- `rvq.commitment_weight`（默认 0.25）
- `rvq.ema_update`（默认 true，避免不稳定）

输出：

- `z_q` `[B,L,d_model]`，量化后的向量
- `codes` `[B,L,num_codebooks]`，每层 code id
- `vq_losses`：embedding/commitment/perplexity 等

必须实现：码本使用率统计

- `perplexity`
- `usage_histogram`（每 N step 打印 top-k 使用 code）

验收标准：训练若干步后 perplexity 非常小（接近 1）时，日志要明确提示“码本塌缩风险”。

3.3 Decoder（对称上采样 + ST-GCN refine）  
输入：`z_q` `[B,L,d_model]`

(1) token 扩展到关节

- 若 encoder 用 pooled，decoder 第一层把 `[B,L,d_model]` 投影到 `[B,L,J,C]`
- 可用线性层 + reshape

(2) Temporal Upsample

- 两层 transposed conv 或 upsample+conv，stride=2+2 恢复到 `[B,T,J,C]`
- `upsample_mode` 可配

(3) ST-GCN refine blocks

- `num_refine_blocks` 可配，默认 2  
  输出：`x_hat` `[B,T,J,D]`

验收标准：`decode(encode(x))` 输出 shape 与输入一致（除了 padding 部分按 mask 忽略）。

---

四、Loss 设计（确保“token 可复原”而不是 decoder 太菜）  
总损失：  
`L = w_pos * L_pos + w_vel * L_vel + w_acc * L_acc + w_bone * L_bone + w_vq * L_vq (+ 可选 w_mask * L_maskpred)`

必须实现的项：

4.1 位置重建 `L_pos`

- L1：`|x_hat - x|`，按 `[B,T,J,D]` 与 `mask` 加权

4.2 速度误差 `L_vel`

- `Δx = x[t] - x[t-1]`
- L1，按有效帧对齐（从 t=1 开始）

4.3 加速度误差 `L_acc`（可选，默认开）

- `Δ²x = Δx[t] - Δx[t-1]`

4.4 骨长约束 `L_bone`

- 对 skeleton_definition 里的每条骨段 `(i,j)`
- 约束 `||x_i - x_j||` 在重建前后接近
- 对 2D 建议默认开启

4.5 VQ 损失 `L_vq`

- commitment + embedding（RVQ 各层累加）
- EMA 模式下 embedding loss 的实现细节按 quantizer 设计

重要：所有权重都必须来自配置文件，默认值给一套能收敛的起步：

- `w_pos=1.0`
- `w_vel=0.5`
- `w_acc=0.25`
- `w_bone=0.5`
- `w_vq=1.0`

验收标准：训练日志里单独打印每项 loss，禁止只打印总 loss。

---

五、训练流程（无监督 tokenizer 的标准训练）  
5.1 Trainer 必须支持

- 单机多卡 DDP
- 混合精度（bf16 或 fp16，配置可选）
- 梯度裁剪 `grad_clip_norm`
- 自动保存 best checkpoint（按验证集指标）

5.2 数据切片策略（避免超长序列炸显存）  
支持两种方式，可配：

- `clip_mode="random_window"`：从长序列随机采样固定窗口 `window_T`
- `clip_mode="full"`：整段输入，padding 到 batch max

建议默认 `random_window`，比如 `window_T=256`，这样 batch 更稳定。该数值必须可配。

5.3 学习率与 warmup

- AdamW
- `lr`, `weight_decay`, `warmup_steps`, `cosine_decay` 可配

5.4 码本稳定化策略（必须实现其中至少一个）

- EMA 更新 codebook（默认开启）
- dead code 重置（`dead_code_threshold` 可配）

验收标准：训练 1 小时内（取决于数据），perplexity 从接近 1 上升到明显大于 1，且 usage histogram 不集中在极少 code。

---

六、评估指标与可视化（没有这块就等于瞎训练）  
6.1 必须输出的指标

- `recon_l1`
- `vel_l1`
- `acc_l1`
- `bone_len_mae`
- `codebook_perplexity`（每层 RVQ 单独统计）
- `dead_code_ratio`（使用频率低于阈值的 code 比例）
- token 长度统计：平均 L、P95 L（按真实 T 计算）

6.2 可视化（至少一种）

- 把某个样本的原始 pose 与重建 pose 画成时间曲线（关节坐标随时间）
- 或导出成简单动画（可选）

验收标准：`scripts/eval_tokenizer.py` 生成一个 `metrics.json` 和若干图，开发者不用额外工具就能打开。

---

七、可选增强任务（先写好接口，默认关闭）  
这部分对应“ASR token / self-supervised”的同构技巧，用来把 token 从“压缩码”推向“单位符号”。

7.1 Masked Token Prediction（离散单位预测）  
实现：

- 对 `codes` 序列随机 mask 连续 span（`mask_prob`, `mask_span_len` 可配）
- 用一个小 Transformer 预测被 mask 的 code id
- 损失：cross entropy，权重 `w_mask`

验收标准：开关打开后训练能跑通，日志打印 mask accuracy。

7.2 多尺度 token（分层时间分辨率）  
实现：

- stride=4 的细粒度 token
- stride=16 的粗粒度 token
- 训练时同时重建或做一致性约束

只需要预留结构与配置项，默认不启用。

---

八、配置文件规范（必须完整列出，开发按它写）  
提供一个 YAML schema，至少包含以下字段：

yaml

复制代码

```
data:
  dataset_path: "..."
  use_confidence: true
  dims: 2                # 2 or 3
  fps: 25
  skeleton_def: "configs/skeleton_definition.json"
  preprocess:
    root_relative: true
    root_joint: 0
    scale_norm: true
    scale_mode: "shoulder_width"
    smoothing: false
    smooth_window: 9
    smooth_polyorder: 2
    interp_missing: true
    interp_max_gap: 3

train:
  seed: 42
  batch_size: 16
  num_workers: 8
  clip_mode: "random_window"
  window_T: 256
  max_epochs: 100
  amp: "bf16"            # off/fp16/bf16
  grad_clip_norm: 1.0
  optimizer:
    name: "adamw"
    lr: 3e-4
    weight_decay: 1e-2
  scheduler:
    name: "cosine"
    warmup_steps: 2000

model:
  d_model: 256
  downsample:
    factor_l: 4          # {2,4,8} ablation
    tconv_kernel: 5
    tconv_layers: 2
    tconv_dilation: 1
  encoder:
    num_stgcn_blocks: 3
    channels: [64,128,256]
    token_latent_mode: "pooled"  # pooled/per_joint
  quantizer:
    type: "rvq"
    num_codebooks: 3
    codebook_size: 1024
    ema_update: true
    commitment_weight: 0.25
    dead_code_threshold: 1e-5
  decoder:
    num_refine_blocks: 2
    upsample_mode: "transposed_conv"

loss:
  w_pos: 1.0
  w_vel: 0.5
  w_acc: 0.25
  w_bone: 0.5
  w_vq: 1.0
  masked_pred:
    enabled: false
    w_mask: 1.0
    mask_prob: 0.15
    mask_span_len: 6

logging:
  log_every_steps: 50
  eval_every_steps: 2000
  save_every_steps: 2000
  out_dir: "runs/tokenizer_v1"
```

验收标准：所有 “需要实验验证的数值” 都只能从 config 读，代码里禁止写死常数。

---

九、开发任务清单（按里程碑推进，工程师照抄执行）  
Milestone 0：脚手架与可运行（1 天）

- T0.1 初始化仓库结构、lint、format、pre-commit
- T0.2 配置系统（OmegaConf/Hydra 或纯 yaml + dataclass），实现 config 校验
- T0.3 统一 logger，输出到 console + jsonl

Milestone 1：数据与预处理（2–4 天）

- T1.1 实现 npz 数据集类，支持可变长 T
- T1.2 实现 collate_fn：padding + mask
- T1.3 实现 preprocess pipeline（root-relative/scale/smooth/interp）
- T1.4 写 `prepare_dataset.py`，生成统计报告  
  验收：随机取 10 个样本，预处理后数值范围合理，mask 正确，统计文件齐全。

Milestone 2：模型核心（3–6 天）

- T2.1 实现 skeleton_definition 读取与图结构构建
- T2.2 实现 ST-GCN block（含 residual、BN）
- T2.3 实现 temporal downsample 模块（stride conv）
- T2.4 实现 RVQ quantizer（含 EMA、dead code reset、perplexity）
- T2.5 实现 decoder（upsample + refine）  
  验收：前向传播不报错，`x -> z_e -> z_q -> x_hat` 全链路 shape 正确。

Milestone 3：Loss 与训练（2–5 天）

- T3.1 实现 L_pos/L_vel/L_acc/L_bone
- T3.2 实现训练循环（DDP 可选，单卡先跑通）
- T3.3 实现 checkpoint 保存与恢复
- T3.4 实现评估循环与 metrics 输出  
  验收：在小数据集上训练 2–3 个 epoch，loss 单调下降，perplexity 提升，重建误差下降。

Milestone 4：实验与稳定性（3–7 天）

- T4.1 ablation 配置：`l={2,4,8}`，`num_codebooks={1,2,3,4}`，`codebook_size={512,1024,2048}`
- T4.2 码本塌缩检测与告警
- T4.3 重建可视化脚本  
  验收：每组实验产出统一的 `metrics.json`，可对比表格自动生成。

Milestone 5：可选增强（2–6 天）

- T5.1 Masked token prediction 模块（默认关闭）
- T5.2 多尺度 token 结构预留（默认关闭）  
  验收：打开开关训练不崩，mask accuracy 有意义输出。

Milestone 6：测试与文档（并行进行）

- T6.1 单元测试：shape、mask、quantizer 输出范围、dead code reset
- T6.2 回归测试：固定 seed 下 100 step loss 数值落在区间
- T6.3 完整文档：运行命令、常见报错（NaN、perplexity=1、骨长爆炸）的排查流程

---

十、常见失败模式与强制排查项（写进 DEV_GUIDE）

1.  perplexity 长期接近 1：码本塌缩  
    排查：学习率过大、commitment 权重不合适、encoder 输出方差过小、EMA 没开、dead code reset 没触发
2.  重建很平滑但动作失真  
    排查：stride 太大（l=8）、vel/acc 权重太低、decoder capacity 太弱或太强导致绕过量化瓶颈
3.  2D token 学到视角/尺度而非动作  
    排查：root-relative 与 scale_norm 是否开启、conf 是否参与输入、平滑是否开启、是否启用 masked_pred
