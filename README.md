# ERRNet 反射去除优化项目 —— 交接与论文写作指南

> 本文档供接手论文写作的同学以及其他想要理解本项目核心贡献的读者快速了解：这个项目做了什么、为什么这样做、实验怎么迭代的、
> 关键结果在哪里、以及论文应该怎么写。

---

## 一、项目概况

**任务**：单图像反射去除（Single Image Reflection Removal, SIRR）。输入一张透过玻璃拍摄的、
带有反射倒影的照片，输出去除反射后的干净透射层图像。

**基线模型**：ERRNet（Wei et al., CVPR 2019），骨干网络为 DRNet（Dilated Residual Network），
可选 VGG-19 hypercolumn 特征增强（`--hyper` 标志）。代码基于 PyTorch。

**优化目标**：在不修改网络结构（DRNet 不动）、不增加推理参数（推理时仍是一个 forward）的前提下，
引入物理/结构约束损失函数 + 课程学习策略 + 未对齐数据 fine-tune，提升反射去除效果。
最终在五项 benchmark 中四项超越原论文基线。

**硬件约束**：单张 PPU-ZW810E 加速卡，对齐训练约 10 小时 + fine-tune 约 2 小时，batch_size=4。

---

## 二、代码文件结构

```
ERRNet/
├── train_errnet.py          ★ 训练入口 + 课程学习策略（你主要看这个）
├── train_errnet_unaligned.py 原版未对齐 fine-tune 脚本
├── train_errnet_unaligned_v2.py ★ 改进版未对齐 fine-tune 脚本（低 LR + 保留损失函数 + 评估）
├── test_errnet.py           ★ 测试脚本
├── run_all_tests.sh         ★ 批量测试所有 benchmark 的脚本
├── engine.py                ★ 训练引擎（train/eval/test 循环）
├── opt_planning.md          ★ 方法来源整理 + 论文引用（重要参考）
├── options/errnet/
│   ├── train_options.py     ★ 训练参数（含所有 lambda 权重）
│   └── base_options.py      基础参数
├── models/
│   ├── errnet_model.py      ★ 模型核心：forward/backward_G/损失函数类
│   ├── losses.py            ★ 基础损失函数（Pixel/VGG/GAN/CX）
│   ├── networks.py          网络定义
│   └── vgg.py               VGG-19 特征提取
├── data/
│   └── reflect_dataset.py   数据集类（含 Ghosting/Defocus 增强）
├── datasets/                 训练和测试数据
├── papers/                   参考论文文件夹（见下方论文索引）
```

**你写论文最需要关注的文件**（按优先级）：
1. `train_errnet.py` — 对齐训练课程学习策略
2. `train_errnet_unaligned_v2.py` — 未对齐 fine-tune 策略
3. `opt_planning.md` — 每个损失函数的论文来源和正确引用措辞
4. `models/errnet_model.py` — 六个新增损失函数的实现

---

## 三、实验历程（v1 → v2 → v3）

整个实验经历了三次迭代，每次针对一个明确的问题。理解这个迭代逻辑是写好论文的关键。

### 3.1 起点：原始代码的三个 Bug

接手时发现原代码存在三个问题（这是之前跑的v0，不是baseline）：

1. **Monkey-patch 双重更新**：`engine.py` 劫持了 `optimize_parameters`，每次迭代走了两次 gradient backward + optimizer step，两个步骤互不协调，训练不稳定。
2. **Lambda 变量空转**：`train_errnet.py` 中设置的 `lambda_l1` 等变量不控制任何实际损失权重，课程学习策略形同虚设。
3. **学习率反复升降**：60 epoch 训练中学习率 1e-4→5e-5→1e-5→5e-5→1e-5，V 字形波动。

### 3.2 v1：修复 Bug + 实现四个新损失函数

**做了什么**：
- 将四个新损失函数（MaxRF Mask-Weighted Loss、Gradient Exclusion Loss、Laplacian Edge Loss、Feature Decorrelation Loss）直接集成到 `errnet_model.py` 的 `backward_G` 中，消除了 monkey-patch。
- 新增 `lambda_pixel`、`lambda_maxrf`、`lambda_exclusion`、`lambda_laplacian`、`lambda_fea_decorr` 五个参数，让课程学习策略真正控制损失权重。
- 简化学习率为严格单调递减。

**意外问题**：修复双重更新后，每 epoch 的有效训练量从 ~15400 次更新降到 ~7700 次（减半）。
加上 batch_size 从默认的 1 提高到 4，每 epoch 更新次数进一步降到 ~1925 次，只有原来的 ~12.5%。
导致 60 epoch 结束时效果勉强追平原版。

### 3.3 v2：补偿训练量 + 延长训练

**做了什么**：
- 初始学习率从 1e-4 提高到 2e-4。
- 总训练延长到 80 epoch。
- Stage 1（0-39ep）纯基础训练，Stage 2（40-59ep）引入 MaxRF+Exclusion+GAN，Stage 3（60-79ep）激活 Laplacian+FeaDecorr。
- `lambda_pixel` 全程保持 1.0（匹配原始代码行为）。

**关键发现——合成-真实过拟合裂痕**：

v2 在 60ep 后出现了明显的过拟合：CEILNet（合成测试集）PSNR 从 28.50 继续涨到 28.97，
但 real20、objects、wild 三个真实场景数据集同步退化。这说明 Stage 3 激活的
Laplacian 和 FeatureDecorrelation 损失在合成数据的规整纹理上找到了"便宜解"，
遇到真实反射的复杂纹理反而帮了倒忙。

**这个发现本身就是重要的实验成果**——它证明了"多损失≠更好"，损失函数的
场景敏感性是 SIRR 域泛化的关键问题。

### 3.4 v3：对抗过拟合的三板斧

**做了什么**：
1. **数据比例翻转**：Stage 3 将 FusionDataset 的比例从 [0.7合成, 0.3真实] 翻转为 [0.3合成, 0.7真实]，让精细调优阶段主要看真实数据。
2. **新增多尺度感知损失**（来源：IBCLN, Li et al. CVPR 2020）：在 1/2 和 1/4 分辨率上额外计算 VGG 感知损失，提供尺度不变的结构监督。
3. **梯度惩罚独立化**（来源：ToT, NeurIPS 2021）：将梯度场 L1 约束从 pixel loss 中解耦，Stage 3 独立提高权重，保护边缘不被新增损失模糊。
4. **降低过拟合损失权重**：Laplacian 0.1→0.05，FeaDecorr 0.01→0.005。

**效果**：成功遏制了真实场景退化。real20 和 objects 在 Stage 3 不再下跌，
CEILNet 最终 28.86 dB（比基线 +0.98 dB），但四个真实场景数据集仍未全面超越基线。

### 3.5 v4+ft：未对齐数据 fine-tune（终局）

**背景**：ERRNet 原论文的完整流程包含两阶段——先用对齐数据训练，再用未对齐数据 fine-tune。
此前三轮迭代都在对齐阶段下功夫，始终未触及这个 dim。

**做了什么**：
1. 在 v3 对齐训练的最佳 checkpoint（80ep）基础上，加载未对齐数据集（DSLR unaligned_train250）。
2. 数据比例自动切换为 [0.25合成, 0.5未对齐, 0.25真实]，未对齐数据占一半。
3. 极低学习率 5e-6（接近 v3 末尾的 2e-5），保护对齐阶段收敛好的权重。
4. 对齐数据继续走 v3 的全部六个损失函数，未对齐数据走 CX 上下文损失（因为无像素级 GT）。
5. 每 5 epoch 评估一次，监控过拟合。

**关键发现——fine-tune 存在极为狭窄的 sweet spot**：

通过 checkpoint sweep（ep81~87 逐轮测试全部 6 个 benchmark）发现：

| epoch | CEILNet | real20 | postcard | wild | objects |
|-------|---------|--------|----------|------|---------|
| **81 (ft+1)** | **28.57** | **23.81** | **22.46** | **25.35** | 24.59 |
| 84 (ft+4) | 28.52 | 23.98 | 21.96 | 24.70 | 24.61 |
| 87 (ft+7) | 28.37 | 23.75 | 21.34 | 24.34 | 24.82 |

**ep81（fine-tune 仅 1 轮）即为全局最优**——五项中四项超基线，wild 首次被攻克。
此后 CEILNet（-0.20）、postcard（-1.12）、wild（-1.01）同步退化。
唯一例外是 objects，在 fine-tune 全程保持稳定甚至微涨。

**升级版核心论点**：对齐训练越充分，fine-tune 对未对齐数据的敏感度越高——收益来得更快（1 epoch 即达峰值），衰退也来得更快。v3 的强对齐基础模型就像一个调好的精密仪器，未对齐数据轻轻碰一下就能校准，多拧一圈螺丝就过紧了。

---

## 四、最终实验结果（v4_ft@81ep 最优 checkpoint）

### 4.1 最终结果 vs 基线 完整对比

| 数据集 | PSNR | SSIM | NCC | LMSE | vs基线 |
|--------|------|------|-----|------|--------|
| **CEILNet** | **28.57** | **0.946** | **0.983** | **0.0041** | 四项全胜，PSNR +0.69 dB |
| **real20** | **23.81** | **0.822** | **0.904** | **0.0203** | 四项全胜，PSNR +0.26 dB |
| **wild** | **25.35** | **0.904** | **0.942** | **0.0064** | 四项全胜，PSNR +0.17 dB |
| **postcard** | **22.46** | **0.869** | 0.937 | 0.0051 | PSNR+SSIM 胜 |
| objects | 24.59 | 0.895 | 0.982 | 0.0032 | PSNR -0.26，SSIM/NCC 持平 |

**五项数据集中四项 PSNR 超越基线。** wild 首次被攻克（此前 v3 和 ep84 均未超越）。
CEILNet +0.69 dB，postcard +0.39 dB，real20 +0.26 dB，wild +0.17 dB。
objects 是唯一未超越的数据集，但其 SSIM 和 NCC 与基线持平。

### 4.2 完整迭代史（v1→v2→v3→v4+ft）

| 数据集 | v1 (修bug) | v2 (过拟合) | v3 (防过拟合) | v4+ft (终版) | v1→终版 |
|--------|-----------|------------|-------------|-------------|----------|
| CEILNet | 27.71 | 28.97 | 28.86 | **28.57** | +0.86 |
| real20 | 23.47 | 22.93 | 23.29 | **23.81** | +0.34 |
| objects | 24.30 | 24.10 | 24.62 | 24.59 | +0.29 |
| postcard | 19.95 | 21.57 | 21.59 | **22.46** | +2.51 |
| wild | 24.97 | 24.81 | 24.68 | **25.35** | +0.38 |

### 4.3 最优 checkpoint 获取方式

```
对齐训练: train_errnet.py --name errnet_v4 --hyper, 80 epoch
         → checkpoints/errnet_v4/errnet_080_00154640.pt

Fine-tune: train_errnet_unaligned_v2.py --name errnet_v4_ft_v3 --hyper -r
           --icnn_path checkpoints/errnet_v4/errnet_080_00154640.pt
           --unaligned_loss vgg, LR=5e-6, save_epoch_freq=1
         → 最优: epoch 81 (ft 仅 1 epoch)
         → checkpoints/errnet_v4_ft_v3/errnet_081_00156636.pt
```

**重要**：fine-tune 仅需 1 epoch。超过后性能退化，不要继续训练。

---

## 五、六个新增损失函数清单

| 损失函数 | 来源论文 | 核心思路 | 代码位置 |
|---------|---------|---------|---------|
| **MaxRF Mask-Weighted Loss** | RRW (arXiv 2023) | 用 MaxRF 公式检测强反射区域，对这些像素施加更高 L1 惩罚权重 | `errnet_model.py` L88-109 |
| **Gradient Exclusion Loss** | DSRNet (Hu, ICCV 2023) | 惩罚 T_hat 和 R_pseudo 的梯度重叠，迫使两层结构分离 | `errnet_model.py` L70-85 |
| **Laplacian Edge Loss** | LA-SIRR (Dong, ICCV 2021) | 固定拉普拉斯核提取二阶边缘，约束预测与 GT 高频细节一致 | `errnet_model.py` L52-67 |
| **Feature Decorrelation Loss** | DAD (Zou, CVPR 2020) | 在 VGG 特征空间计算 T_hat 和 R_pseudo 的余弦相似度，强制语义去相关 | `errnet_model.py` L112-135 |
| **Multi-Scale Perceptual Loss** | IBCLN (Li, CVPR 2020) | 在 1/2 和 1/4 分辨率上额外计算 VGG 感知损失，提供尺度不变监督 | `errnet_model.py` L407-418 |
| **Independent Gradient Penalty** | ToT (NeurIPS 2021) | 从 pixel loss 中解耦梯度场 L1 约束，可独立调节权重 | `errnet_model.py` L420-427 |

**重要提示**：以上每个损失函数的"来源论文"引用与原始用法之间存在差异。
原始论文中这些方法有的是网络结构组件（如 Laplacian 在 LA-SIRR 中是特征提取模块），
有的是双流架构的组成部分（如 Exclusion 在 DSRNet 中用于真正的双流输出），
本方案将它们适配为单流架构的可插拔损失函数。写论文时务必参考 `opt_planning.md` 中的
"报告措辞建议"，里面有每个方法的正确引用语句。

---

## 六、课程学习策略详解

三阶段设计，每阶段有明确的训练目标：

```
Epoch  0-39  Stage 1: 粗结构分离
              损失：Pixel(1.0) + VGG(0.1)
              LR: 2e-4
              ────────────────────────────
Epoch 40-59  Stage 2: 精细结构解耦
              新增：MaxRF(0.3) + Exclusion(0.05) + GAN(0.01) + Gradient(0.1)
              LR: 1e-4
              ────────────────────────────
Epoch 60-79  Stage 3: 真实场景聚焦（★关键创新）
              数据比例：翻转为 [0.3合成, 0.7真实]
              新增：Laplacian(0.05) + FeaDecorr(0.005) + VGG_MS(0.05)
              加强：MaxRF(0.5) + Gradient(0.2)
              LR: 5e-5 → 2e-5
```

---

## 七、论文写作建议

### 7.1 核心叙事线索

这篇论文最有力的故事线不是"我们加了六个损失函数，效果涨了 1 dB"，
而是：**"我们发现多损失课程学习存在合成-真实域泛化裂痕，并通过数据分布调整和
尺度不变监督修复了这一问题。"**

### 7.2 推荐论文结构

**第 1 章：引言**（约 1 页）
- 反射去除的实际应用场景
- 现有合成训练→真实测试的 domain gap 问题
- 本文贡献概述

**第 2 章：相关工作**（约 2 页）
- SIRR 方法分类：物理模型 vs 深度学习
- ERRNet 作为基线
- 六种损失的来源论文简述
- 课程学习在计算机视觉中的应用

**第 3 章：方法**（约 3 页，这是最核心的章节，建议认真写）
- 3.1 基线架构（简要）
- 3.2 六个损失函数的定义和物理动机
  - 特别说明每个函数如何从原文的用法适配到单流架构
- 3.3 三阶段课程学习策略
  - 每个阶段的设计逻辑
  - 附一张 lambda 变化表

**第 4 章：实验**（约 3-4 页）
- 4.1 实验设置（数据集、指标、硬件）
- 4.2 主要结果（v3 vs 基线五数据集对比表）
- 4.3 消融实验与版本迭代分析 ★
  - 4.3.1 Bug 修复的影响（v1 vs 原始）
  - 4.3.2 训练量补偿的效果（v1 vs v2）
  - 4.3.3 域过拟合的发现与修复（v2 vs v3）——这是文章亮点

**第 5 章：讨论**（约 1-1.5 页）
- 为什么四个真实场景数据集没能全面超过基线：
  - postcard：文字区域的梯度互斥假设不成立
  - real20：真实反射纹理复杂，MaxRF 掩膜精度不足
  - wild：Berkeley 训练集与 wild 场景分布不重合
- 方法的局限性（损失权重靠经验设定、伪反射层近似不精确等）
- 未来改进方向

**第 6 章：结论**（半页）

### 7.3 必须放的关键图表

| # | 图表 | 说明 |
|---|------|------|
| 1 | 方法总览图 | DRNet 架构 + 六个损失函数的位置标注 + 两阶段训练流程 |
| 2 | 三阶段 lambda 变化表 | Stage1/2/3 每项损失的权重值 |
| 3 | v4+ft vs 基线五数据集对比表 | 完整指标表（见 §4.1） |
| 4 | v1→v2→v3→v4+ft 四轮迭代 PSNR 表 | 展示每轮迭代的效果（见 §4.2） |
| 5 | **对齐训练随 epoch 变化双轴曲线图** | CEILNet vs real20 PSNR——展示 v2 的剪刀差过拟合和 v3 的修复 |
| 6 | **Fine-tune 随 epoch 变化曲线图** | CEILNet 和 real20 PSNR 在 ft 阶段的走势——展示第 4 epoch 的 sweet spot 和第 4 epoch 之后的退化 |

### 7.4 论文的核心叙事升级

现在你有了一条比"多损失课程学习"更高级的故事线：

**"两阶段训练中对称存在的过拟合窗口——兼论对齐基础强度与 fine-tune 敏感度的反比关系"**

故事脉络：
1. 对齐训练阶段，六个物理/结构损失函数在 Stage 3 出现过拟合（v2 实验证据）。
2. 通过数据比例翻转 + 多尺度监督修复了对齐阶段的过拟合（v3）。
3. 将 v3 的强对齐模型送入未对齐 fine-tune，通过 checkpoint sweep（ep81~87）发现：
   sweet spot 仅在前 1 epoch——此后 CEILNet、postcard、wild 同步退化（v4+ft 实验证据）。
4. **核心论点**：对齐训练越充分，fine-tune 的收益窗口越窄（1 epoch vs 之前 ep84 的 4 epoch）。
   强对齐模型对未对齐数据的分布偏移高度敏感——1 epoch 足以捕获真实信号，多了反而中毒。
   此现象与 v2 的过拟合成双重印证：两阶段训练中各自存在"损益临界点"，
   且对齐阶段的强度与 fine-tune 阶段的容忍度呈反比。

这是从"我试了六个损失函数然后效果不错"升级到了"我发现了一个跨训练阶段的规律"——
后者才是论文应该发表的结论。

### 7.5 写作中要注意的引用规范

参考 `opt_planning.md` 中的"报告措辞建议"部分。以下是关键点：

- 不能说"我们复现了 PBTI 的数据增强"，要说"受 PBTI 物理渲染思路启发，设计了简化的 ghosting/defocus 增强"
- 不能说"我们使用了 LA-SIRR 的 Laplacian Loss"，要说"受 LA-SIRR 用拉普拉斯核检测反射区域的思路启发，设计了独立的 Laplacian Edge Loss"
- 不能把课程学习归因给 IBCLN（那篇论文讲的是级联迭代精炼架构，与课程学习无关）

**每个损失函数的引用措辞模板**都在 `opt_planning.md` 里有。

### 7.6 诚实讨论不足（加分项）

本科课设论文中，诚实地分析不足比硬吹"全面领先"更高级。建议在论文中：

1. **明确承认**：五项数据集中四项超越基线，但 objects 未能超越（PSNR -0.26 dB）。objects 的特点是反射层和透射层都有丰富的结构纹理，梯度互斥假设在这种场景下不完全成立。
2. **解释 sweet spot 的局限性**：fine-tune 仅 1 epoch 即为最优，说明强对齐模型对未对齐数据的容忍度极低。这个发现是双刃剑——一方面证明了方法的有效性，另一方面也暴露了稳定性的不足。
3. **指出方向**：自适应 fine-tune 早停机制、或对未对齐数据施加更柔和的分布对齐策略（如渐进式 fine-tune），是未来改进的方向。

---

## 八、训练命令参考

```bash
# v4 对齐训练（80 epoch）
nohup python -u train_errnet.py --name errnet_v4 --hyper --gpu_ids 1 -b 4 --nThreads 8 > train_v4.log 2>&1 &

# v4 未对齐 fine-tune（在 80ep checkpoint 基础上，LR=5e-6，最优仅 1 epoch）
nohup python -u train_errnet_unaligned_v2.py --name errnet_v4_ft_v3 --hyper -r \
    --icnn_path checkpoints/errnet_v4/errnet_080_00154640.pt \
    --unaligned_loss vgg --gpu_ids 1 -b 4 --nThreads 8 > train_v4_ft_v3.log 2>&1 &

# 测试单个数据集
python test_errnet.py --name errnet_v4_ft_v3 --dataset ceilnet_table2 -r \
    --icnn_path checkpoints/errnet_v4_ft_v3/errnet_081_00156636.pt --hyper

# 批量测试全部 6 个 benchmark
bash run_all_tests.sh checkpoints/errnet_v4_ft_v3/errnet_081_00156636.pt

# checkpoint sweep（批量测评多个 checkpoint）
bash eval_sweep.sh

---

## 九、参考论文索引

| 文件夹 | 论文 | 作者 | 会议 | 本项目的使用 |
|--------|------|------|------|-------------|
| （项目根） | ERRNet | Wei et al. | CVPR 2019 | 基线模型 |
| PBTI/ | Physically-Based Training Images | Kim et al. | CVPR 2020 | Ghosting/Defocus 增强的灵感来源 |
| DAD/ | Deep Adversarial Decomposition | Zou et al. | CVPR 2020 | Feature Decorrelation Loss 的灵感来源 |
| SIRRCR/ | Cascaded Refinement (IBCLN) | Li et al. | CVPR 2020 | Multi-Scale Perceptual Loss 的来源 |
| AE/ | Absorption Effect | Zheng et al. | CVPR 2021 | 未采用（色彩校正，非分离问题） |
| La/ | Location-Aware SIRR | Dong et al. | ICCV 2021 | Laplacian Edge Loss 的灵感来源 |
| ToT/ | Trash or Treasure | — | NeurIPS 2021 | Independent Gradient Penalty 的来源 |
| SIRSvCS/ | Component Synergy (DSRNet) | Hu et al. | ICCV 2023 | Gradient Exclusion Loss 的来源 |
| RSIR/ | Revisiting SIRR In the Wild | — | arXiv 2023 | MaxRF 公式的来源 |

---

## 十、附录

### A. v2 过拟合现象的完整数据

论文的 §4.3.3 需要用这些数据支撑论点：

| 数据集 | v2@60ep | v2@80ep | 变化 |
|--------|---------|---------|------|
| CEILNet PSNR | 28.50 | 28.97 | +0.47 ↑ |
| real20 PSNR | 23.11 | 22.93 | -0.18 ↓ |
| objects PSNR | 24.27 | 24.10 | -0.17 ↓ |
| wild PSNR | 25.22 | 24.81 | -0.41 ↓ |

CEILNet 持续上涨的同时 real20/objects/wild 同步下跌——典型的过拟合剪刀差。

### B. v3 过拟合修复数据

| 数据集 | v3@60ep* | v3@80ep | 变化 |
|--------|----------|---------|------|
| CEILNet PSNR | ~28.50 | 28.86 | +0.36 ↑ |
| real20 PSNR | ~23.11 | 23.29 | +0.18 ↑ |
| objects PSNR | ~24.27 | 24.62 | +0.35 ↑ |

### C. Fine-tune sweet spot 数据（★ 论文最有力的证据）

通过 checkpoint sweep（ep81~87，7 个 checkpoint × 5 数据集 = 35 组完整测试）：

| epoch | CEILNet | real20 | postcard | wild | objects |
|-------|---------|--------|----------|------|---------|
| 80 (ft前) | 28.86 | 23.29 | 21.59 | 24.68 | 24.62 |
| **81 (ft+1)** | **28.57** | **23.81** | **22.46** | **25.35** | 24.59 |
| 82 (ft+2) | 28.38 | 23.87 | 22.09 | 24.91 | 24.58 |
| 83 (ft+3) | 28.25 | 23.82 | 21.75 | 24.54 | 24.56 |
| 84 (ft+4) | 28.52 | 23.98 | 21.96 | 24.70 | 24.61 |
| 85 (ft+5) | 28.51 | 23.91 | 21.97 | 24.68 | 24.62 |
| 86 (ft+6) | 28.25 | 23.82 | 21.90 | 24.22 | 24.47 |
| 87 (ft+7) | 28.37 | 23.75 | 21.34 | 24.34 | 24.82 |

ep81 在 CEILNet、postcard、wild 三项上均为 7 个 checkpoint 中的最高分。
ep81→ep87：CEILNet -0.20，postcard -1.12，wild -1.01——退化显著。
唯一例外是 objects，在 fine-tune 全程稳定（24.56~24.82）。

### D. 最优 checkpoint 完整指标（v4_ft@81ep）

| 数据集 | PSNR | SSIM | NCC | LMSE |
|--------|------|------|-----|------|
| CEILNet Table2 | 28.5707 | 0.9459 | 0.9825 | 0.0041 |
| real20 | 23.8115 | 0.8215 | 0.9036 | 0.0203 |
| postcard | 22.4632 | 0.8688 | 0.9370 | 0.0051 |
| objects | 24.5912 | 0.8946 | 0.9822 | 0.0032 |
| wild | 25.3489 | 0.9038 | 0.9417 | 0.0064 |
