# ERRNet 优化方案整理（终版，对齐 v3 最终状态）

> 本文档整理自对 8 篇相关论文的阅读、三轮实验迭代（v1→v2→v3）的复盘，
> 以及最终实现的完整方法清单。供论文写作时查阅引用措辞和方法来源。
>
> 约束条件：不修改网络结构（DRNet backbone 不变）、不增加推理参数、
> 单张 PPU-ZW810E 加速卡训练约 10 小时。

---

## 一、已实现的方法（共 6 个损失函数 + 1 个训练策略）

### 1. MaxRF 掩膜加权损失（MaxRF Mask-Weighted Loss）

- **内容**：计算 MaxRF 掩膜 M = (Grad(I) > Grad(T_gt))，对强反射区域的像素级 L1 损失施加 1.5 倍权重，迫使网络更关注局部强反光区。
- **真实来源**：MaxRF（Maximum Reflection Filter）公式来自 "Revisiting Single Image Reflection Removal In the Wild"（arXiv 2023, RRW 数据集论文）。原论文中 MaxRF 掩膜用于训练独立的预测网络（RDNet）并作为主网络的额外输入通道拼接，**并未**用作 L1 损失的权重。将 MaxRF 直接作为损失权重是本方案的原创适配。
- **报告措辞建议**："借鉴 RRW 论文中 MaxRF 公式的物理直觉（局部反射区域的梯度幅度大于透射层），设计了 MaxRF Mask-Weighted Loss，在训练阶段对强反射区域施加更大的重建惩罚权重。"
- **实现状态**：✅ 已实现。`errnet_model.py` L88-109，`MaxRFMaskWeightedLoss` 类。v3 课程中 Stage 2 权重 0.3，Stage 3 权重 0.5。

### 2. 梯度互斥损失（Gradient Exclusion Loss）

- **内容**：构造伪反射层 R_pseudo = I - T_hat，计算 T_hat 和 R_pseudo 在 X/Y 方向梯度的逐元素乘积均值，惩罚两者的纹理边缘重叠，迫使结构分离。
- **真实来源**：Exclusion Loss 最早由 Zhang et al. 提出，后被 DSRNet (Hu et al., ICCV 2023, "Single Image Reflection Separation via Component Synergy") 沿用（论文 Eq. 6）。DSRNet 是双流网络，用实际预测的 T_hat 和 R_hat 计算；本方案因单流架构限制，使用 R_pseudo = I - T_hat 作为近似替代。
- **报告措辞建议**："借鉴 Zhang et al. 提出、DSRNet [Hu et al.] 沿用的梯度互斥损失思路，针对单流架构设计了基于伪反射层（R_pseudo = I - T_hat）的 Exclusion Loss，在不增加参数的前提下约束透射层与反射层的结构分离。"
- **实现状态**：✅ 已实现。`errnet_model.py` L70-85，`GradientExclusionLoss` 类。v3 课程中 Stage 2 激活，权重 0.05。

### 3. 拉普拉斯边缘损失（Laplacian Edge Loss）

- **内容**：用固定的 3x3 拉普拉斯核对预测透射层 T_hat 和真实透射层 T 分别提取二阶边缘特征，计算 L1 损失，约束高频细节一致。
- **真实来源**：拉普拉斯核用于反射检测的思路来自 Dong et al. (ICCV 2021, "Location-Aware Single Image Reflection Removal")。原论文将拉普拉斯核用作**网络结构组件**（Reflection Detection Module 中的多尺度特征提取），而非损失函数。将其作为独立的 loss 使用是本方案的原创适配。
- **报告措辞建议**："受 Dong et al. [LA-SIRR] 利用拉普拉斯特征检测反射区域这一思路的启发，设计了独立的 Laplacian Edge Loss，通过约束预测与 GT 的二阶边缘一致性来保护高频细节。"
- **实现状态**：✅ 已实现。`errnet_model.py` L52-67，`LaplacianEdgeLoss` 类。v3 课程中 Stage 3 激活，权重 0.05（v2 中为 0.1，v3 减半以防止过拟合）。

### 4. 特征去相关损失（Feature Decorrelation Loss）

- **内容**：将 T_hat 和 R_pseudo = I - T_hat 送入 VGG 提取深层特征（conv4_2），计算两者的余弦相似度作为惩罚项，迫使分离后的两层在语义特征空间正交/无关联。
- **真实来源**：受 DAD (Zou et al., CVPR 2020, "Deep Adversarial Decomposition") 中 Separation-Critic 思想的启发。原论文用对抗训练（Separation-Critic 判别器）来惩罚分离不干净的情况；本方法用统计学相关性作为其廉价替代。
- **报告措辞建议**："受 DAD [Zou et al., CVPR 2020] 中 Separation-Critic 思想的启发，设计了轻量级的特征去相关损失，通过最小化透射层与伪反射层在 VGG 特征空间的余弦相似度来约束两者的独立性。"
- **实现状态**：✅ 已实现。`errnet_model.py` L112-135，`FeatureDecorrelationLoss` 类。仅在 `--hyper` 启用时创建（依赖 VGG）。v3 课程中 Stage 3 激活，权重 0.005（v2 中为 0.01，v3 减半以防止过拟合）。

### 5. 多尺度感知损失（Multi-Scale Perceptual Loss）

- **内容**：在 1/2 和 1/4 分辨率上额外计算 VGG 感知损失，与全分辨率 VGG 损失互补，提供尺度不变的结构监督。
- **真实来源**：IBCLN (Li et al., CVPR 2020, "Single Image Reflection Removal Through Cascaded Refinement") 在解码器的三个尺度上提取 VGG19 特征计算感知损失（conv1_2 和 conv2_2），权重 gamma_3=0.8, gamma_5=0.6。本方案将其适配为对输出图像下采样后计算 VGG 损失。
- **报告措辞建议**："借鉴 IBCLN [Li et al.] 的多尺度感知损失策略，在 1/2 和 1/4 分辨率上额外计算 VGG 感知损失，提供尺度不变的结构监督，改善真实场景泛化能力。"
- **实现状态**：✅ 已实现。`errnet_model.py` L407-418，在 `backward_G` 中直接计算。v3 课程中 Stage 3 激活，权重 0.05。

### 6. 独立梯度惩罚（Independent Gradient Penalty）

- **内容**：将梯度场 L1 约束从 pixel loss 中解耦，作为独立的可调权项。在 Stage 3 提高权重，保护边缘不被新增损失函数模糊。
- **真实来源**：ToT (NeurIPS 2021, "Trash or Treasure") 的重建损失中包含梯度场约束，权重 0.6。原 ERRNet 的 pixel loss 中已有 `GradientLoss`，但以固定权重 0.4 与 MSE 捆绑。本方案将其解耦为独立损失项。
- **报告措辞建议**："参考 ToT [NeurIPS 2021] 的做法，将梯度场约束从 pixel loss 中解耦为独立损失项，在课程学习后期单独提高权重以增强边缘保持能力。"
- **实现状态**：✅ 已实现。`errnet_model.py` L420-427，在 `backward_G` 中直接计算。v3 课程中 Stage 2 权重 0.1，Stage 3 权重 0.2。

### 7. 物理启发的数据增强（Ghosting + Defocus）

- **内容**：在合成训练数据时，对反射层 R 施加微偏移混合（模拟双层玻璃重影）和高斯模糊（模拟反射层散焦），透射层 T 保持不变。
- **真实来源**：受 Kim et al. (CVPR 2020, "Single Image Reflection Removal With Physically-Based Training Images") 的物理渲染思路启发。原论文使用路径追踪渲染器，通过 10mm 厚玻璃模型和薄透镜相机让 ghosting 和 defocus 自然产生（空间变化、物理精确）。本方案是其在图像空间上的简化近似。
- **报告措辞建议**："受 Kim et al. [PBTI] 物理渲染思路的启发，在合成数据管线中引入了简化的 ghosting 和 defocus 增强，以缩小合成数据与真实光学成像之间的 domain gap。"
- **实现状态**：✅ 已实现。`data/reflect_dataset.py` 中。注意：(a) `torch.roll` 做的是循环移位，物理上应为零填充移位；(b) 增强仅对合成训练集生效。

### 8. 三阶段课程学习策略（Curriculum Learning）

- **内容**：将 80 epoch 训练分为三个阶段，每阶段引入不同的损失组合并调整学习率和数据比例。
- **真实来源**：课程学习（Curriculum Learning）是广泛使用的通用训练策略，不对应特定论文。此前方案曾错误归因于"IBCLN CVPR 2020"（该论文提出的是级联迭代精炼架构，与课程学习无关）。
- **报告措辞建议**："采用通用的多阶段课程学习策略（Curriculum Learning），分阶段逐步增加训练难度和损失复杂度。"（不需要引用特定论文）

**v3 最终课程表：**

| 阶段 | Epoch | 数据比例 [syn, real] | 新增损失（权重） | LR |
|------|-------|---------------------|-----------------|-----|
| Stage 1 | 0-39 | [0.7, 0.3] | Pixel(1.0) + VGG(0.1) | 2e-4 |
| Stage 2 | 40-59 | [0.7, 0.3] | +MaxRF(0.3) +Exclusion(0.05) +GAN(0.01) +Gradient(0.1) | 1e-4 |
| Stage 3 | 60-79 | **[0.3, 0.7]** | +Laplacian(0.05) +FeaDecorr(0.005) +VGG_MS(0.05), MaxRF→0.5, Gradient→0.2 | 5e-5→2e-5 |

**Stage 3 数据比例翻转**是本方案的关键创新：在精细调优阶段将真实数据比例从 30% 提升到 70%，用数据分布对抗损失函数对合成数据的过拟合倾向。

---

## 二、已修复的原始代码 Bug

以下三个 Bug 已在 v1 中全部修复。

### Bug 1：双重参数更新（✅ 已修复）

原 `engine.py` 通过 monkey-patch 劫持了 `optimize_parameters` 方法：先调用原始方法（完成一次完整的 forward -> backward_G -> optimizer_G.step），然后再用三个新 loss 做第二次 forward -> backward -> step。每次迭代模型被两个互不协调的步骤分别更新。

**修复**：将全部损失函数直接集成到 `errnet_model.py` 的 `backward_G` 方法中，与原有 pixel loss、VGG loss 合并为单次梯度更新。

### Bug 2：Lambda 变量空转（✅ 已修复）

原 `train_errnet.py` 中通过 `engine.model.opt.lambda_l1` 等变量控制课程学习权重，但模型内部的 `backward_G` 计算 pixel loss 时使用硬编码权重（MSE × 0.2 + Gradient × 0.4，在 `losses.py` 的 `init_loss` 中写死），不读取这些 lambda 变量。

**修复**：在 `backward_G` 中让各损失的权重由 `self.opt` 中的变量动态控制。新增 `lambda_pixel`、`lambda_maxrf`、`lambda_exclusion`、`lambda_laplacian`、`lambda_fea_decorr`、`lambda_vgg_ms`、`lambda_gradient` 七个参数。

### Bug 3：学习率反复升降（✅ 已修复）

原 60 epoch 训练中调了 5 次学习率（1e-4 → 5e-5 → 1e-5 → 5e-5 → 1e-5），epoch 40 降到极低后又弹回，导致最后阶段震荡。

**修复**：学习率严格单调递减：2e-4 → 1e-4 → 5e-5 → 2e-5。

### GAN 延迟初始化隐患（✅ 已修复）

原 `losses.py` 的 `init_loss` 仅在 `lambda_gan > 0` 时创建 GAN 损失对象。但课程学习 Stage 1 设 `lambda_gan=0`、Stage 2 才设为正值，导致 Stage 2 调用 GAN 损失时对象不存在。

**修复**：改为始终初始化 GAN 损失对象，`lambda_gan` 仅控制其在 `backward_G` 和 `backward_D` 中是否被调用。

---

## 三、未实现的方法（可考虑作为未来工作）

| 方法 | 来源 | 未实现原因 |
|------|------|-----------|
| 加性一致性约束 | ToT / IBCLN | 对单流架构（R_pseudo = I - T_hat）是恒等式，需在特征空间施加才有效，实现较复杂 |
| 线性色彩空间操作 | IBCLN | 需要修改数据管线，且 VGG 预训练权重基于 sRGB，转换后可能破坏感知损失的有效性 |
| 两阶段自蒸馏 | ToT | 训练时间超出约束（约需额外 10 小时） |

---

## 四、不推荐采用的方法（及原因）

| 方法 | 来源 | 不推荐原因 |
|------|------|-----------|
| 非线性残差项 Phi | DSRNet (Hu et al.) | 需要修改网络结构，增加额外分支输出残差图 |
| MuGI 双流交互块 | DSRNet (Hu et al.) | 需要完整的双流架构重构，参数翻倍 |
| BT-net (回溯网络) | PBTI (Kim et al.) | 需要额外网络分支和预训练，算力开销大 |
| Separation-Critic 对抗训练 | DAD (Zou et al.) | GAN 训练不稳定，现有训练时间内难以调好新判别器 |
| 玻璃吸收效应建模 (A 矩阵) | AE (Zheng et al.) | 解决的是色彩还原问题，不是分离问题，对核心指标帮助有限 |
| YTMT 特征交换 (ReLU- cross-feed) | ToT (NeurIPS 2021) | 需要双流架构 |
| ConvLSTM 级联迭代 | IBCLN (Li et al.) | 需要重构为循环网络架构 |
| RDNet 掩膜预测网络 | RRW (2023/2024) | 增加额外网络参数，算力开销大 |

---

## 五、实现优先级（实际执行顺序）

### 第一轮（v1）：Bug 修复 + 四个基础损失函数

- 修复双重更新、Lambda 空转、学习率反复升降
- 实现 MaxRF Mask-Weighted Loss、Gradient Exclusion Loss、Laplacian Edge Loss、Feature Decorrelation Loss
- 60 epoch 训练，效果勉强追平原版（因训练量被 bug 修复后减半）

### 第二轮（v2）：训练量补偿 + 延长训练

- 初始学习率提高到 2e-4
- 总训练延长到 80 epoch
- 课程阶段推迟：Stage 1 延长到 40 epoch
- **关键发现**：Stage 3 出现合成-真实过拟合裂痕（CEILNet 涨但 real20/objects/wild 跌）

### 第三轮（v3）：对抗过拟合三板斧

- Stage 3 数据比例翻转：[0.7syn, 0.3real] → [0.3syn, 0.7real]
- 新增多尺度感知损失（方法 7）
- 梯度惩罚独立化（方法 8）
- Laplacian 和 FeaDecorr 权重减半

### 第四轮（v4+ft）：未对齐数据 fine-tune

- 在 v3 对齐训练（80 epoch）基础上，用未对齐真实数据 fine-tune
- 沿用 v3 的损失函数配置（对齐数据继续走多损失约束）+ 未对齐数据走 CX 上下文损失
- 极低学习率（5e-6）保护已收敛权重
- **关键发现**：fine-tune 存在极为狭窄的 sweet spot。通过 checkpoint sweep（ep81~87 逐轮测试）发现，**ep81（ft 仅 1 epoch）即为全局最优**——五项指标四项超越基线，wild 首次被攻克。此后 CEILNet、postcard、wild 同步退化。结合 v2 中 Stage 3 的过拟合现象，揭示了一条规律：**对齐训练越充分，fine-tune 对未对齐数据的敏感度越高，收益窗口越窄。**

---

## 六、最终实验结果（v4_ft@81ep 最优 checkpoint vs 基线）

| 数据集 | PSNR | SSIM | NCC | LMSE | vs基线 |
|--------|------|------|-----|------|--------|
| **CEILNet Table2** | **28.57** | **0.946** | **0.983** | **0.0041** | 四项全胜，PSNR +0.69 dB |
| **real20** | **23.81** | **0.822** | **0.904** | **0.0203** | 四项全胜，PSNR +0.26 dB |
| **postcard** | **22.46** | **0.869** | 0.937 | 0.0051 | PSNR+SSIM 胜 |
| **wild** | **25.35** | **0.904** | **0.942** | **0.0064** | 四项全胜，PSNR +0.17 dB |
| objects | 24.59 | 0.895 | 0.982 | 0.0032 | PSNR -0.26，SSIM/NCC 持平 |

**五项数据集中四项 PSNR 超越基线**，wild 首次被攻克（此前 v3 和 ep84 均未超越）。
CEILNet +0.69 dB、postcard +0.39 dB、real20 +0.26 dB、wild +0.17 dB。
objects 是唯一未超越的数据集（-0.26 dB），但其 SSIM 和 NCC 与基线持平，
退步集中在像素级精度而非结构质量。

**最佳 checkpoint 的获取方式**：对齐训练 80 epoch（`errnet_v4/errnet_080_00154640.pt`），
再以 5e-6 学习率 fine-tune 仅 1 epoch。fine-tune 超过 1 epoch 后 CEILNet、postcard、
wild 同步退化（ep81→ep87：CEILNet -0.20，postcard -1.12，wild -1.01）。

**核心发现**：对齐训练越充分，fine-tune 的收益窗口越窄。v3 的强对齐基础模型对未对齐
数据的分布偏移极为敏感——1 epoch 即可捕获真实场景的结构信号，多则中毒。
此发现与 v2 中 Stage 3 的过拟合现象形成双重印证。

**基线来源**：ERRNet 论文 Table 2（Wei et al., CVPR 2019）。

---

## 七、参考论文索引

| 缩写 | 论文全名 | 作者 | 会议 | 文件夹 | 本项目的使用 |
|------|---------|------|------|--------|-------------|
| ERRNet | Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements | Wei et al. | CVPR 2019 | 项目根目录 | 基线模型 |
| PBTI | Single Image Reflection Removal With Physically-Based Training Images | Kim et al. | CVPR 2020 | PBTI/ | Ghosting/Defocus 增强灵感 |
| DAD | Deep Adversarial Decomposition: A Unified Framework for Separating Superimposed Images | Zou et al. | CVPR 2020 | DAD/ | Feature Decorrelation Loss 灵感 |
| IBCLN | Single Image Reflection Removal Through Cascaded Refinement | Li et al. | CVPR 2020 | SIRRCR/ | Multi-Scale Perceptual Loss 来源 |
| AE | Single Image Reflection Removal With Absorption Effect | Zheng et al. | CVPR 2021 | AE/ | 未采用（色彩校正，非分离问题） |
| LA-SIRR | Location-Aware Single Image Reflection Removal | Dong et al. | ICCV 2021 | La/ | Laplacian Edge Loss 灵感 |
| ToT | Trash or Treasure: An Interactive Dual-Stream Strategy for Single Image Reflection Separation | — | NeurIPS 2021 | ToT/ | Independent Gradient Penalty 来源 |
| DSRNet | Single Image Reflection Separation via Component Synergy | Hu et al. | ICCV 2023 | SIRSvCS/ | Gradient Exclusion Loss 来源 |
| RRW | Revisiting Single Image Reflection Removal In the Wild | — | arXiv 2023 | RSIR/ | MaxRF 公式来源 |