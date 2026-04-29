# HunyuanImage 3.0 Technical Report

> **Authors:** Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weijie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyang Peng, Yuanbo Peng, Xiangwei Shen, Yixuan Shi, Jiale Tao, Yangyu Tao, Qi Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Qixun Wang, Weiyan Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fang Yang, Xiaofeng Yang, Xuan Yang, Zhantao Yang, Jingmiao Yu, Zheng Yuan, Chao Zhang, Jian-Wei Zhang, Peizhen Zhang, Shi-Xue Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zijian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jianchen Zhu, Zhao Zhong 等
> **Affiliation:** Tencent
> **Venue:** arXiv:2509.23951, September 2025 (revised February 2026)
> **Link:** [https://arxiv.org/abs/2509.23951](https://arxiv.org/abs/2509.23951)

---

## TL;DR

HunyuanImage 3.0 是腾讯推出的**原生多模态自回归模型**，统一了图像理解与图像生成能力。其核心是一个基于 MoE（Mixture-of-Experts）架构的解码器 LLM，总参数量超过 **80B**，每 token 推理时激活约 **13B** 参数。模型通过精心的数据管理（近 50 亿图像）、原生 Chain-of-Thought 推理模式、四阶段渐进式预训练、以及包括 SFT/DPO/MixGRPO/SRPO/ReDA 在内的多层次后训练策略，在文本-图像对齐和视觉质量方面达到与顶级闭源模型（如 GPT-Image）竞争的水平。该模型是当时最大的开源图像生成模型。

---

## 研究背景与动机

### 问题定义

近年来，基于扩散模型和 Transformer 的图像生成取得了飞速进展，但**主流领先系统大多为闭源**。这限制了学术界和产业界对前沿技术的复现与深入研究。HunyuanImage 3.0 旨在提供一个与闭源模型竞争的开源替代方案。

### 现实重要性

- 开源社区缺乏一个真正能与 GPT-Image、Seedream 等闭源系统竞争的大规模图像生成模型
- 将图像理解与生成统一到一个自回归框架中，是多模态大模型的重要发展方向
- 原生 CoT 推理能力使模型在处理复杂 prompt 时具有更强的语义理解和组合控制能力

### 核心贡献

1. **数据工程**：从 100 亿+ 原始图像中筛选近 50 亿高质量训练图像 + 1 亿+ 图像对数据
2. **模型架构**：基于 MoE LLM 的统一多模态架构，融合自回归文本建模和扩散式图像 token 建模
3. **原生 CoT 模式**：将链式推理能力原生嵌入图像生成流程
4. **渐进式训练**：四阶段预训练 + 五种后训练策略
5. **高效推理**：基于 MeanFlow 的蒸馏方案，将采样步数降至 4-8 步
6. **新评测体系**：提出 SSAE 结构化语义对齐评测基准

---

## 数据准备

### 数据过滤（三阶段流水线）

从 **100 亿+** 原始图像出发，保留率 < 45%，最终获得近 **50 亿** 高质量图像。

| 阶段 | 操作 | 关键策略 |
|------|------|----------|
| **Stage 1：技术清洗** | 去除低分辨率（<512px）、损坏文件、曝光异常、过饱和、MD5 精确去重 | 基础质量保障 |
| **Stage 2：主体筛选** | 客观过滤器 + 主体评分 | 水印/Logo/文字/拼贴/边框/AIGC 检测；清晰度模型；美学模型（色彩/光影/构图） |
| **Stage 3：精细去重与补充** | 基于 embedding 聚类的去重 + 专项数据集补充 | 知识增强、文字相关、风格化、平面设计等专项数据 |

**AIGC 污染问题**：论文特别指出 AI 生成图像的污染会扭曲自然数据分布、损害收敛。策略包括：自动 AIGC 检测模型 + 移除高 AIGC 比例的整体数据源。

### 多图/交错数据集

- **规模**：1 亿+ 图像对和多图样本
- **来源**：
  - **图像聚类管线**：从 20 亿+ 图像聚类中选取具有编辑关系的图像对，经图像关系判别和复杂度过滤
  - **视频挖掘管线**：镜头边界检测 → 统一场景过滤 → 相机运动分类 → 关键帧提取 → 运动模糊过滤

### 图像描述（Captioning）

采用**双语层级结构化 schema**：

| 维度 | 内容 |
|------|------|
| **描述层级** | 短摘要 → 长描述 → 详尽前景/背景描绘（4 个细节等级） |
| **风格属性** | 艺术风格、镜头类型、光照、氛围、构图 |
| **事实实体** | 人物、地标、品牌、艺术品等命名实体/IP |

**关键技术**：
- **组合式描述合成**：动态采样和组合 schema 字段，生成 30~1000 词的双语描述，提高泛化能力
- **事实锚定**：OCR 代理 + 命名实体/IP 代理，提供辅助输入
- **双向验证循环**：生成描述与检测实体交叉验证，仅保留通过验证的样本
- **图像差异描述**：对配对图像生成编辑指令式的差异描述

### 推理数据集构建

论文认为模型具有潜在的推理和语义理解能力，可通过少量专门数据微调激发。构建三类 CoT 数据：

| 类型 | 目的 | 内容 |
|------|------|------|
| **T2T（文本到文本）** | 提升指令跟随与逻辑推理 | 写实生成、艺术渲染、UI/海报设计、知识驱动查询、科学可视化等领域的 prompt |
| **T2TI（文本到文本+图像）** | 提升端到端文本推理与视觉保真度 | 高质量类别平衡图像 + 短/长描述 + 推理 trace |
| **TI2TI（文本&图像到文本&图像）** | 提升复杂图像编辑 | 源图像 + 编辑指令 + 真值编辑图像 + 详细编辑 trace（将复杂请求分解为原子步骤） |

---

## 模型架构

### 统一建模方式

采用**离散-连续混合策略**：
- **文本 token**：自回归 next-token prediction
- **图像 token**：扩散式预测（Diffusion-based prediction）

类似于 Transfusion 和 JanusFlow 的统一思路，在一个模型中同时支持语言建模、图像理解和图像生成。

![HunyuanImage 3.0 Architecture](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/hunyuan3_fig3_architecture.png)
> **Figure 3：HunyuanImage 3.0 统一架构总览。** 三种任务模式共享同一个 Decoder-Only Transformer（Hunyuan-A13B）主干：
> - **左（Image Understanding）**：条件图像通过 Gen. Encoder（VAE）和 Und. Encoder（ViT）双路径编码，与文本 prompt 一同输入 Transformer，通过 next-token prediction 输出理解结果
> - **中（Language Modeling）**：纯文本输入，标准自回归语言建模
> - **右（Image Generation）**：文本 prompt 经 Text Tokenizer 编码，噪声图像经 Gen. Encoder 编码，Transformer 预测扩散速度场（Velocity / diffusion prediction），再由 Gen. Decoder（VAE Decoder）解码为最终图像

### 主干网络（Backbone）

| 属性 | 规格 |
|------|------|
| 基础模型 | Hunyuan-A13B |
| 架构 | Decoder-only LLM + MoE |
| 总参数量 | **80B+** |
| 专家数量 | 64 个专家 + 1 个共享 MLP |
| 每 token 激活专家数 | 8 |
| 每 token 激活参数量 | **~13B** |

### 图像编码设置

#### 生成用 VAE

| 属性 | 规格 |
|------|------|
| 潜在空间维度 | 32 维 |
| 下采样因子 | **16x** |

论文指出，相比先前 DiT 系统常用的 8x 下采样 VAE + 2x patchification，单一 16x 下采样 VAE 更简洁且生成质量更好。

#### 条件图像的双编码器策略

对于条件图像输入，拼接两种特征：
- **VAE 潜在特征**
- **视觉编码器（ViT）特征**

这种统一方法在单个序列中支持：交错对话、图像生成、图像理解、图像编辑，避免在不同管线间切换。

### 投影模块（Projectors）

| 特征来源 | 投影方式 |
|----------|----------|
| VAE 特征 | 时间步调制残差块（Timestep-modulated residual block） |
| ViT 特征 | 两层 MLP |

同时在序列中插入**时间步嵌入（Timestep embeddings）**来调节扩散过程。

### 广义因果注意力（Generalized Causal Attention）

结合了文本的因果注意力和图像段内的全注意力：

- **文本 token**：仅关注先前的多模态 token（标准因果注意力）
- **图像 token**：关注所有先前的多模态 token + 同一图像段内的所有图像 token（全局段内注意力）

![Generalized Causal Attention](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/hunyuan3_fig4_attention.png)
> **Figure 4：两种注意力掩码实现。** 橙色区域表示 token 之间可以互相 attend 的区域。

**图 (a) — Case A：无生成图像或单个生成图像**

序列结构为 `[Text → Gen Image → Cond Image → Text]`，注意力掩码规则如下：

| Token 类型 | 可关注范围 | 图中对应 |
|-----------|-----------|---------|
| **Text（前段）** | 仅前面的 text token（标准因果三角） | 左上角橙色阶梯 |
| **Gen Image** | 前面所有 text token + **自身整个图像段**（段内全注意力） | 绿色方框区域——注意生成图像区域形成完整矩形块，表示段内任意两个 image token 均可互相 attend |
| **Cond Image** | 前面所有 token + **自身整个图像段** | 蓝色方框区域——同样形成完整矩形块 |
| **Text（后段）** | 前面所有 text + 所有图像 token（标准因果） | 右下角继续沿对角线扩展 |

核心观察：图像段内部是**双向全注意力**（矩形填满），而跨段则遵循**因果方向**（只能看到前面的内容）。

**图 (b) — Case B：多个生成图像（训练专用）**

序列结构为 `[Text → Gen Image₁ → Cond Image → Text → Gen Image₂]`，关键差异：

- 第一个 **Gen Image₁** 生成后，后续的 Cond Image、Text、Gen Image₂ **都无法 attend 到 Gen Image₁ 区域**
- 这在下三角注意力矩阵中形成了一个明显的**空洞/白色区域**（Gen Image₁ 列对应的下方行为空白）
- 第二个 **Gen Image₂** 仍保持段内全注意力（右下角的完整矩形块）

设计原因：训练时一个序列中可能包含多个要生成的图像，但需要防止后面的图像"偷看"前面的生成结果，保持因果一致性。

**推理行为**：推理时永远只有一个正在生成的图像，已生成图像转为条件图像（Cond Image），因此始终使用 Case A 的标准广义因果注意力掩码，不存在空洞。

### 位置编码（Position Embedding）

引入**广义 2D RoPE**：

- **文本 token**：使用标准 1D RoPE
- **图像 token**：将 1D 位置重塑为 2D 坐标 (x, y)，使用各向异性正弦函数编码

![1D RoPE vs Generalized 2D RoPE](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/hunyuan3_fig5_rope.png)
> **Figure 5：1D RoPE 与广义 2D RoPE 的对比。**

**左图（1D Positions）**：传统 1D RoPE 下，所有 token（文本 + 图像）排列在一维序列上，位置编号为 1, 2, 3, ..., 11。黄色为文本 token，蓝色为图像 token，它们在同一条直线上按序编号。

**右图（2D Positions）**：广义 2D RoPE 下的位置分配策略：
- **文本 token（黄色）** 保持在对角线上的 1D 位置——位置 (1,1), (2,2), (3,3) 和 (10,10), (11,11)，等价于标准 1D RoPE
- **图像 token（蓝色）** 被重新映射到 2D 网格——占据位置 (5,6) ~ (8,7) 的矩形区域，反映图像的空间二维结构

**兼容性保证**：当序列中没有图像 token 时，所有位置退化为对角线上的 1D 编号，与预训练 LLM 的标准 RoPE 完全一致，实现**向后兼容**。

**多生成图像的位置偏移**：训练中出现多个生成图像时，对后续 token 的位置编码进行偏移，以对齐训练和推理的位置一致性。

### 自动分辨率（Automatic Resolution）

不同于需要用户指定图像尺寸的 DiT 系统，HunyuanImage 3.0 支持**自动模式**：

- **尺寸 token**：`<img_size_256>`, `<img_size_512>`, `<img_size_768>`, ...
- **比例 token**：`<img_ratio_0>` ~ `<img_ratio_32>`（覆盖 1:4 到 4:1）

模型从用户 prompt 和上下文自动预测形状 token，用户也可以通过数值比例或"竖版"等关键词给出提示。

---

## 模型训练

### 预训练（四阶段渐进式）

多任务训练框架涵盖：T2I、LM（语言建模）、MMU（多模态理解）、INTL（交错）、CoT

| 阶段 | VAE 分辨率 | ViT 分辨率 | 可训练部分 | 任务 | 目标 |
|------|-----------|-----------|-----------|------|------|
| **I** | 256px | 512px | Transformer | T2I, LM, MMU | 对齐文本-图像潜在表示 |
| **II** | 256px | 512px | ViT | MMU | 提升视觉理解 |
| **III** | 512px | 512px | ViT + Transformer | T2I, LM, MMU, INTL | 强化多模态建模 |
| **IV** | 1024px | 512px | ViT + Transformer | T2I, LM, MMU, INTL, CoT | 高分辨率生成 + 推理能力 |

**关键策略**：
- 数据从粗到精过滤
- VAE 图像分辨率逐步提升，ViT 分辨率固定
- 训练过程中保持图像长宽比
- Stage IV 加入推理数据，启用 CoT 多模态建模

预训练后进行**指令微调**，使用 T2I/LM/CoT 模板格式化的数据。

### 后训练（五阶段策略栈）

| 方法 | 目标 | 核心机制 |
|------|------|----------|
| **SFT** | 高质量生成基础 | 精选景观/人像/OCR 数据集 + 编辑数据集 + 推理数据，多阶段训练 |
| **DPO** | 减少结构失真和伪影 | 两阶段：Stage 1 大量配对数据提升稳定性，Stage 2 精选高质量子集提升真实感 |
| **MixGRPO** | 在线 RL 优化多目标 | 扩展 GRPO 至 flow-based 模型，混合 ODE-SDE 采样，使用美学/风格/构图/光照/畸变/伪影等多奖励模型 |
| **SRPO** | 真实感与美学的梯度引导 RL | 向潜在特征注入噪声先验 → 单步去噪 → 优化去噪轨迹早期区间，使用可微奖励信号 |
| **ReDA** | 奖励分布对齐 | 任务特定投影器将生成映射到压缩空间，支持身份一致性、真实感等指标的针对性优化 |

**MixGRPO 详细说明**：
- 针对美学、风格、构图、光照、畸变、伪影等多维度使用专有奖励模型
- 对人脸 ID 保持使用专门的奖励模型
- 平衡多任务联合训练
- 改进的优势估计加速收敛

**SRPO 详细说明**：
- 目标：人类偏好对齐，减少过饱和，更连贯的光照和色彩，更好的皮肤纹理
- 机制：在去噪轨迹的早期区间进行优化（模型灵活度最大的区间）

**ReDA 详细说明**：
- 当存在参考图像时，比较从参考到生成的**过渡向量**而非静态特征
- 更好的训练效率和数据扩展性

### 蒸馏

基于 **MeanFlow** 框架将蒸馏扩展到 80B 统一多模态模型：
- 缓解训练不稳定性
- 在 MeanFlow 目标中添加轨迹分布对齐
- 将采样步数（NFE）降至 **4-8 步**，同时保持竞争力质量

---

## 评估与结果

### SSAE（结构化语义对齐评测）

论文认为现有 T2I 基准（如 T2I-CompBench、GenEval）存在两大缺陷：
1. 过于简单的 prompt，无法反映真实用户指令
2. 严重依赖 CLIP score 等自动指标，可能偏离人类判断

**SSAE 设计**：
- 500 条多样化 prompt，提取 3,500 个关键点
- 12 个细粒度语义维度：名词、主/次主体属性与动作、场景名词、场景属性、镜头、风格、构图等
- 基于 LLM 的结构化语义点解析 + 人工校正
- 使用先进 MLLM + CoT 推理进行 0/1 评分
- 指标：字段准确率、平均图像准确率（Mean Image Accuracy）、全局准确率（Global Accuracy）

**结果**：HunyuanImage 3.0 在 SSAE 所有细粒度维度上与领先模型持平。

### GSB（人类评测）

采用成对比较的人类评测方法：
- 1,000 条 prompt，场景平衡
- 单次生成，无 cherry-picking
- 所有对比模型使用默认设置
- 100+ 专业评估员

**相对胜率**：

| 对比模型 | HunyuanImage 3.0 净胜率 |
|----------|------------------------|
| HunyuanImage 2.1 | **+14.10%** |
| GPT-Image | **+5.00%** |
| Nano Banana | **+2.64%** |
| Seedream 4.0 | **+1.17%** |

**结论**：显著超越此前最佳开源模型（HunyuanImage 2.1），与顶级闭源商业模型竞争力相当。

### 专家激活分析

对 MoE 层的模态专门化进行分析：
- 随着网络深度增加，KL 散度上升，图像 token 和文本 token 的专家激活分布愈发分化
- 专家逐渐专门化为图像 vs 文本处理
- 这表明 MoE 可能通过将模态职责分配给专门化的专家来改善多模态学习

---

## 关键技术总结

### 模型规格一览

| 维度 | 规格 |
|------|------|
| 架构类型 | 原生多模态自回归 + 扩散式图像建模 |
| 骨干网络 | Hunyuan-A13B (Decoder-only MoE LLM) |
| 总参数量 | 80B+ |
| 激活参数量 | ~13B / token |
| MoE 配置 | 64 专家 + 1 共享 MLP，每 token 激活 8 专家 |
| VAE | 32 维潜在空间，16x 下采样 |
| 注意力 | 文本因果 + 图像段内全注意力 |
| 位置编码 | 广义 2D RoPE（兼容 1D RoPE） |
| 训练数据 | ~50 亿图像 + 1 亿+ 图像对 |
| 推理加速 | MeanFlow 蒸馏至 4-8 NFE |

### 创新点排序

1. **统一架构**：将 MoE LLM 扩展为同时支持图像理解和生成的原生多模态模型
2. **数据工程**：从 100 亿级原始数据构建高质量训练集的系统化流水线
3. **原生 CoT**：将链式推理直接嵌入图像生成过程
4. **后训练策略栈**：SFT → DPO → MixGRPO → SRPO → ReDA 的逐层精炼
5. **广义因果注意力**：统一文本因果和图像全注意力的灵活机制
6. **自动分辨率**：模型自主预测最佳图像尺寸和比例

---

## 局限性与展望

- **当前开源版本仅包含文本到图像**功能，图像到图像（编辑）的训练仍在进行中，计划后续释放
- 论文未提供独立的消融实验部分，缺少对各组件贡献的定量分析
- 80B+ 参数量对部署硬件要求较高，即使激活参数为 13B
- SSAE 评测结果缺少具体数值表格

---

## 个人评价

### 优势

- **工程完整度极高**：从数据清洗到模型设计到后训练到推理加速，形成了完整的技术链路
- **开源意义重大**：作为最大的开源图像生成模型，填补了社区在大规模多模态生成模型上的空白
- **后训练策略丰富**：五种后训练方法各有侧重，形成了系统化的质量提升管道
- **统一架构设计优雅**：广义因果注意力和 2D RoPE 的设计在保持兼容性的同时支持多模态

### 不足

- **消融实验缺失**：无法量化评估各组件（CoT、MoE、后训练策略等）的独立贡献
- **评测自建为主**：SSAE 和 GSB 均为自建基准，外部可比性有限
- **技术报告性质**：更偏向系统描述，对设计选择的分析深度不足
- **编辑能力未开源**：当前最具吸引力的图像编辑能力尚未在开源版本中体现
