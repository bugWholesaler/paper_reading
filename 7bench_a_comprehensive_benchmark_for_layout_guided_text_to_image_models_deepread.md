# 7Bench:面向布局引导文生图模型的综合性基准

> **作者:** Elena Izzo\*、Luca Parolari\*、Davide Vezzaro\*、Lamberto Ballan(\*共同一作)—— 帕多瓦大学 & Brain Technologies srl,意大利
> **会议:** ICIAP 2025(International Conference on Image Analysis and Processing)
> **链接:** [arXiv:2508.12919](https://arxiv.org/abs/2508.12919)
> **代码 / 权重 / 数据:** 代码与基准 ✅ [github.com/Elizzo/7Bench](https://github.com/Elizzo/7Bench) · 权重 ❌(使用现成模型)· 数据 ✅(公开 224 条 prompt/bbox 对)

---

## 一句话总结

7Bench 是**第一个同时评测语义(文本)对齐与空间(布局)对齐**的布局引导文生图(layout-guided T2I)模型基准。它贡献了 224 条精心模板化的「文本 + 边界框」配对,均匀覆盖 **7 个挑战性场景**(物体绑定、小框、重叠框、颜色绑定、属性绑定、物体关系、复杂组合),并提出**双指标协议**:基于 TIFA 的文本对齐分,以及一个全新的**布局对齐分**(在多个 IoU 阈值上对 accuracy@k 曲线取 AUC,用 OWL-ViT 开放词表检测器计算)。在 GLIGEN 与 3 个免训练方法上的评测显示:文本对齐分落在 0.55–0.9,而**布局对齐分从未超过 ~0.5**,且小边界框是所有模型最难的场景。

---

## 1. 研究背景与动机

### 1.1 问题定义

标准的文生图(T2I)扩散模型只接受文本 prompt。**布局引导**的 T2I 模型额外引入第二个条件信号——通常是一组**边界框**,prompt 里提到的每个物体对应一个框——从而让用户不仅能指定「画什么」,还能指定每个物体「画在哪里」。因此这类模型的质量天然有**两个正交维度**:

1. **文本(语义)对齐** —— 生成的图里是否包含正确的物体、正确的颜色/属性/关系?
2. **布局(空间)对齐** —— 每个物体是否真的落在用户指定的框内?

本文的核心论点是:学界对维度 1 的衡量尚算合理,但**没有任何公认、有原则的方法去同时衡量维度 2**。

### 1.2 为什么重要

布局引导生成正越来越多地被用作下游计算机视觉任务的**合成数据引擎**(例如指代表达理解的合成数据集 —— [Parolari et al., ICPR 2024](https://arxiv.org/abs/2311.04372);实例级数据增强 —— [Kupyn & Rupprecht, ECCV 2024](https://arxiv.org/abs/2406.08249))。如果你用框来生成训练图,但物体并没有真正落在框里,**那你自动生成的标注就是错的**——布局本身就是标签。空间误差会悄悄注入标签噪声,污染任何基于这些合成数据训练的模型。所以空间保真度不是一个表面问题,而是数据质量问题。

### 1.3 既有工作的局限

作者点名了当前评测格局的具体缺陷:

- **纯文本基准忽略布局。** HRS-Bench([Bakr et al., ICCV 2023](https://arxiv.org/abs/2304.05390))、TIFA([Hu et al., ICCV 2023](https://arxiv.org/abs/2303.11897))、T2I-CompBench([Huang et al., NeurIPS 2023](https://arxiv.org/abs/2307.06350))、ConceptMix([Wu et al., NeurIPS 2024](https://arxiv.org/abs/2408.14339))都测颜色/属性/关系/计数,但**既不提供输入布局,也没有布局指标**。
- **布局评测各行其是、不可比。** 研究者各自从 COCO2014([Lin et al., ECCV 2014](https://arxiv.org/abs/1405.0312))或 Flickr30K Entities([Plummer et al., ICCV 2015](https://arxiv.org/abs/1505.04870))随机抽一个子集,报告 FID / YOLO score——但子集大小不一致:GLIGEN([Li et al., CVPR 2023](https://arxiv.org/abs/2301.07093))生成 **30k** 个样本,而 Attention Refocusing([Phung et al., CVPR 2024](https://arxiv.org/abs/2306.05427))只用 **5k**。
- **COCO 数据污染。** BoxDiff([Xie et al., ICCV 2023](https://arxiv.org/abs/2307.10816))明确警告:在 COCO 上评测是泄漏的,因为 COCO 同时也是这些模型的*训练*集。
- **唯一的空间基准又忽略语义。** [Cho et al., CVPRW 2024](https://arxiv.org/abs/2304.06671) 的诊断基准测空间技能(位置/尺寸/形状),但**忽略了颜色、属性、关系**这个语义维度。

### 1.4 本文填补的空白

没有任何基准在**单一、固定、可复现的协议**下,用一个**均衡、无污染**的 prompt 集**同时**衡量文本与布局对齐。7Bench 正是为填补这个空白而生:均衡的 7 场景覆盖语义与空间双重挑战,加上一个能嵌入既有文本对齐框架(TIFA)的布局对齐指标。

---

## 2. 相关工作

### 2.1 文生图基准

扩散模型([Ho et al., DDPM, NeurIPS 2020](https://arxiv.org/abs/2006.11239))在质量与多样性上超越了基于 GAN 的文生图([Reed et al., ICML 2016](https://arxiv.org/abs/1605.05396)),但仍然无法**把属性绑定到正确的物体上**(灾难性遗忘、属性泄漏 —— 由 TIAM 诊断,[Grimal et al., WACV 2024](https://arxiv.org/abs/2307.05134))。随后涌现一批保真度基准(TIFA、T2I-CompBench、HRS-Bench、ConceptMix、Structured Diffusion —— [Feng et al., ICLR 2023](https://arxiv.org/abs/2212.05032))。**空白:** 它们都不提供输入布局,也没有布局指标。

### 2.2 布局引导文生图扩散模型

两大流派:
- **训练型 / 微调型** —— 通过新的可训练层或 token 注入 grounding 信息:GLIGEN([Li et al., CVPR 2023](https://arxiv.org/abs/2301.07093))加入门控自注意力层;ReCo([Yang et al., CVPR 2023](https://arxiv.org/abs/2211.15518))加入位置 token。
- **免训练型** —— 在推理时操纵交叉注意力:BoxDiff([Xie et al., ICCV 2023](https://arxiv.org/abs/2307.10816))、Cross-Attention Guidance([Chen et al., WACV 2024](https://arxiv.org/abs/2304.03373))、Attention Refocusing([Phung et al., CVPR 2024](https://arxiv.org/abs/2306.05427))。

### 2.3 本文定位

布局引导模型的评测是一种**两步、分开评估**的做法(文本用 T2I 基准;布局用 COCO/Flickr 子集配 FID/YOLO),既没有共享协议,样本数也不一致,还存在 COCO 污染。7Bench 把这两步统一进一个固定协议、一个均衡且专门构建的 prompt 集中。

---

## 3. 核心方法

注:7Bench 是一篇**基准 + 协议**论文,不是模型论文。它的「方法」是 (A) 基准构建方法论,与 (B) 双指标评测协议。我把这两者当作两个核心模块,分别套用「方法精读清单」。(完整的 prompt/bbox 构建流程在 §4 数据构建中进一步展开。)

![7Bench 总览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig1_overview.png)

*图 1 —— 7Bench 一览。左侧:指令集(Instruction Set),一个 7 楔形的场景轮盘(物体绑定、小框、重叠框、颜色绑定、属性绑定、物体关系、复杂组合),每个楔形配有示例 prompt + 彩色布局框。中部:prompt + 布局被喂给布局引导扩散模型,产出生成图集合(Evaluation Set)。右侧:评测协议 —— 文本对齐分支(出题 → VQA → 与标准答案比对 → 文本对齐分)与布局对齐分支(零样本目标检测器 → 把检测框与目标框比对 → 布局对齐分)。*

### 3.1 模块 A —— Prompt 与布局构建(7 个场景)

**目的与定位。** 该模块产出基准的*输入*:224 对(prompt,框集合)。它是每个被测模型消费的上游产物;它的设计正是基准「综合」「均衡」的来源。

**输入与输出。** 输入:一套受控词表——物体集 $\mathcal{O}$、属性集 $\mathcal{A}$、颜色集 $\mathcal{C}$、关系集 $\mathcal{R}$,外加手工绘制的布局。输出:224 个样本 = **每场景 32 个 × 7 个场景**,每个样本 = 一条文本 prompt + 一组 $N\in\{1,2,3,4\}$ 个边界框。框坐标是**归一化**的:$bbox_i=[x^i_{min},y^i_{min},x^i_{max},y^i_{max}]$,满足 $0\le x_{min}<x_{max}\le1$ 且 $0\le y_{min}<y_{max}\le1$。

**构建形式化(prompt 模板)。** Prompt 由一套植根于*解耦表示*理论([Trager et al., ICCV 2023](https://arxiv.org/abs/2302.14383))的模板生成。对一条 $N=2$ 物体的 prompt,通用模板为:

$$t = \text{“}\,det(\mathcal{o}_1,\mathcal{a}_1)\ \mathcal{a}_1\ \mathcal{o}_1\ rel(\textit{and},\mathcal{r}_{12})\ det(\mathcal{o}_2,\mathcal{a}_2)\ \mathcal{a}_2\ \mathcal{o}_2\,\text{”}$$

通俗解释:每个物体槽 $\mathcal{o}_i$ 从物体集 $\mathcal{O}$ 中取;它可被属性 $\mathcal{a}_i\in\mathcal{A}$ 修饰;相邻物体之间或用简单连词 *"and"*,或用空间关系 $\mathcal{r}_{ij}\in\mathcal{R}$ 连接;$det(\cdot)$ 根据后续词选择语法正确的冠词("a"/"an")。$N$ 在 $\{1,2,3,4\}$ 上变化,使基准从单物体一直压力测试到四物体场景(物体数分布见图 2)。

**七个场景(每个都是「模板特化 + 布局约束」):**

1. **物体绑定(Object Binding)** —— 考察**灾难性遗忘**与定位。无属性、无关系,只有 $N\in\{1,2,3,4\}$ 个裸物体。模板:
   $$t_{obj}=\text{“}det(\mathcal{o}_1)\,\mathcal{o}_1,\ det(\mathcal{o}_2)\,\mathcal{o}_2,\ det(\mathcal{o}_3)\,\mathcal{o}_3\ \text{and}\ det(\mathcal{o}_4)\,\mathcal{o}_4\text{”}$$
2. **小框(Small Bboxes)** —— 文本同 $t_{obj}$,但每个框的面积被约束在**图像面积的 3%–10%**。形式上,框面积 $A_i=(x^i_{max}-x^i_{min})(y^i_{max}-y^i_{min})$ 被随机取值使 $0.03\,A_{image}\le A_{image}\cdot A_i\le 0.10\,A_{image}$。小框在去噪时给的条件信号太弱 → 物体遗漏/错位([Patel & Serkh, WACV 2025](https://arxiv.org/abs/2405.14101))。
3. **重叠框(Overlapping Bboxes)** —— 文本同 $t_{obj}$,但**每个框至少与另一个框重叠**:$\forall i\,\exists j\neq i\ \text{s.t.}\ A_i\cap A_j>0$。考察前景/背景分层。
4. **颜色绑定(Color Binding)** —— 每个物体配一个颜色,取自 **Berlin–Kay 的 11 种普适基本色** $\mathcal{C}=\{$黑、蓝、棕、灰、绿、粉、紫、红、白、黄、橙$\}$([Berlin & Kay, 1991](https://www.ucpress.edu/book/9780520076358/basic-color-terms)):
   $$t_{color}=\text{“}det(\mathcal{c}_1)\,\mathcal{c}_1\,\mathcal{o}_1,\dots\ \text{and}\ det(\mathcal{c}_4)\,\mathcal{c}_4\,\mathcal{o}_4\text{”}$$
5. **属性绑定(Attribute Binding)** —— 把颜色绑定扩展到 **30 个属性**,横跨颜色/形状/材质/外观/尺寸:$\mathcal{A}=\{$aggressive, black, blue, bright, clean, crowded, dark, fast, fluffy, fuzzy, green, happy, large, pink, red, rotten, rough, shiny, short, silver, small, smooth, snowy, soft, tall, warm, white, wooden, yellow$\}$(论文称 30 个;正文列出 29 个)。$N\in\{1,2,3,4\}$,无关系:
   $$t_{attr}=\text{“}det(\mathcal{a}_1)\,\mathcal{a}_1\,\mathcal{o}_1,\dots\ \text{and}\ det(\mathcal{a}_4)\,\mathcal{a}_4\,\mathcal{o}_4\text{”}$$
6. **物体关系(Object Relationship)** —— 2 或 4 个物体由关系连接,无属性。**11 种关系**取自空间认知研究([Landau & Jackendoff, 1993](https://doi.org/10.1017/S0140525X00029733)):$\mathcal{R}=\{$above, below, beside, far from, near, next to, on, over, to the left of, to the right of, under$\}$:
   $$t_{rel}=\text{“}det(\mathcal{o}_1)\,\mathcal{r}_{12}\,det(\mathcal{o}_2)\,\mathcal{o}_2\ \text{and}\ det(\mathcal{o}_3)\,\mathcal{r}_{34}\,det(\mathcal{o}_4)\,\mathcal{o}_4\text{”}$$
7. **复杂组合(Complex Composition)** —— 「开放世界」综合压力测试,**把以上全部组合**:1–4 个物体,每个都带 ≥1 个属性(颜色/形状/材质/外观/尺寸)**且**与另一个物体有关系,框可以是小框(<10%)**且**可重叠。

**布局采集。** 除非场景另有规定(小框/重叠),框都是**手工绘制**,力求是该 prompt 下*真实、比例协调、不重叠*的布局。

**使用的外部工具。** 构建阶段无 —— prompt 来自受控模板,框由人工标注。(外部模型在评测模块 §3.2 才登场。)

**设计选择与替代方案。** 刻意采用**全合成的模板生成 prompt + 手工绘框**(而非从 COCO/Flickr 采样),是对 BoxDiff 提出的 COCO 污染批评的直接回应:专门构建的集合不可能从模型训练数据里泄漏。严格平衡到**每场景 32 个**则避免任何单一技能主导总分。

### 3.2 模块 B —— 评测协议(两个互补分数)

**目的与定位。** 消费生成图(每个样本 16 张,见 §5.1),为每张图输出两个 $[0,1]$ 区间的标量:$s_{text}$ 与 $s_{layout}$。两者是*互补*的,不合并成单一数字 —— 论文坚持分开报告,因为一个模型可能在一个维度满分、另一个维度崩溃。

#### 3.2.1 文本保真度 —— $s_{text}$(TIFA)

$s_{text}\in[0,1]$ 即 **TIFA 分数**([Hu et al., ICCV 2023](https://arxiv.org/abs/2303.11897))。流程:

1. 一个 **LLM** 读 prompt 并生成一组关于它的问答对(例如 prompt "a red car" → 问:"图里有车吗?" 答:"是";问:"车是什么颜色?" 答:"红色")。从文本本身出题让该指标**独立于图像生成器**。
2. 一个 **VQA 模型**看着*生成*的图回答这些问题。
3. $s_{text}$ = VQA 答案相对 LLM 标准答案的**准确率**。越高代表语义保真度越好。

论文使用**公开的 TIFA 权重**(`github.com/Yushi-Hu/tifa`)。具体的 LLM/VQA backbone 未再说明 —— *论文中除「预训练 TIFA 权重」外未指明。*

#### 3.2.2 布局保真度 —— $s_{layout}$(本文新贡献)

$s_{layout}\in[0,1]$ = **accuracy@k 曲线下面积(AUC)**,曲线在 IoU 阈值上扫描得到。逐步:

1. 在生成图 $\hat{I}$ 上运行**开放词表目标检测器**(OWL-ViT,[Minderer et al., ECCV 2022](https://arxiv.org/abs/2205.06230)),得到 $M$ 个检测 $D=\{(d_j,l_j,c_j)\}_{j=1}^M$ —— 框、标签、置信度。
2. 对每个**目标物体** $o_i$,只保留标签匹配的检测:$\hat{D}_i=\{(d_j,l_j,c_j)\in D \mid l_j=o_i\}$。
3. 在其中取**置信度最高**的检测:$\hat{d}_i=d_{j^*}$,$j^*=\arg\max_{j} c_j$。
4. 与**目标框** $t_i$ 计算 IoU:$\text{IoU}_i=\text{IoU}(t_i,\hat{d}_i)$。
5. 在 $k\in\{0.1,0.2,\dots,0.9\}$ 上对 IoU 设阈值,对每个 $k$ 算出有多少物体过线:

$$accuracy@k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{IoU}_i\ge k]$$

通俗解释:$accuracy@k$ 就是「被要求的物体里,有多大比例至少被摆得有 $k$ 那么准」。宽松阈值($k=0.1$)容忍粗糙摆放;严格阈值($k=0.9$)要求近乎完美的重叠。

6. $s_{layout}$ = 由此得到的 $accuracy@k$ 曲线**下面积**(所有 $k$ 上)。单个 AUC 标量一次性概括了所有严苛程度下的空间精度,避免了只挑一个 IoU 阈值的随意性。

**使用的外部模型。** 布局用 OWL-ViT(开放词表检测器,公开 HF 权重);文本用 TIFA 栈(LLM + VQA)。

**设计选择与为何用 AUC。** 在一*段* IoU 阈值范围上取 AUC(而非只用 accuracy@0.5),正是布局分数稳健的原因:它不偏袒某一种「足够接近」的定义,并天然惩罚那种在宽松阈值下勉强过关、却在严苛阈值下崩盘的模型。

### 3.3 直觉解释

把 7Bench 想象成一张**双科目成绩单 + 一场固定公平的考试**。考题(模块 A)由一份严格的考纲拟定 —— 七个主题,每个主题题量相同,没有泄漏的真题(不用 COCO)。每张生成图随后由两位独立的考官评分(模块 B):一位**阅读考官**(TIFA:"句子要你画的东西你画了吗?颜色对吗?")和一位**地理考官**(OWL-ViT 布局 AUC:"你把每个东西放到我给你的地图上那个精确位置了吗?")。关键在于两个分数**分开报** —— 一个把红色汽车画得完美却摆错角落的模型,阅读拿 A、地理拿 F,而 7Bench 拒绝把它平均成一个误导性的 B。

---

## 4. 数据构建

### 4.1 数据来源

7Bench 的 **prompt 完全合成、布局全人工标注** —— *没有*任何爬取的图像语料。因此来源是 (i) 受控词表 与 (ii) 人工绘框:

| 词表 | 符号 | 规模 | 来源 / 依据 |
|---|---|---|---|
| 物体 | $\mathcal{O}$ | (开放) | 作者的物体表;分布见图 2 |
| 颜色 | $\mathcal{C}$ | 11 | Berlin–Kay 普适基本色(1991) |
| 属性 | $\mathcal{A}$ | 30 | 作者自定(颜色/形状/材质/外观/尺寸) |
| 关系 | $\mathcal{R}$ | 11 | Landau & Jackendoff(1993)空间认知 |

![物体分布](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig2_object_distribution.png)

*图 2 —— 全部 224 条 prompt 中各物体的出现次数。物体集多样(没有单一物体主导),支撑「广覆盖」的说法。*

### 4.2 流程逐步拆解

1. **选场景**(7 选 1)→ 固定模板与任何布局约束。
2. **采样物体** $\mathcal{o}_i$,从 $\mathcal{O}$ 中取,$N\in\{1,2,3,4\}$。
3. **挂修饰词**(按场景):颜色绑定取 $\mathcal{C}$、属性绑定取 $\mathcal{A}$、物体关系取 $\mathcal{R}$,或不挂(物体/小框/重叠框)。
4. **渲染 prompt**:用场景模板,配冠词选择 $det(\cdot)$。
5. **绘制布局:** 默认手工画真实框;小框场景采样面积落在 [3%,10%];重叠框场景强制至少一对重叠。
6. **平衡:** 重复直到每场景恰好 **32 个样本** → 总 **224 个**。

没有报告任何基于模型的过滤或产出损耗步骤 —— 集合是手工整理到目标数量的,所以从构造上看「产出率」是 100%。

### 4.3 标注方法

框由作者**手工排布**,力求真实、比例协调。论文**未**报告标注者人数、资质、报酬、标注者间一致性(IAA)或正式 QA 流程 —— *论文未指明。* 考虑到规模不大(224 个布局)且作者即标注者,这是单一团队的整理而非众包工作。

### 4.4 合成 / 模型生成数据

Prompt 是**模板生成,而非 LLM 生成** —— 没有可复现的生成器 prompt。下游*图像*由 5 个被测模型(§5.1)产出,而非基准本身。去污染是结构性达成的:因为 prompt/框都是合成且手画的,它们独立于任何模型的训练分布(这正是对 COCO 泄漏的明确修复)。

### 4.5 最终统计

| 属性 | 数值 |
|---|---|
| 样本总数(prompt + 框集合 对) | **224** |
| 场景数 | **7** |
| 每场景样本数 | **32**(均衡) |
| 每 prompt 物体数 $N$ | $\{1,2,3,4\}$ |
| 小框面积约束 | 图像面积的 3%–10% |
| 颜色词表 | 11 |
| 属性词表 | 30 |
| 关系词表 | 11 |
| 用于评测的生成图 | **17,920**(= 224 × 16 种子 × 5 模型) |
| 图像分辨率 | 512 × 512 |

![边界框面积分布](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig3_bbox_area_distribution.png)

*图 3 —— 各场景的框面积分布(占图像面积 %)。"小框"如设计般紧贴 0–5%;"重叠框"与"复杂组合"分布最宽(框可达 ~80%);"物体关系"偏小到中。这证实每个场景确实在锻炼其意图中的空间区间。*

### 4.6 基准协议

评测协议*即*基准的协议 —— 在 §3.2 中已完整说明:文本用 TIFA,布局用基于 OWL-ViT 的 accuracy@k AUC,二者均在 $[0,1]$,均为现成预训练,IoU 阈值 $\{0.1,\dots,0.9\}$,每样本 16 个种子(1–16)。评判模型都是公开 checkpoint,使协议确定且可复现。

### 4.7 已知偏差 / 局限

- **仅英文**,单语言 prompt。
- **绝对规模小**(224 条 prompt)—— 作为固定评测集尚可,但单场景统计功效仅 32 个样本。
- **模板化 prompt** —— 语法高度统一;不是自然的自由口语 caption,因此真实世界的 prompt 多样性被低估。
- **受检测器限制的布局指标** —— $s_{layout}$ 继承 OWL-ViT 的检测误差(一个摆放正确但检测器漏检的物体得 0 分)。

---

## 5. 实验与评测

### 5.1 实验设置

**被测模型(5 个):**

| 标记 | 模型 | 类型 | 基座 |
|---|---|---|---|
| **G** | GLIGEN([Li et al., CVPR 2023](https://arxiv.org/abs/2301.07093)) | 带 grounding 训练 | 自有 |
| **G_AR** | Attention Refocusing([Phung et al., CVPR 2024](https://arxiv.org/abs/2306.05427)) | 免训练 | 基于 GLIGEN |
| **G_BD** | BoxDiff([Xie et al., ICCV 2023](https://arxiv.org/abs/2307.10816)) | 免训练 | 基于 GLIGEN |
| **SD_CAG** | Cross-Attention Guidance([Chen et al., WACV 2024](https://arxiv.org/abs/2304.03373)) | 免训练 | 基于 Stable Diffusion |
| **SD** | Stable Diffusion v1.4([Rombach et al., CVPR 2022](https://arxiv.org/abs/2112.10752)) | 纯文本基线 | — |

SD 没有布局输入,所以**只出现在文本对齐**图中(无 $s_{layout}$)。

**生成预算。** 每个样本 16 张图(种子 1→16)→ 共 **17,920** 张,分辨率 **512×512**。**指标:** 预训练 TIFA + OWL-ViT。

### 5.2 主要结果

论文将结果以**柱状图(按场景)**呈现,而非数值表。下面我把柱高转录出来(约 ±0.01)。**每场景最优加粗。**

**各场景文本对齐分 $s_{text}$(图 4):**

| 场景 | SD | SD_CAG | G | **G_BD** | G_AR |
|---|---|---|---|---|---|
| 物体绑定 | 0.71 | 0.81 | 0.80 | **0.92** | 0.83 |
| 小框 | 0.69 | 0.76 | 0.71 | **0.77** | 0.76 |
| 重叠框 | 0.69 | 0.78 | 0.75 | **0.87** | 0.79 |
| 颜色绑定 | 0.56 | 0.71 | 0.54 | **0.69** | 0.59 |
| 属性绑定 | 0.76 | 0.83 | 0.82 | **0.87** | 0.84 |
| 物体关系 | 0.67 | 0.76 | 0.79 | **0.83** | 0.82 |
| 复杂组合 | 0.75 | 0.79 | 0.79 | **0.81** | 0.80 |

![各场景文本对齐](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig4_results_text.png)

*图 4 —— 文本对齐。范围 ≈ 0.55–0.92。**BoxDiff(G_BD,红色)在每个场景都最优。** 颜色绑定的文本最难(SD 0.56,G 0.54);属性绑定与物体绑定最易。注意 SD_CAG 在颜色绑定上击败了原始 G(0.71 vs 0.54),尽管它基座更弱 —— 一个免训练方法挽救了概念保真度。*

**各场景布局对齐分 $s_{layout}$(图 5):**(SD 省略 —— 无布局输入)

| 场景 | SD_CAG | **G** | G_BD | G_AR |
|---|---|---|---|---|
| 物体绑定 | 0.23 | **0.44** | 0.44 | 0.39 |
| 小框 | 0.05 | **0.27** | 0.22 | 0.21 |
| 重叠框 | 0.24 | **0.49** | 0.48 | 0.45 |
| 颜色绑定 | 0.23 | **0.40** | 0.40 | 0.40 |
| 属性绑定 | 0.22 | **0.49** | 0.47 | 0.46 |
| 物体关系 | 0.15 | **0.42** | 0.40 | 0.39 |
| 复杂组合 | 0.22 | **0.46** | 0.44 | 0.41 |

![各场景布局对齐](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig5_results_layout.png)

*图 5 —— 布局对齐。**没有模型超过 ~0.5** —— 空间保真度是该领域的薄弱维度。**GLIGEN(G,绿色)几乎处处领先。** 小框场景是灾难性的(G 0.27,SD_CAG 0.05)。SD_CAG(基于 SD 的免训练)远低于 GLIGEN 家族,说明训练型 grounding 模型在空间上从根本上更强。*

### 5.3 消融研究 —— 性能 vs. 物体数

论文的消融变化 $N$(每 prompt 物体数)。数值从图 6–7 转录。

**文本对齐 vs. 物体数(图 6):**

| 物体数 | SD | SD_CAG | G | G_BD | G_AR |
|---|---|---|---|---|---|
| 1 | 0.92 | 0.91 | 0.85 | 0.91 | 0.88 |
| 2 | 0.72 | 0.81 | 0.76 | 0.81 | 0.79 |
| 3 | 0.60 | 0.75 | 0.69 | 0.80 | 0.74 |
| 4 | 0.56 | 0.67 | 0.69 | **0.79** | 0.71 |

![消融:文本 vs 物体数](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig6_ablation_text.png)

*图 6 —— 文本对齐随物体数增加单调下降,符合预期。**纯 SD 崩得最狠**(0.92→0.56)。**BoxDiff 最稳健**(0.91→0.79),且在 $N=4$ 时领先所有模型。布局引导方法比纯文本基线衰减得更平缓。*

**布局对齐 vs. 物体数(图 7):**(SD 省略)

| 物体数 | SD_CAG | G | G_BD | G_AR |
|---|---|---|---|---|
| 1 | 0.26 | 0.45 | 0.45 | 0.42 |
| 2 | 0.24 | **0.48** | 0.46 | 0.44 |
| 3 | 0.16 | 0.42 | 0.40 | 0.37 |
| 4 | 0.11 | 0.36 | 0.37 | 0.32 |

![消融:布局 vs 物体数](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig7_ablation_layout.png)

*图 7 —— 布局对齐也随物体数下降,但 GLIGEN 家族在 $N=2$ 时见顶(G 0.48)后才下滑,而 SD_CAG 从本就很低的 0.26 一路衰减到 0.11。训练型模型(G)的衰减斜率明显小于基于 SD 的免训练方法。*

### 5.4 扩展 / 容量研究

没有模型规模或训练 token 的扩展研究(用的是现成模型)。物体数消融(§5.3)是唯一的「难度扩展」轴,显示两个指标都呈清晰、一致的退化。

### 5.5 定性结果

![定性结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig8_qualitative.png)

*图 8 —— 列:Cross-Attention Guidance、GLIGEN、Attention Refocusing、BoxDiff;行是各示例 prompt(如 "A chair and a fox"、"An elephant on a skyscraper"、"A boy and a mouse"、"A rabbit and a tree"、"A purple airplane and a white car"、"A round bowl and a soft chair"、"A butterfly landed gently on the rose in the garden")。虚线框 = 真值目标位置;实线框 = OWL-ViT 检测(每物体配色)。视觉结论:Cross-Attention Guidance(最左列)经常漏掉或摆错物体;GLIGEN 家族各列把物体放得离虚线目标近得多,与定量差距一致。*

### 5.6 失败案例

- **小边界框**是主导失败模式 —— 每个模型的布局分都在这里暴跌(最优 G = 0.27,SD_CAG = 0.05),因为太小的框给去噪过程提供的定位信号太少。
- **颜色绑定**是最难的*语义*场景(原始 GLIGEN 文本 0.54),反映持续存在���属性绑定/泄漏失败。
- **重叠框反而不是失败模式** —— 反直觉地,最优布局分恰恰出现在这里(~0.45–0.49);模型「想出有创意的办法」来满足重叠要求。

### 5.7 成本与效率

除了致谢 CINECA/ISCRA HPC 资源外,未报告延迟、GPU 时数或显存数字。*论文未指明。*

### 5.8 人工评测

无 —— 全部评测自动化(TIFA + OWL-ViT)。无人工胜率或 IAA。

### 5.9 逐基准点评

- **物体绑定:** 文本容易(G_BD 0.92),布局中等(~0.44)。适合作热身场景。
- **小框:** 区分度最强的场景 —— 把空间能力拉开得很明显(G 0.27 vs SD_CAG 0.05)。是整个基准中信息量最大的场景。
- **重叠框:** 两个维度都高 —— 并非作者预想的难点。
- **颜色 / 属性绑定:** 颜色在语义上难;属性两个维度都相对容易(可能因为属性集含很多容易的视觉线索)。
- **物体关系:** SD_CAG 布局弱(0.15)—— 关系是空间性的,基于 SD 的方法处理得差。
- **复杂组合:** 所有布局引导模型**收敛**到 ~0.79–0.81 文本与 ~0.41–0.46 布局 —— 在最大难度下它们的差异被抹平,暗示存在一个共同天花板。

---

## 6. 优点

1. **首个在单一固定协议下衡量两个维度。** 文本+布局的联合评测配一套可复现的固定配方(TIFA + OWL-ViT AUC,固定种子),直接弥合了论文诊断的异质性空白;图 4–5 的逐场景拆分让两个维度清晰可辨地分开。
2. **无污染、均衡的设计。** 合成模板 prompt + 手画框规避了 COCO 泄漏(BoxDiff 的批评),而精确的每场景 32 个的平衡(图 3 证实每场景都打到其空间区间)防止任何单一技能主导总分。
3. **有原则的布局指标。** 在 IoU 阈值上对 accuracy@k 取 **AUC** 比单阈值 IoU 或 FID/YOLO 更稳健:它奖励跨严苛程度的一致性,且与文本分*并列*而非*混合*报告(图 5)。
4. **可操作的结论。** 基准立刻浮现出一张具体、有排序的弱点地图:布局处处 ≤ 0.5、小框灾难性、SD 上的免训练 < 训练型 grounding、物体数增多时平缓 vs 陡峭的退化(图 6–7)。

## 7. 缺点与局限

1. **规模小且仅英文。** 224 条 prompt / 每场景 32 个,统计功效有限,也无语言/领域多样性;关于「state-of-the-art 模型」的宽泛结论建立在一个小而风格单一的集合上。*改进:* 扩到数千条 prompt,并加入多语言/自然 caption 变体。
2. **未报告统计可靠性。** 尽管每样本 16 个种子,论文只报告柱高单点 —— **没有误差棒、标准差或显著性检验**。若干场景差距(如 G vs G_BD 布局 ~0.44 vs 0.44)很可能落在种子噪声内。*改进:* 报告种子上的均值±标准差并做显著性检验。
3. **指标继承评判模型盲点。** $s_{layout}$ 受 OWL-ViT 召回率限制(摆放正确但漏检的物体得 0),$s_{text}$ 受 TIFA VQA/LLM 栈限制;两个评判都未在 *7Bench 上*对人工的空间/语义判断做校验。*改进:* 一个小型人工一致性研究来校准两个分数。
4. **无数值表 / 总分。** 全部结果都在柱状图里,逼读者目测数值;没有给出单一的榜单头条数字。*改进:* 发布带精确数值的 CSV/榜单。
5. **模型阵容偏旧。** GLIGEN/BoxDiff/CAG/AR(2023–2024)早于更新的 grounding 方法和基于 DiT 的生成器;基准的结论未必能迁移到当前 SOTA。*改进:* 加入现代 grounded 生成器。

## 8. 与同期 / 相关工作的对比

| 基准 | 文本对齐? | 布局输入? | 布局指标? | 样本数 | 均衡场景 | 抗污染 |
|---|---|---|---|---|---|---|
| **7Bench(本文)** | ✅ TIFA | ✅ 框 | ✅ accuracy@k AUC | 224(7×32) | ✅ | ✅(合成) |
| TIFA([Hu 2023](https://arxiv.org/abs/2303.11897)) | ✅ | ❌ | ❌ | ~4k 题 | 部分 | ❌ |
| T2I-CompBench([Huang 2023](https://arxiv.org/abs/2307.06350)) | ✅ | ❌ | ❌ | 6k prompt | ✅ | ❌ |
| HRS-Bench([Bakr 2023](https://arxiv.org/abs/2304.05390)) | ✅ | ❌ | ❌(空间靠文字) | 45k prompt | ✅ | ❌ |
| Cho 等诊断基准([2024](https://arxiv.org/abs/2304.06671)) | ❌ | ✅ | ✅ 仅空间 | — | 仅空间 | 部分 |
| COCO/Flickr 子集评测(GLIGEN/BoxDiff) | ❌ | ✅ | FID / YOLO | 5k–30k(不一) | ❌ | ❌(COCO 泄漏) |

7Bench 是唯一在**文本对齐、布局输入、布局指标、均衡、抗污染上同时 ✅** 的一行 —— 代价是**原始样本数最小**。

## 9. 可复现性审计

| 项 | 是否公开? | 备注 |
|---|---|---|
| 代码 | ✅ | [github.com/Elizzo/7Bench](https://github.com/Elizzo/7Bench) |
| 基准数据(prompt + 框) | ✅ | 公开 224 对 |
| 模型权重 | ➖ 不适用 | 使用公开现成模型(GLIGEN、BoxDiff、CAG、AR、SD v1.4) |
| 评判模型 | ✅ | 公开 TIFA + OWL-ViT checkpoint |
| 超参(评测) | ✅ | IoU 阈值、16 个种子(1–16)、512×512 完整说明 |
| 生成超参(CFG、步数) | ❌ | 各模型生成设置未详述 |
| 评判 prompt(TIFA LLM/VQA) | ➖ | 沿用 TIFA 仓库,未重新印出 |
| 数值结果表 | ❌ | 结果仅为柱状图 |
| 硬件规格 | ❌ | 仅致谢 "CINECA/ISCRA HPC" |
| 统计方差 | ❌ | 尽管有 16 个种子但无误差棒 |

**结论:** **协议/数据层面可复现,结果层面薄弱。** 任何人都能重跑 7Bench:224 样本集、评判 checkpoint、IoU 阈值、16 种子方案都公开且锁定,且不涉及私有权重。短板在*报告*:结果仅柱状图(无精确数字,有 16 个种子却无误差棒),各模型生成超参(CFG scale、去噪步数)缺失,且评判模型从未在本集上对人工一致性做校验。实践者能忠实复现*实验*;复现*确切报告的柱高*则需目测。总体是一项扎实、诚恳的基准贡献,主要风险是过度解读那些小而无方差的逐场景差距。

---

## 总结

7Bench 填补了一个真实且论证充分的空白:它是第一个在单一固定、无污染的协议下,对布局引导文生图模型「画了什么」**和**「画在哪里」**双重**打分的基准。它的核心实证信息直白且有用 —— **当今模型在语义上尚可(0.55–0.9),但空间控制很差(≤0.5),小边界框是主导失败模式,且训练型 grounding(GLIGEN)在空间保真度上明显胜过基于 SD 的免训练方法。** 其贡献是方法论层面的而非新模型;主要弱点是规模小、不报方差、结果仅有图表。
