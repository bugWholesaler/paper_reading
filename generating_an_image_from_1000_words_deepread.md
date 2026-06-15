# Generating an Image From 1,000 Words: Enhancing Text-to-Image With Structured Captions

> **作者:** Eyal Gutflaish, Eliran Kachlon, Hezi Zisman, Tal Hacham, Nimrod Sarid, Alexander Visheratin, Saar Huberman, Gal Davidi, Guy Bukchin, Kfir Goldberg, Ron Mokady（BRIA AI）
> **会议/年份:** arXiv 预印本(CVPR 模板), 2025 年 11 月
> **链接:** [arXiv:2511.06876](https://arxiv.org/abs/2511.06876)
> **Code / Weights / Data:** Weights ✅ ([HuggingFace: briaai/FIBO](https://huggingface.co/briaai/FIBO))；Code ✅(论文声明 "release model weights, code, and full implementation details")；训练数据 ❌(120M 授权图文对，未公开)

---

## TL;DR

FIBO 是**第一个完全用「长结构化 caption」(JSON 格式、平均 1160 词)训练的开源文生图模型**(8B 参数)。它配套提出三件东西:(1) **DimFusion** —— 在**嵌入维度**而非序列维度上融合 LLM 中间层表示的高效融合架构,把长 caption 的注意力开销压下来(每步从 0.8s 降到 0.5s,约 1.6× 加速);(2) **TaBR** —— 用「真实图 → caption → 重建图 → 人评相似度」的循环一致性协议来度量长 caption 的可控性/表达力;(3) 在 PRISM-Bench-licensed 上取得**开源模型中最高的对齐分(Overall 84.9)**,在 GenEval 上以 8B 参数达到 0.85(Position 0.80 为全场最高)。核心论点:**扩展「语言」本身(caption 的长度与结构),而不是只扩展模型/数据,是通往可控生成的杠杆。**

---

## 1. 背景与动机

### 1.1 问题定义

文生图模型(text-to-image, T2I)绝大多数是把**短 prompt** 映射到**信息极其丰富的图像**。这造成一个根本的**信息不对称**:输入文本是稀疏的,输出图像是稠密的。模型必须自己「脑补」prompt 没说的所有细节(光照、景深、构图、人物表情、相机角度……)。

### 1.2 为什么重要

为了让脑补的结果讨喜,主流模型(借助 user study + 偏好优化,如 [DPO-Diffusion, Wallace et al., 2024](https://arxiv.org/abs/2311.12908)、[Pick-a-Pic, Kirstain et al., 2023](https://arxiv.org/abs/2305.01569))会**偏向「平均人类偏好」**。结果是:
- 对休闲用户友好(出图好看);
- 但对**专业用户**(需要精确控制构图、光照、景深的设计师/广告从业者)**牺牲了精度与创造力**。

论文用谚语"an image is worth a thousand words"点题:**短 prompt 在物理上无法完整指定一张图**。

### 1.3 既有工作的具名局限

| 方向 | 代表工作 | 论文指出的局限 |
|---|---|---|
| 合成 caption | [DALL·E 3, Betker et al., 2023](https://cdn.openai.com/papers/dall-e-3.pdf)、[SD3, Esser et al., 2024](https://arxiv.org/abs/2403.03206)、[Playground v3, Liu et al., 2024](https://arxiv.org/abs/2409.10695) | 合成 caption **不穷尽**:captioning 模型会**按自己判断省略**部分细节,导致训练时关键信息缺失,生成器学会「忽略 prompt」 |
| LLM 融合(序列拼接) | [HiDream-I1, Cai et al., 2025](https://arxiv.org/abs/2505.22705) | 双流 decoder-only LLM,但每个 attention block 同时吃**最后层 + 中间层** token → **序列长度翻倍** → 对 1000+ token 的长 caption 计算成本高 |
| LLM 融合(cross-attn) | [Playground v3](https://arxiv.org/abs/2409.10695)、[Tang et al., 2025](https://arxiv.org/abs/2503.20314) | 用 cross-attention 融合多层,**但缺乏双向 text-image 混合(joint attention)**,而双向混合已被证明能提升 prompt adherence([SD3](https://arxiv.org/abs/2403.03206)) |
| 评测 | user-preference("你更喜欢哪张?") | 长 caption(>1000 词)**人类无法解析**;偏好评测奖励所谓 "AI 美学"(过饱和、主体居中),即 [FLUX](https://github.com/black-forest-labs/flux) 论文称的 *bakeyness* |

### 1.4 本文填补的空白

训练「永远完整」的长结构化 caption + 高效融合(DimFusion) + 图像锚定的评测(TaBR),从而把优化目标从"偏好流行度"转向"表达力与可控性"。

---

## 2. 相关工作

### 2.1 文生图模型
扩散模型主线:GLIDE/Imagen/DALL·E 2 确立强语言编码器条件 → [Latent Diffusion (LDM), Rombach et al., 2022](https://arxiv.org/abs/2112.10752) 让大规模训练可行 → [SDXL, Podell et al., 2023](https://arxiv.org/abs/2307.01952) 扩展 UNet → 新一波转向 **Transformer 骨干 + flow-matching**:[SD3](https://arxiv.org/abs/2403.03206)(DiT 风格、双向 text-image 混合)、[FLUX](https://github.com/black-forest-labs/flux)、[HiDream-I1](https://arxiv.org/abs/2505.22705)、[Qwen-Image](https://arxiv.org/abs/2508.02324)。

### 2.2 合成 caption
从 [LAION](https://arxiv.org/abs/2210.08402)(HTML alt-text,噪声大)→ [DALL·E 3](https://cdn.openai.com/papers/dall-e-3.pdf) 证明合成描述性 caption 提升评测 → 混合人写/合成 caption。本文进一步用现代 VLM 生成**结构化 JSON caption**,显式描述对象属性、空间关系、构图、摄影风格,且**训练与推理用同一套 schema**。

### 2.3 LLM 融合
早期用 CLIP/T5 作文本编码器 → 近期用 LLM(更丰富语义、组合推理)。因 LLM 不同中间层编码互补语言信息,Playground v3 / Tang et al. 用 cross-attention 融合多层;HiDream 用双流但成本高。**DimFusion 的定位**:既用中间层+最后层、又保持双向混合、还不增加 token 长度。

### 2.4 循环一致性评测
循环一致性(image→text→image 应能近似重建原图,源自 [CycleGAN, Zhu et al., 2017](https://arxiv.org/abs/1703.10593))。近期用作 RL 奖励([Meng et al., 2025](https://arxiv.org/abs/2502.xxxxx)、[Bahng et al., 2025](https://arxiv.org/abs/2506.02095))或自动对齐指标。Meng et al. 做了基于 caption 重建的自动评测,**但没有对原图做像素/感知级直接比较**。TaBR 的区别:**显式把重建图与原图比较**,且用人评。

---

## 3. 核心方法

论文方法分四节展开:§3.1 长结构化 caption(动机+workflow);§3.2 DimFusion 架构;§3.3 长 caption 评测(TaBR);§3.4 大规模训练(FIBO)。下面逐节用「Method Deep-Dive Checklist」深挖。

![FIBO teaser:从长结构化 caption 生成图像](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_teaser.png)

### 3.1 用长结构化 caption 训练

**目的与定位。** 这是整篇论文的「数据哲学」。观察:即便 SOTA 模型也搞不定复杂场景 —— 反常识组合(熊倒立、女人用飞镖写字)、多个个体各自不同表情、微妙的光照/相机参数。论文归因于**训练数据的根本缺陷**:大多数 caption 缺少细粒度视觉因子描述。即使合成 caption 也非穷尽,captioning 模型会按自己判断省略 → 训练时关键信息时有时无 → 生成器习惯了「不完整条件」→ 学会忽略 prompt。

**解法。** 训练**长结构化 caption**,保证颜色、光照、构图等关键方面**在每个样本里都出现**,从而消除歧义、建立更强的 text-image 对齐,迫使生成器**接地(ground)到 caption** 而非依赖先验。

**关键经验发现 1:更快收敛 + 更好质量。** 用同一批图、1B 小模型、低分辨率、训练 100K 步,对比长结构化 caption vs 短 caption:

| Caption 类型 | FID ↓ (COCO-2014 val, 30K 图, 100K 步) |
|---|---|
| **长结构化 caption** | **19.01** |
| 短 caption | 34.04 |

长 caption 的 FID 几乎是短 caption 的一半,且定性上更连贯、细节更丰富(下图)。

![长结构化 caption vs 短 caption:定性对比 + FID](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_caption_length.png)

**关键经验发现 2:解耦(disentanglement)自发涌现。** 这是论文最惊艳的发现。在标准 T2I 模型里,改 prompt 一个细节常常牵动一堆无关因子。而 FIBO **只改结构化 caption 里的一个属性 + 固定随机种子**,通常**只有对应的视觉因子改变**,其余保持不变 —— 无需任何额外编辑技术([Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)、[Null-text Inversion](https://arxiv.org/abs/2211.09794))。

> 关键:**这不是图像编辑**(模型不吃输入图),模型**训练时也从未见过成对图像**。解耦纯粹从「持续详细的条件」中无监督地涌现。

![解耦:修改单一属性(black mug→white、apple→orange、knight 去头盔、换季节等)只改对应因子](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_disentanglement.png)

**结构化 caption 的 schema(JSON 字段)。** 每条 caption 是一个长 JSON,把图像分解为可解释字段。从最显著细节(对象、背景、文字)开始,对象带丰富属性(大小、位置、形状、颜色、纹理、朝向、与其他对象关系),人物额外带姿态、表情、衣着、动作、性别、肤色;然后补充常被遗漏的信息(光照、景深、构图、相机参数)。完整字段:

| 字段 | 类型 | 内容 |
|---|---|---|
| `short_description` | String | 图像内容简短摘要(≤200 词) |
| `objects` | Array | ≤5 个显著对象;每个含 description/location/relative_size/shape_and_color/texture/appearance_details/relationship/orientation;若是人额外含 pose/expression/clothing/action/gender/skin_tone_and_texture;若是物体簇含 number_of_objects |
| `background_setting` | String | 环境/背景 |
| `lighting` | Object | conditions(日光/影棚…)、direction(主光方向)、shadows |
| `aesthetics` | Object | composition(三分法/对称…)、color_scheme、mood_atmosphere |
| `photographic_characteristics` | Object(可选) | depth_of_field、focus、camera_angle、lens_focal_length |
| `style_medium` | String | 媒介(照片/油画/3D 渲染…) |
| `text_render` | Array | ≤5 个文字渲染;每个含 text/location/size/color/font/appearance_details |
| `context` | String | 图像类型与用途(如时尚大片/产品图) |
| `artistic_style` | String | 高层艺术风格(电影感/极简…,≤3 词) |

**caption 用什么生成。** 用 SOTA VLM **[Gemini 2.5, Comanici et al., 2025](https://arxiv.org/abs/2507.06261)**,配专用 system prompt + Gemini 的 structured outputs(强制符合 JSON schema,保证完整性/一致性)。论文发现:**VLM 擅长客观描述,但主观判断不可靠(倾向于谄媚地夸)**,因此补充两个专门的美学打分器([Pick-a-Pic / PickScore](https://arxiv.org/abs/2305.01569)、[LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/))来量化每张图的美学水平并写进 caption。

**caption 长度统计(Table 1):**

| | 均值 | 中位 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|---|
| **#Tokens** | 1160.3 | 1176 | 316.4 | 192 | 1800 |

**用户接口:专用 VLM 的三种操作。** 用户不可能手写 1000 词严格 schema,所以集成一个 VLM 支持三种操作:
- **(1) Generate**:把短 prompt 扩展成结构化 prompt;
- **(2) Refine**:给定编辑指令修改已有结构化 caption(如「去掉头盔,马改成白色」);
- **(3) Inspire**:从输入图提取结构化 caption 并可选编辑,图作为创意引导。

![Workflow:Caption→(VLM)JSON→FIBO 出图;Refiner(VLM)按指令改 JSON→再出图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_workflow.png)

VLM 引入额外算力,但带来世界知识、推理能力、多语言支持,且能与解耦能力结合做迭代精修。

**开源 VLM 训练细节(可复现关键)。** Gemini 2.5 有效但贵。为社区提供高效替代,论文微调开源 **[Qwen-3 VL 4B](https://github.com/QwenLM/Qwen3-VL)**:
- **训练数据**:合成生成的短 prompt + 编辑指令,用与图像模型**同一套结构化 schema** 以保证对齐;
- **算力**:8× H100,共 **2B tokens**;
- **鲁棒性技巧**:把「图像条件任务」和「纯文本任务」**解耦**,各自用不同种子重复训练,最后用 **[model merging, Yang et al., 2024](https://arxiv.org/abs/2408.07666)** 合并权重。

**直觉解释。** 想象教一个学徒画画:如果你每次只给他一句话("画只猫"),其余全靠他猜,他会慢慢养成"反正你说不清,我按我喜欢的来"的习惯 —— prompt 形同虚设。而如果你每次都给他一份**标准化的完整工单**(猫的姿态、毛色、光从哪来、景深多浅、构图怎么摆),他被迫一项项照做,自然学会"工单里改哪项就改画里哪项",这就是解耦自发涌现的原因。

---

### 3.2 DimFusion 架构

**目的与定位。** 解决 §3.1 带来的新难题:**长 caption(1000+ token)的高效融合**。要充分利用结构化 caption,模型必须处理长文本 token 序列并与图像 token 融合。理想做法是:用 LLM 中间层(语义/组合理解强)+ 双向混合 + 最后层与中间层结合。但**朴素地沿序列维度拼接中间层会增加 token 长度**,attention 计算暴涨 —— 尤其在预训练早期低分辨率阶段,图像 token 远少于文本 token,attention 成本随 caption 长度而涨。

**核心机制(数学层面)。** 设 $d_{\text{model}}$ 为嵌入维度(token 特征向量长度,即形状 $(B, L, d_{\text{model}})$ 的最后一维),$D$ 为 transformer 的隐藏维度。DimFusion 的做法:

1. **初始文本编码**:把 LLM **最后两层**沿**嵌入维度**拼接,投影到 $D/2$。
   - (为何取两层:脚注解释 SmolLM 隐藏维度相对 LLaMA/T5-XXL 较小,故取最后两层拼接以补足表达力。)
2. **每个 block 内**:把**对应 LLM 层**的 hidden states 投影到 $D/2$,沿嵌入维度与当前文本编码拼接 → 恢复到维度 $D$。
3. **与噪声 latent 一起**:在 Dual-stream 和 Single-stream block 里做**双向混合(joint attention)**。
4. **block 处理后**:丢弃附加的 $D/2$ 分量,文本编码回到 $D/2$。

> 关键洞察:**因为每个 token 始终保留一半($D/2$)是「自己的」**,DimFusion 天然支持 [SD3](https://arxiv.org/abs/2403.03206) 式的「先 dual-stream 后 single-stream」结构。token 数始终不变 → 长 caption 下计算成本大幅下降。

![DimFusion 架构:每层把文本编码(绿)与对应 LLM hidden states 沿嵌入维度拼接,与噪声 latent(蓝)双向混合,block 后丢弃附加分量](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_dimfusion_arch.png)

**消融实验(Figure / Table)。** 1B transformer + SmolLM3-3B 作文本编码器,150K 步,128× H200,同 batch size。三种配置:
- **Baseline (T5)**:用 T5-XXL 最后层,无 LLM 融合;
- **TokenFusion**(仿 HiDream):沿**序列维度**拼接 LLM hidden states(文本 token 翻倍);
- **DimFusion**(本文):沿嵌入维度拼接。

| 架构 | FID ↓ (30K COCO-2014 val) | 每步平均耗时 (秒) ↓ |
|---|---|---|
| T5 | 36.46 | 1.28 |
| TokenFusion | 15.90 | 0.8 |
| **DimFusion** | **15.58** | **0.5** |

结论:两种 LLM 融合都**碾压 T5 baseline**(FID 36→16,证实 LLM 表示的价值);DimFusion 的 FID **略优于** TokenFusion,同时把每步耗时从 0.8s 降到 0.5s(**约 1.6× 加速**)。训练 loss 曲线显示 DimFusion 与 TokenFusion 几乎收敛到同一值,但算力低得多。

![DimFusion vs TokenFusion vs T5:FID 表 + 训练 loss 曲线 + 与 TokenFusion 的 loss 差](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_dimfusion_ablation.png)

**直觉解释。** TokenFusion 像是把 LLM 的笔记**逐页插进**你的笔记本里 —— 笔记本越来越厚(序列变长),翻阅(attention)越来越慢。DimFusion 则是把 LLM 的笔记**写在每页的下半页空白处**(嵌入维度),页数不变,看完这页就擦掉下半页换下一层的笔记 —— 既吸收了 LLM 各层信息,又不让本子变厚。

---

### 3.3 评测长 caption:TaBR

**目的与定位。** 解决「长 caption 没法评测」的问题。传统 user-preference("你更喜欢哪张?")对短 prompt 有效,但对 1000+ 词的长 caption 失效(人读不完),且偏好评测奖励 "AI 美学"(过饱和、主体居中,即 *bakeyness*),激励模型优化短 prompt 与平均美学,而非表达力与可控性。

**核心思想。** 利用一个**不对称性**:人**读不懂长文本,却能可靠比较复杂图像**。于是 TaBR 把**图像本身**作为评测锚点。

**协议(逐步)。**
```
1. 取一张真实图像 I(不在训练集中)
2. 用 captioner 给 I 生成 caption C
3. 把 C 作为生成器的唯一输入 → 生成重建图 I'
4. 生成两个候选重建(来自不同模型)
5. 问标注者:"哪张图更像上面这张原图?"
```
因为有原图作为参照(信息远比短文本丰富),评判不会塌缩成个人口味,而是**直接度量模型的表达力**。

![TaBR 定性例:Original / FLUX / Qwen-Image / FIBO 重建对比;FIBO 最忠实保留姿态、相机角度、场景布局](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_tabr_qual.png)

**直觉解释。** 与其让裁判读一份 1000 字的"嫌疑人画像描述"再判断哪张拼图更准(读不完、易主观),不如直接把**嫌疑人本人照片**摆出来,让裁判比对哪张拼图更像本人 —— 这就是"用图像当锚点"的妙处。

---

### 3.4 大规模训练:FIBO

**模型规格(Table 2)。** 8B 参数,采用 DimFusion 架构,融合 LLM 为 **[SmolLM3-3B](https://huggingface.co/blog/smollm3)**;在 latent 空间运行,VAE 用 **[Wan 2.2 VAE](https://arxiv.org/abs/2503.20314)**,patch size = 1。

| Dual-stream blocks | Single-stream blocks | FFN 维度 | Attention heads | Head dim | $(d_t, d_h, d_w)$ |
|---|---|---|---|---|---|
| 8 | 38 | 12288 | 24 | 128 | (16, 56, 56) |

**训练数据。** **120M 授权(licensed)图文对**,caption 是 Gemini 2.5 生成的长结构化 JSON。(数据分布见附录饼图。)

![训练数据类别分布(People 39%、Objects 20%、Transportation 10%、Nature 6%、Animals 6%、Food 5%、Text 4%…)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_data_distribution.png)

**训练流程(附录细节)。**
- **优化器**:AdamW,weight decay $1\times10^{-4}$,$\beta_1=0.9$,$\beta_2=0.999$,$\epsilon=1\times10^{-15}$;
- **学习率**:$1\times10^{-4}$,常数调度,warmup 10K 步(每次升分辨率再 +2.5K warmup);
- **目标**:flow-matching([Lipman et al., 2023](https://arxiv.org/abs/2210.02747)),logit-normal 噪声调度 + 分辨率相关 timestep shifting([SD3](https://arxiv.org/abs/2403.03206));
- **渐进式分辨率训练**:
  - $256^2$:**652K 步**(前 **300K 步用 REPA**,[Yu et al., 2024](https://arxiv.org/abs/2410.06940),DINOv2-L 编码,系数 0.1);
  - $512^2$:100K 步;
  - $1024^2$:50K 步;
  - 每阶段有效 batch size 从约 1K 逐渐增到 2K;
- **后训练**:① 美学微调(3000 张精选图);② **DPO 训练**([Wallace et al., 2024](https://arxiv.org/abs/2311.12908))配 **dynamic beta**([Liu et al., 2025](https://arxiv.org/abs/2501.13918))以改进文字渲染。

---

## 4. 数据构建

### 4.1 数据来源
- **训练图像**:120M **授权(licensed)** 图文对 —— 论文将「100% 授权数据」作为伦理卖点(见 §伦理声明),宣称大幅降低版权侵权风险。来源/许可证/时间范围的更细粒度未披露。
- **caption**:由 Gemini 2.5 在每张图上合成生成(非人写)。
- **美学分数**:PickScore + LAION-Aesthetics 两个预测器。

### 4.2 caption 生成流水线(逐步)
1. **输入**:一张训练图像;
2. **VLM**:Gemini 2.5,配「Visual Art Director」system prompt(见 §4.4 逐字还原);
3. **结构化输出**:启用 Gemini structured outputs,绑定预定义 JSON schema → 强制完整、一致、符合 schema;
4. **输出**:一条平均 1160 词的结构化 JSON caption;
5. **后处理**:对主观字段(美学)不信任 VLM,改用两个美学打分器的分数补充进 caption。

> 注:论文**未披露 caption 生成的 yield(过滤率)**,也未说明对生成 caption 是否做了规则/人工过滤或与评测集的去污染(decontamination)。这是可复现性上的缺口。

### 4.3 标注方法学(人评部分)
人评只出现在**评测**环节(TaBR + user preference),非训练数据标注:
- **TaBR**:60 张测试图(不在训练集);问题"哪张更像原图?";
- **user preference**:150 条 caption,跨多种视觉/风格域;问题"你更喜欢哪张?";
- 两项研究都**随机化图序与模型身份**;
- **总票数 > 5000**,以保证稳健;
- 报告每个模型的 **win rate(胜出题目占比)** 与 **draw rate(平局占比)**。
- **未披露**:标注者人数/资质/报酬/地域、每对图被多少人评、IAA(标注者间一致性)。

### 4.4 合成数据(System Prompt 逐字还原)

论文在附录给出了生成结构化 caption 的完整 system prompt(Table:System prompt)。关键约束逐字摘录:

> *"You are a meticulous and perceptive Visual Art Director working for a leading Generative AI company... The output MUST be ONLY a valid JSON object. Do not include any text before or after the JSON object."*
>
> *"IMPORTANT: When describing human body parts, positions, or actions, always describe them from the PERSON'S OWN PERSPECTIVE, not from the observer's viewpoint."*(人体部位从**人物自身视角**而非观察者视角描述 —— 如左臂始终指人物自己的左臂。)

字段级约束(摘):`short_description` ≤200 词;`objects` 最多列 5 个显著对象(超过则归入背景),每个对象 `description` ≤100 词;人物追加 pose/expression/clothing/action/gender/skin_tone_and_texture;物体簇追加 `number_of_objects`(整数);`text_render` 最多 5 个;`artistic_style` ≤3 词。

**开源 VLM 的合成训练数据**:用合成短 prompt + 编辑指令,schema 与图像模型一致(详见 §3.1 末)。

### 4.5 最终数据统计

| 项目 | 数值 |
|---|---|
| 训练图文对 | 120M(授权) |
| caption 平均长度 | 1160.3 tokens(中位 1176,std 316.4,范围 192–1800) |
| 美学微调集 | 3000 张精选图 |
| TaBR 测试集 | 60 张真实图(训练集外) |
| user-preference 测试集 | 150 条 caption |
| 类别分布 | People 39% / Objects 20% / Transportation 10% / Nature 6% / Animals 6% / Food 5% / Text 4% / 其余 |

### 4.6 评测协议(TaBR 作为新基准)
见 §3.3 与 §5.1。TaBR 实质上引入了一个新的评测范式,但论文**未明确承诺公开 TaBR 评测数据集**(源码注释中作者间还在讨论是否提供 dataset)。

### 4.7 已知偏置/局限
- caption 由单一 VLM(Gemini 2.5)生成 → 继承该 VLM 的描述偏好与盲点;
- 数据类别分布偏向 People(39%),可能影响非人物域的表现;
- 美学是用打分器代理的,主观维度仍受限。

---

## 5. 实验与评测

### 5.1 评测设置

四个互补协议:
1. **PRISM-Bench-licensed**:[PRISM-Bench, Fang et al., 2025](https://arxiv.org/abs/2509.09680) 的授权变体(滤掉涉版权/未授权概念的 prompt,以与 FIBO 的 100% 授权训练公平对比)。PRISM 专门设计来暴露在旧基准上饱和的对齐差距。**用 GPT-4.1-Vision 自动打分**,分 alignment(Ali.)/aesthetics(Aes.)/average(Avg.),覆盖 Imagination/Text rendering/Style/Affection/Composition/Long text 六个类别。
2. **GenEval**:[Ghosh et al., 2023](https://arxiv.org/abs/2304.13826),用检测器 + CLIP 相似度,探测物体计数、颜色归属、空间关系等(短 prompt)。
3. **TaBR**:见 §3.3,60 张测试图,人评。
4. **Overall user preference**:150 条 caption,盲测并排比较,人评。

后两项共收集 **>5000 票**。

### 5.2 主结果

**Table 3:PRISM-Bench-licensed**(Ali./Aes./Avg.;**粗体**=最佳,下划线=次佳)。下表为各类别的 Avg. 列摘要 + Overall(完整三联值见原表图)。

| 模型 | Imagination | Text rend. | Style | Affection | Composition | Long text | **Overall (Ali./Aes./Avg.)** |
|---|---|---|---|---|---|---|---|
| SD3.5-Large (8B) | 75.3 | 62.2 | 79.7 | 87.0 | 84.9 | 63.5 | 78.8 / 77.6 / 78.2 |
| FLUX.1-dev (12B) | 68.6 | 60.4 | 75.9 | 88.9 | 87.4 | 70.1 | 77.0 / 78.7 / 77.8 |
| FLUX.1-Krea-dev (12B) | 72.4 | 57.6 | 81.9 | 88.2 | 88.3 | 71.5 | 79.7 / 79.7 / 79.7 |
| HiDream-I1-Full (17B) | 74.2 | 68.5 | 84.1 | 89.3 | 88.5 | 58.8 | 80.0 / 79.1 / 79.5 |
| Qwen-Image (20B) | 80.2 | **76.3** | 83.1 | 86.5 | 89.3 | 74.1 | 84.1 / 81.5 / 82.8 |
| *Gemini2.5-Flash-Image(闭源)* | *89.1* | *75.0* | *90.2* | *92.9* | *91.5* | *84.8* | *91.2 / 87.3 / 89.3* |
| **FIBO (Ours)** | 84.0 | 69.3 | 86.9 | 89.4 | 88.3 | 77.7 | **87.8 / 82.1 / 84.9** |
| FIBO open-VLM | 82.3 | 67.0 | 86.8 | 88.4 | 87.6 | 74.8 | 86.4 / 80.9 / 83.6 |

**解读**:FIBO 的 Overall Avg. **84.9 为所有开源模型最高**(Qwen-Image 82.8 次之,且 Qwen 是 20B vs FIBO 8B),并大幅缩小与闭源 Gemini 2.5 Flash Image(89.3)的差距。FIBO 的 alignment(87.8)在开源中最高。值得注意:**用开源 VLM(Qwen-3 VL 4B)生成 prompt 的 FIBO open-VLM 仍达 83.6**,几乎不输闭源 VLM 版本,说明 prompt 转换这一环可以低成本化。

![Table 3:PRISM-Bench-licensed 完整结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_tab_prism.png)

**Table 4:GenEval**(**粗体**=最佳,下划线=次佳)。

| 模型 | Size | Single | Two obj | Counting | Colors | **Position** | Color attr. | **Overall** |
|---|---|---|---|---|---|---|---|---|
| SD3.5-Large | 8B | **1.00** | 0.91 | 0.79 | 0.81 | 0.23 | 0.54 | 0.71 |
| FLUX.1-dev | 12B | 0.99 | 0.80 | 0.75 | 0.77 | 0.19 | 0.49 | 0.66 |
| FLUX.1-Krea-dev | 12B | 0.99 | 0.91 | 0.70 | 0.88 | 0.31 | 0.69 | 0.75 |
| HiDream-I1-Full | 17B | **1.00** | **0.98** | 0.79 | **0.91** | 0.60 | 0.72 | 0.83 |
| Qwen-Image | 20B | 0.99 | 0.94 | **0.91** | 0.87 | 0.77 | **0.75** | **0.87** |
| **FIBO (Ours)** | **8B** | **1.00** | 0.88 | 0.79 | 0.90 | **0.80** | 0.71 | 0.85 |

**解读**:FIBO 以 **8B 参数**取得 Overall 0.85(仅次于 20B 的 Qwen-Image 0.87),其中 **Position(空间位置)0.80 为全场最高** —— 这正是大多数模型的老大难(SD3.5 仅 0.23、FLUX 仅 0.19)。论文归因于结构化 caption 显式编码空间关系。

![Table 4:GenEval 结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_tab_geneval.png)

**Table 5:TaBR 重建保真度**(FIBO 对每个 baseline 的胜率/对手胜率/平局)。

| | vs SD3.5 | vs FLUX-dev | vs FLUX-Krea | vs HiDream | vs Qwen |
|---|---|---|---|---|---|
| **FIBO 胜** | **90.5%** | **76.9%** | **66.4%** | **88.9%** | **59.2%** |
| Baseline 胜 | 2.9% | 8.0% | 13.9% | 4.5% | 22.0% |
| 平局 | 6.6% | 15.1% | 19.7% | 6.6% | 18.8% |

**解读**:FIBO 在 TaBR 上**对所有开源 baseline 全胜**,对 SD3.5 胜率 90.5%、对 HiDream 88.9%,即便对最强的 Qwen-Image 也以 59.2% vs 22.0% 领先。这直接验证了「长结构化 caption 训练带来更强表达力」。

![Table 5:TaBR 胜率](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_tab_tabr.png)

**Table 6:Overall user preference**。

| | vs SD3.5 | vs FLUX-dev | vs FLUX.1-Krea | vs HiDream | vs Qwen |
|---|---|---|---|---|---|
| **FIBO** | **69.1%** | **58.6%** | **47.5%** | 42.3% | **42.1%** |
| Baseline | 19% | 30.4% | 35.1% | **43.5%** | 40.2% |
| 平局 | 11.9% | 11% | 17.4% | 19.7% | 17.7% |

**解读**:在「单纯好不好看」的偏好维度,FIBO **明显胜过 SD3.5、FLUX.1-dev、FLUX.1-Krea**,与 Qwen-Image、HiDream **打平**(对 HiDream 略低 42.3 vs 43.5,对 Qwen 略高 42.1 vs 40.2)。即:FIBO 在**保持有竞争力的感知质量**的同时,拿到了**更强的 prompt 对齐**。

![Table 6:Overall user preference](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_tab_userpref.png)

### 5.3 消融研究

论文的消融嵌在方法节里(已在 §3.1、§3.2 详述),汇总:

| 消融 | 配置 | 结果 | 启示 |
|---|---|---|---|
| **caption 长度**(§3.1) | 1B 模型,100K 步,长 vs 短 caption | FID 19.01 vs 34.04 | 长结构化 caption 收敛更快、质量更高 |
| **融合架构**(§3.2) | 1B 模型,150K 步:T5 / TokenFusion / DimFusion | FID 36.46 / 15.90 / **15.58**;耗时 1.28 / 0.8 / **0.5** s | LLM 融合远胜 T5;DimFusion 兼得最佳 FID 与最低耗时 |

**消融用的小模型架构**(附录 Table):8 dual-stream + 20 single-stream block,FFN 6144,24 heads,head dim 64,$(d_t,d_h,d_w)=(4,30,30)$,用 SDXL VAE。融合消融在 **70M** 授权图文对上训练,有效 batch 1024,REPA 系数 0.5(注意:大模型 FIBO 用 0.1),FID 在 COCO-2014 val 30K 样本上算,50 步推理,无 CFG。

### 5.4 缩放/容量研究
论文**未做系统的模型规模/数据规模缩放曲线**(只有 1B 消融 vs 8B 主模型两个点),这是一个缺口。

### 5.5 定性结果
- **TaBR 重建**(Figure 8/9):FIBO 重建最接近原图,尤其保留姿态、相机角度、场景布局(如赛跑选手的姿态、桥的结构)。
- **Contextual Contradictions**(Figure 3,取自 [ContraBench, Huberman et al., 2025](https://arxiv.org/abs/2506.01929) 与 [Whoops, Bitton-Guetta et al., 2023](https://doi.org/10.1109/ICCV51070.2023.00247)):探测反常识关系。FIBO 能完整保留指定属性与关系(熊倒立、女人用飞镖写字、拳击手劈叉),而其他模型常默认成"合理但错误"的配置。

![Contextual Contradictions:FLUX/HiDream/Qwen/FIBO 对"女人用飞镖写字"、"职业拳击手劈叉"的生成](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_contextual.png)

- **可控编辑**(Figure 4):景深、表情等细粒度控制 —— FIBO 能正确实现浅/深景深,而 FLUX 在同 prompt 下控制失败。

![Control:景深与多人物表情控制,FLUX vs FIBO](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/fibo_control.png)

### 5.6 失败案例
论文未设专门的失败案例章节(弱点)。从数据看,FIBO 在 PRISM 的 **Text rendering(69.3)** 与 **Aesthetics** 维度仍落后 Qwen-Image(76.3),文字渲染是已知短板(故后训练专门用 DPO 改进文字)。

### 5.7 成本与效率
- DimFusion 每步 0.5s vs TokenFusion 0.8s(1.6× 加速)、vs T5 1.28s;
- 开源 VLM 训练:8× H100,2B tokens;
- 融合消融:150K 步,128× H200;
- **未报告 FIBO 主模型的总训练 GPU-hours/天数、推理延迟、显存占用**。

### 5.8 人评
TaBR + user preference,>5000 票,随机化图序与模型身份。但缺 IAA、每对评判人数、标注者背景。

### 5.9 逐基准点评
- **PRISM-licensed**:衡量复杂对齐;FIBO 开源第一,8B 打平/超越 12–20B 对手 → 结构化 caption 的"信息密度"换算成了"参数效率"。
- **GenEval**:衡量短 prompt 组合;FIBO 在 Position 拔尖印证空间显式编码,但 Counting/Two-object 略逊 HiDream/Qwen,说明短 prompt 下的优势不如长 caption 场景明显。
- **TaBR**:衡量长 caption 表达力;FIBO 全胜 —— 这是它"主场"。
- **user-preference**:衡量纯美学;FIBO 打平第一梯队 —— 说明追求可控性**没有**牺牲观感。

---

## 6. 优点

1. **数据范式的清晰假设 + 强证据。** "训练永远完整的长结构化 caption → 消除条件歧义 → 强制接地"这一论点由 FID 消融(19.01 vs 34.04,Figure)与 TaBR 全胜(Table 5)双重支撑,逻辑闭环漂亮。
2. **DimFusion 是实用的工程贡献。** 在嵌入维度而非序列维度融合,既保留双向混合与多层信息,又把长 caption 的 attention 成本压住(1.6× 加速),且 FID 还略优于 TokenFusion(Table/Figure §3.2)—— 几乎是"免费午餐"。
3. **解耦的无监督涌现极具价值。** 固定种子、只改一个 JSON 字段就能精确编辑单一视觉因子(Figure 4/6),且模型从未见过成对图 —— 这为"可预测的图形原语(predictable graphics primitives)"打开了产品想象空间。
4. **TaBR 提供了长 caption 的可操作评测。** 用图像锚点绕开"人读不懂长文本"的死结,比偏好评测更客观,且可系统优化可控性(§3.3)。
5. **伦理立场 + 可复现承诺。** 100% 授权数据 + 公开权重(已在 HuggingFace),在版权敏感的当下是差异化卖点。

---

## 7. 弱点与局限

1. **可复现性的数据缺口。** 120M 训练数据未公开、来源/许可证细节模糊;caption 流水线**未报告 yield、过滤规则、与评测集的去污染**。若 TaBR/PRISM 的图与训练分布重叠,优势可能被高估。**建议**:公开数据卡 + 去污染说明。
2. **人评细节不足。** TaBR/user-preference 缺 IAA、每对评判人数、标注者背景与报酬,统计可靠性无法核验(无置信区间/方差)。**建议**:报告 IAA 与 bootstrap 置信区间。
3. **缺缩放研究。** 只有 1B(消融)与 8B(主模型)两点,无法判断"长 caption 优势是否随规模保持/放大",也未给 token-seen 缩放曲线。
4. **TaBR 的循环依赖偏置。** TaBR 用 captioner 生成 caption 再喂回模型 —— 若 captioner(Gemini)与 FIBO 的训练 caption 同源,FIBO 在"理解这种 caption"上天然占优,对其他模型可能不公平。**建议**:用多个不同 captioner 交叉验证 TaBR。
5. **文字渲染与美学仍落后 Qwen-Image。** PRISM Text rendering 69.3 vs 76.3,需靠 DPO 后训练补救,说明结构化 caption 对文字渲染帮助有限。

---

## 8. 与并行/相关工作对比

| 工作 | 问题框定 | 模型规模 | 数据规模 | 头条指标 | Code/Weights |
|---|---|---|---|---|---|
| **FIBO (本文)** | 长结构化 caption 训练 + 高效融合 + 图锚评测 | 8B | 120M(授权) | PRISM-licensed Overall **84.9**(开源第一) | Weights ✅ / Code ✅ |
| [HiDream-I1](https://arxiv.org/abs/2505.22705) | 稀疏 DiT + 双流 decoder-only LLM(序列拼接) | 17B | 未明确 | PRISM Overall 79.5 | Weights ✅ |
| [Qwen-Image](https://arxiv.org/abs/2508.02324) | 大规模 T2I + 文字渲染强项 | 20B | 大规模 | GenEval **0.87**、PRISM 82.8 | Weights ✅ |
| [SD3](https://arxiv.org/abs/2403.03206) | MM-DiT + flow-matching + 双向混合 | 8B | 大规模 | —(本文 SD3.5 基线) | Weights ✅ |
| [Playground v3](https://arxiv.org/abs/2409.10695) | LLM deep-fusion(cross-attn,无双向混合) | — | — | — | ❌ |

定位:FIBO 在**同等或更小参数**下,凭"caption 信息密度"超越更大的开源模型,且 DimFusion 比 HiDream 的序列拼接更省算力、比 Playground v3 多了双向混合。

---

## 9. 可复现性审计

| 项目 | 是否公开 | 说明 |
|---|---|---|
| Code | ✅ | 论文声明公开 code + full implementation details |
| Weights | ✅ | [HuggingFace briaai/FIBO](https://huggingface.co/briaai/FIBO);消融小模型/open-VLM 是否一并公开未明确 |
| 训练数据 | ❌ | 120M 授权图文对未公开,来源/许可证细节模糊 |
| 评测数据 | ⚠️ | PRISM-Bench-licensed 是 PRISM 的滤过变体(派生方式描述但脚本未明确);TaBR 的 60 图测试集是否公开未承诺 |
| 超参数 | ✅(大体) | 优化器/调度/分辨率阶段/REPA/DPO 都给了;但**总训练步数对应的 GPU 天数、batch 升降的精确 schedule** 不全 |
| 评测/judge prompt | ⚠️ | caption 生成 system prompt 逐字给出 ✅;但 PRISM 用 GPT-4.1-Vision 打分的 **judge prompt 未给**(依赖 PRISM 官方协议) |
| 硬件规格 | ⚠️ | 消融(128× H200)、open-VLM(8× H100)有;**FIBO 主模型训练硬件/总时长未报告** |

**复现裁决:** **中等偏上(权重可用,流程可复述,但数据不可得)。** 凭借公开权重,任何人都能**使用**和**评测** FIBO;但要**从零复现训练**几乎不可能 —— 120M 授权数据是闭源的护城河,且 caption 流水线的 yield/去污染、主模型总算力缺失。DimFusion 与 TaBR 这两个方法贡献描述足够清晰、可独立实现。总体上,这是一篇"方法可复现、产物可复用、但训练不可复刻"的工业界论文。

---

## 10. 总体评价

FIBO 最大的思想价值不在某个模块,而在**重新定义了 T2I 的优化轴**:与其在固定数据集下卷模型/卷偏好,不如**扩展"语言"本身** —— 让每张训练图都配一份标准化、永远完整的"工单"。这一转变带来三个连锁红利:更快收敛、参数效率(8B 打 20B)、以及最意外的**无监督解耦**。配套的 DimFusion 解决了"长 caption 太贵"的工程瓶颈,TaBR 解决了"长 caption 没法评"的评测瓶颈,三者构成一个自洽的体系。

它指向一个更大的设计空间:**用户写自然语言 → 翻译成结构化中间方言 → 解锁可预测的图形控制原语**。如果说当前 T2I 像"对画师喊一句话碰运气",FIBO 的路线更像"给画师一份可逐项修改的规格说明书"—— 这对专业创作工具是正确的方向。主要保留意见是数据闭源与评测自洽性(TaBR 的 captioner 同源偏置),但作为开源社区第一个吃这个螃蟹的工作,贡献是扎实的。
