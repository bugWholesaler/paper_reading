# Multi-Object Sketch Animation by Scene Decomposition and Motion Planning (MoSketch)

> **Authors:** Jingyu Liu, Zijie Xin, Yuhan Fu, Ruixiang Zhao, Bangxiang Lan, Xirong Li (通讯作者)（Renmin University of China）
> **Venue:** arXiv preprint, 2025-03-25 (arXiv:2503.19351, v1)
> **Link:** [arXiv:2503.19351](https://arxiv.org/abs/2503.19351) ・ [HTML](https://arxiv.org/html/2503.19351v1)
> **Code / Weights / Data:** Code 计划开源（"The code will be released."），权重未释放，未发布训练数据集，测试集 60 张多对象 sketch 在补充材料

![Fig 1 — Comparison of different methods for multi-object sketch animation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig1_comparison.png)

---

## TL;DR

MoSketch 是首个面向**多对象矢量草图动画**（multi-object vector sketch animation）的方法：在不引入任何额外训练数据的前提下，把"草图 + 文本指令 → 一段控制点位移序列 ΔZ"这个问题，沿用 [Live-Sketch (Gal et al., CVPR 2024)](https://arxiv.org/abs/2311.13608) 的 **SDS 迭代优化范式**，通过四个模块——**LLM-based Scene Decomposition、LLM-based Motion Planning、Motion Refinement Network、Compositional SDS**——以 divide-and-conquer 思路解决多对象场景下"对象感知运动建模 (object-aware motion modeling)"和"复杂运动优化 (complex motion optimization)"两大难题。在自建的 60 张多对象草图测试集上，MoSketch 在 VBench 的 Text-to-Video Alignment、Sketch-to-Video Alignment、Motion Smoothness、Dynamic Degree 上分别达到 **0.218 / 0.914 / 0.977 / 0.283**，全面超过 Live-Sketch / FlipSketch / CogVideoX / DynamiCrafter。

---

## 1. Background & Motivation

### 1.1 Problem Definition

任务是**文本驱动的草图动画**：给定一张矢量草图 P ∈ ℝ^{n×2}（n 个三次贝塞尔控制点的 2D 坐标）和一段文本 Y，输出一段由控制点位移序列 ΔZ ∈ ℝ^{n×f×2} 描述的短视频（f 帧）。MoSketch 把这个问题专门聚焦到**含多个独立对象**的草图——例如"篮球运动员投篮"、"坦克朝目标射击"——这种场景需要建模相对运动、交互和物理约束。

### 1.2 Why It Matters

- 单对象 sketch 动画（一只猫摇尾巴）几年前已被 [Live-Sketch](https://arxiv.org/abs/2311.13608) 和 [FlipSketch](https://arxiv.org/abs/2412.16209) 较好解决，下一步自然是多对象。
- 多对象动画在 GIF 设计、卡通制作、教学演示中出现的频率更高（绝大多数 GIF 都至少有两个交互的角色）。
- 多对象场景同时对外部 T2V 扩散模型提出"复杂运动建模"挑战，是研究 SDS-based generative animation 的良好测试平台。

### 1.3 Limitations of Prior Work（按论文 §1 / Tab. 1 / Fig. 1 整理）

| 方法 | 草图表征 | Object-aware? | 训练数据 | 优化范式 | 多对象失败原因 |
|---|---|---|---|---|---|
| [FlipSketch (CVPR 2025)](https://arxiv.org/abs/2412.16209) | Raster | No | 需 Live-Sketch 合成数据 fine-tune T2V | DDIM Inversion + Fine-Tune | 微调数据极少包含多对象，DDIM inversion 噪声无法捕获多对象外观，输出 chaos |
| [Live-Sketch (CVPR 2024)](https://arxiv.org/abs/2311.13608) | Vector | No | 无 | SDS | MLP 不分对象建模，T2V 难以引导复杂多对象运动 → "shaky 或几乎静止" / "倒水时瓶里液体反而增加" |
| **MoSketch (本文)** | **Vector** | **Yes** | **无** | **Compositional SDS** | — |

### 1.4 Gap This Paper Fills

作者把 single→multi 的鸿沟归纳成两个具体技术挑战：
1. **Object-aware motion modeling**：相对运动、交互、物理约束（如重力、惯性）必须显式建模到生成网络里。
2. **Complex motion optimization**：当前 T2V 扩散模型本身在复杂多对象运动上不可靠，单一 SDS loss 不足以引导这种场景。

MoSketch 把"先用 LLM 把场景与运动分解 → 再让 T2V 模型分别引导每个简单子运动"作为整体策略，弥合上述 gap。

---

## 2. Related Work

### 2.1 Sketch Animation

- **Manual / template-driven 时代**：K-Sketch [Davis 2008]、Toonsynth [Dvorožnák et al., ToG 2018]、SketchAnim 等需要人工或额外参考视频/骨架。
- **Text-driven，无训练数据时代**：[Live-Sketch (Gal et al., CVPR 2024)](https://arxiv.org/abs/2311.13608) 用 SDS 反向蒸馏 T2V 的运动先验，开启免数据范式；[Yang et al., VCIP 2024](https://arxiv.org/abs/2402.04788) 等沿用此 paradigm 改进。
- **Text-driven，需 fine-tune**：[FlipSketch (Bandyopadhyay & Song, CVPR 2025)](https://arxiv.org/abs/2412.16209) 对 ModelScope T2V 在合成数据上 fine-tune，DDIM 反演保留外观。

### 2.2 Text-guided Image-to-Video Generation

[VideoCrafter1 (Chen et al., 2023)](https://arxiv.org/abs/2310.19512)、[I2VGen-XL (Zhang et al., 2023)](https://arxiv.org/abs/2311.04145)、[CogVideoX (Yang et al., 2024)](https://arxiv.org/abs/2408.06072)、[DynamiCrafter (Xing et al., ECCV 2024)](https://arxiv.org/abs/2310.12190) 是像素域 I2V 的代表。论文指出：把 sketch 当成 image 直接喂给这些 I2V 模型，会因 sketch 与自然图像 domain gap 严重失败（外观无法保持）。

### 2.3 LLM-assisted Compositional Generation

LLM 用于布局/轨迹规划已有大量先例：
- 静态 layout：[LayoutLLM-T2I (Qu et al., 2023)](https://arxiv.org/abs/2308.05095)、[GraphDreamer (Gao et al., CVPR 2024)](https://arxiv.org/abs/2312.00093)。
- 动态轨迹：[VideoDirectorGPT (Lin et al., 2023)](https://arxiv.org/abs/2309.15091)、[LLM-Grounded Video Diffusion (Lian et al., ICLR 2024)](https://arxiv.org/abs/2309.17444)、[GPT4Motion (Lv et al., CVPR 2024)](https://arxiv.org/abs/2311.12631)、[Comp4D (Xu et al., 2024)](https://arxiv.org/abs/2403.16993)、[Trans4D (Zeng et al., 2024)](https://arxiv.org/abs/2410.07155)。
- Scene decomposition + 分而治之：[VideoDirectorGPT](https://arxiv.org/abs/2309.15091)、[Comp4D](https://arxiv.org/abs/2403.16993)、[GraphDreamer](https://arxiv.org/abs/2312.00093) 都验证了把多对象任务拆成单/少对象子任务对生成模型有用。

### 2.4 Positioning

MoSketch 的定位 = **Live-Sketch（SDS-based vector sketch 动画范式） + Comp4D / VideoDirectorGPT 思想（LLM 做 scene & motion 分解，并用 compositional SDS 监督子运动）**。它没有发明新的 backbone 也没有发明新的 SDS 变体，而是把"LLM-as-planner + Compositional SDS"这一现成思路第一次完整地搬到 multi-object **vector sketch** 动画这一具体任务，并在生成网络层面把 Live-Sketch 的非对象感知 MLP 升级为对象感知的 Transformer-based motion refinement network。

---

## 3. Core Method

### 3.0 整体架构（按论文叙事顺序）

![Fig 2 — MoSketch overall pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig2_pipeline.png)

总体上 MoSketch 是 **4 个串行模块**（Fig. 2 中 ❄️ = frozen，🔥 = trainable）：

```
(P, Y)                                                  ┌──────────────────────────────┐
   │                                                    │ Frozen modules (LLM, T2V)    │
   ▼                                                    └──────────────────────────────┘
[3.2.1] LLM-based Scene Decomposition (GPT-4 + Grounding DINO + 距离分配)
   │  ⇒  objects, locations B0, point assignment {P_k}, decomposed instructions {Y_k}
   ▼
[3.2.2] LLM-based Motion Planning (GPT-4 with reasoning step)
   │  ⇒  motion plan B ∈ ℝ^{m×f×4} → coarse object motion ΔZ_c
   ▼
[3.2.3] Motion Refinement Network (Transformer + per-object MLP heads, 唯一 trainable)
   │  ⇒  fine-grained object motion ΔZ_o + fine-grained point motion ΔZ_p
   │      ΔZ = ΔZ_c + ΔZ_o + ΔZ_p
   ▼
[3.2.4] Compositional SDS (T2V 扩散模型 = ModelScope T2V，Frozen)
       L_CSDS = SDS(ΔZ, Y) + Σ_k SDS(ΔZ_k, Y_k)
```

整个系统**没有任何训练阶段**：唯一可学习的就是 Motion Refinement Network 内部的 MLP / Transformer 权重，这些权重通过 SDS 损失对每张测试 sketch **单独迭代优化 500 步**（≈ 1 小时 / RTX 3090 Ti）。这是把 Live-Sketch "per-sample optimization" 范式直接继承下来。

下面按论文 §3.1 → §3.2 顺序逐模块展开。

### 3.1 Preliminaries — Vector Sketch & Live-Sketch 回顾

#### 3.1.1 Vector Sketch Representation

- 每条 stroke = 一条三次贝塞尔曲线，由 4 个控制点定义。
- 整张草图 P ∈ ℝ^{n×2} 由 n 个 2D 控制点参数化。
- 动画结果 ΔZ ∈ ℝ^{n×f×2}（每个控制点在 f 帧的位移），帧数 f = 16。
- 高层抽象：ΔZ ← Model(P, Y)。

#### 3.1.2 Live-Sketch in a Nutshell

Live-Sketch 把 ΔZ 拆成两部分：

- **Sketch-level motion** ΔZ_s：对**整张** sketch 做整体仿射变换（translation 2 + scaling 2 + shearing 2 + rotation 1 = **7 个变换参数**，每帧 7 个，故 B̃ ∈ ℝ^{f×7}）。
- **Point-level motion** ΔZ_p：每个控制点独立的额外位移。

公式 (Eq. 2)：

$$
\begin{aligned}
\hat B, \hat P &\leftarrow \text{MLP}(P) \\
\widetilde B   &\leftarrow \text{MLP}(\hat B) \quad &\widetilde B \in \mathbb{R}^{f\times 7} \\
\Delta Z_s &\leftarrow \text{transformation}(\widetilde B, P) \quad &\in \mathbb{R}^{n\times f \times 2} \\
\Delta Z_p &\leftarrow \text{MLP}(\hat P) \quad &\in \mathbb{R}^{n\times f \times 2} \\
\Delta Z   &\leftarrow \Delta Z_s + \Delta Z_p
\end{aligned}
$$

其中 `transformation(·)` 把 7 个参数视为标准的 2D 仿射，作用到 P 的所有点上得到 f 帧的 ΔZ_s。

#### 3.1.3 SDS Loss

Live-Sketch 用一个冻结的预训练 T2V 扩散模型（[ModelScope T2V (Wang et al., 2023)](https://arxiv.org/abs/2308.06571)）的 score 反向引导 ΔZ：

$$
\mathcal L_{SDS} \leftarrow \text{SDS}(\Delta Z, Y) \tag{3}
$$

迭代最小化 L_SDS，让 ΔZ 渲染出的视频被 T2V 判定为符合 Y。该步骤是可微渲染（vector → raster → diffusion latent）。

---

### 3.2 Multi-object Sketch Animation

#### 3.2.1 Module 1 — LLM-based Scene Decomposition

![Fig 3 — LLM-based Scene Decomposition](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig3_scene_decomp.png)

**Method Deep-Dive Checklist:**

1. **Purpose & placement.** 是后三个模块的**地基**：要识别独立对象、给出每个对象的初始位置、把复杂指令拆成多条简单指令。
2. **Inputs / outputs.**
   - Input: 草图 P ∈ ℝ^{n×2}, 文本 Y。
   - Output:
     - 对象列表 `objects`（数量 m，**约束 m < 7**），m 不固定；
     - 简单指令集合 {Y_k}_{k=1..r}（**约束 r < 5**），每条指令应只涉及一个或少量对象；
     - 初始 bounding box B_0 ∈ ℝ^{m×4}；
     - 控制点到对象的分配 {P_k}_{k=1..m}。
3. **Architecture.** 三步级联：
   - **Object identification & motion decomposition**：GPT-4 一次对话同时返回 objects + decomposed instructions（Fig. 3 上半部分，prompt 详见原论文未公开附录，但功能等价于 [VideoDirectorGPT](https://arxiv.org/abs/2309.15091) 中"identify subjects & decompose actions"。
   - **Object localization**：[Grounding DINO (Liu et al., ECCV 2024)](https://arxiv.org/abs/2303.05499) — 把 sketch 当作图像、把 GPT-4 给出的 object 名称当作 open-vocab text query，得到每个对象的 bounding box B_0（Fig. 3 下半部分）。
   - **Point assignment**：对每条 stroke 取其几何中心 c_s（即 4 个控制点的均值），把 c_s 分配给 bbox 中心距 c_s 最近的对象，将该 stroke 的 4 个控制点都归到该对象。
4. **Mathematical formulation (Eq. 4):**

$$
\begin{cases}
\text{objects},\{Y_k\}_{k=1}^{r} \;\leftarrow\; \text{GPT4}(P, Y) \\
B_0 \;\leftarrow\; \text{grounding}(P, \text{objects}) \\
\{P_k\}_{k=1}^{m} \;\leftarrow\; \text{assign}(B_0, P)
\end{cases}
\tag{4}
$$

5. **Loss.** Not specified — 该模块没有可训练参数，无 loss。
6. **Training procedure.** Frozen / inference-only。GPT-4 与 Grounding DINO 均为 off-the-shelf。
7. **Inference procedure.** GPT-4 单次调用（论文未给具体 system prompt 模板，**Not specified in the paper**）；Grounding DINO 默认阈值；assignment 是闭式几何运算（最近邻），不涉及随机性。
8. **External models used.**
   - GPT-4（论文写作"GPT4"，未指明是否为 GPT-4 Turbo / GPT-4o，**Not specified**）。
   - Grounding DINO（[ECCV 2024](https://arxiv.org/abs/2303.05499)）。
9. **Design choices.**
   - 强约束 m < 7、r < 5：避免 LLM 输出爆炸式膨胀，也避免 Grounding DINO 在草图上召回过多噪声 box；
   - 选择 stroke 中心 + bbox 中心的距离做 assignment（而非"点是否在 box 内"）：作者在 §4.2 强调 sketch 的边界很稀疏，stroke 跨越 bbox 时会失败，因此用整条 stroke 的中心更鲁棒；
   - 限制每条简单指令"只涉及一个或少量对象"：直接来自 [VideoDirectorGPT](https://arxiv.org/abs/2309.15091) / [Comp4D](https://arxiv.org/abs/2403.16993) 的经验。

10. **Pseudocode**:

    ```python
    objects, {Y_k} = GPT4(P, Y)            # 1 次 LLM 调用
    B_0 = GroundingDINO(P_raster, objects) # m 个 bbox
    P_k = {}
    for stroke in strokes(P):              # 共 n/4 条 stroke
        c_s = mean(stroke.control_points)
        k_star = argmin_k ||c_s - center(B_0[k])||
        P_k[k_star] += stroke.control_points
    ```

#### 3.2.2 Module 2 — LLM-based Motion Planning

![Fig 4 — LLM-based Motion Planning](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig4_motion_planning.png)

**Method Deep-Dive Checklist:**

1. **Purpose & placement.** 解决"object-aware motion modeling"挑战的第一步：**预先**用 GPT-4 的物理 / 交互先验，给出每个对象 m 在 f 帧上的 bbox 轨迹，作为**粗粒度对象级运动**。它会被后续 Refinement Net 进一步加细。
2. **Inputs / outputs.**
   - Input: P, Y, B_0 ∈ ℝ^{m×4}。
   - Output: 运动计划 B ∈ ℝ^{m×f×4}（每个对象在 f=16 帧的 4 维 bbox），从而推导粗运动 ΔZ_c ∈ ℝ^{n×f×2}。
3. **Architecture.** GPT-4 二次调用，以 Fig. 4 中的两步式结构：
   - **Motion Reasoning**：先让 GPT-4 输出**自然语言推理**——对每个对象列出 X 坐标和 Y 坐标的趋势（增 / 减）以及触发原因。例如 "basketball player: X-coordinate ↑ (inertia), Y-coordinate ↑ (jumping)"、"basketball: thrown then gravity pulls down"、"hoop: coordinates not change"。这一步遵循 [LLM-Grounded Video Diffusion (Lian et al., 2024)](https://arxiv.org/abs/2309.17444) 的 **"reasoning before answer"** 思路，在生成数值化 motion plan 前显式让 LLM 做物理 / 交互推理（惯性、重力、相对静止）。
   - **Motion Planning**：基于上一步的推理，输出 16 帧的 bbox 数值序列 B。
4. **Mathematical formulation (Eq. 5):**

$$
\begin{cases}
B \;\leftarrow\; \text{GPT4}(P, Y, B_0) \\
\Delta Z_c \;\leftarrow\; \text{gather}(\{P_k\}_{k=1}^{m}, B)
\end{cases}
\tag{5}
$$

   `gather` 的具体含义：对第 k 个对象，把 frame-t 的 bbox B[k,t] 与 frame-0 的 B_0[k] 比较，推出整体平移 + 缩放（4 个数 → 2D 仿射的子集），把该仿射作用到对象内的所有控制点 P_k 上，得到这些控制点在 frame-t 的位移；对所有 (k, t) 拼接得到 ΔZ_c ∈ ℝ^{n×f×2}。

5. **Loss.** 无 — frozen LLM。
6. **Training.** 无。
7. **Inference.** 一次 GPT-4 多轮对话（先 reasoning，再 planning）。论文未公开 prompt（**Not specified in the paper**），但功能上和 [LLM-Grounded Video Diffusion](https://arxiv.org/abs/2309.17444) 中的 LLM layout reasoner 高度一致。
8. **External models.** GPT-4（同 §3.2.1）。
9. **Design choices & alternatives.**
   - 用 bbox 序列而不是 trajectory of single anchor point：4 维 bbox 同时表达"平移"与"缩放"，可以反映"球被抛出后变小（远离镜头）/ 变大（靠近镜头）"等物理细节。
   - 在 plan 之前显式 reasoning：作者在 §4.3 ablation 之外没单独烧 reasoning step（只整体 ablation 了"是否有 ΔZ_c"），但引述 [Lian et al., ICLR 2024](https://arxiv.org/abs/2309.17444) 表明 reasoning 显著提升 LLM 对物理约束的遵循度。
   - 没有让 LLM 直接给 control-point 级轨迹：因为 LLM 对几百到几千个 2D 点的精细位置不擅长，bbox 是一个能与几何 grounding 相结合的合适抽象层级。

#### 3.2.3 Module 3 — Motion Refinement Network

![Fig 5 — Motion Refinement Network](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig5_motion_refine.png)

**Method Deep-Dive Checklist:**

1. **Purpose & placement.** 唯一可学习的模块；负责两件事——
   - 在 ΔZ_c 之上加**对象级细化** ΔZ_o（修正 LLM 给的粗 bbox 轨迹中的几何不准确，例如形状的 shear / rotation）；
   - 生成对象内部的**点级运动** ΔZ_p（如手臂摆动、火焰晃动），这部分 LLM 完全无法给。
2. **Inputs / outputs.**
   - Inputs: P ∈ ℝ^{n×2}（所有控制点）；motion plan B ∈ ℝ^{m×(f×4)}。
   - Outputs: ΔZ_o ∈ ℝ^{n×f×2}（fine-grained 对象级仿射运动），ΔZ_p ∈ ℝ^{n×f×2}（fine-grained 点级位移），合成 ΔZ = ΔZ_c + ΔZ_o + ΔZ_p。
3. **Architecture.** Transformer + per-object MLP heads（Fig. 5）：
   - 输入两个分支：B（object 分支，m 条）经 MLP → m × d；P（point 分支，n 条）经 MLP → n × d。
   - 拼接 (m + n) × d 一起喂入 **2 层 Transformer**，建模对象 ↔ 对象、对象 ↔ 点之间的关系。
   - 输出再分两个分支：
     - 对象嵌入 B̂ ∈ ℝ^{m×d}：每个对象 B̂_k 经**专属 MLP_k** 预测 7 维 × f 帧的仿射参数 B̃_k ∈ ℝ^{f×7}，作用到对应控制点 P_k 上得到 ΔZ_o；
     - 点嵌入 P̂ ∈ ℝ^{n×d}：按 object 分组喂入 **不同 MLP_k** 直接预测 ΔZ_p（这就是论文说的"将 Live-Sketch 的 point-level motion 改成 object-aware"——同一对象内部的点共享一组 MLP 权重，但不同对象用不同 MLP）。
   - **Hidden dim d = 128，Transformer 层数 L = 2，f = 16**。
4. **Mathematical formulation (Eq. 6 + 7):**

$$
\hat B, \hat P \leftarrow \text{Transformers}(\text{MLP}(B), \text{MLP}(P)) \tag{6}
$$

$$
\begin{cases}
\{\hat B_k\}_{k=1}^{m}, \{\hat P_k\}_{k=1}^{m} \leftarrow \hat B, \hat P \\
\{\widetilde B_k\}_{k=1}^{m} \leftarrow \{\text{MLP}_k(\hat B_k)\}_{k=1}^{m} \\
\Delta Z_o \leftarrow \{\text{transformation}(\widetilde B_k, P_k)\}_{k=1}^{m} \\
\Delta Z_p \leftarrow \{\text{MLP}_k(\hat P_k)\}_{k=1}^{m} \\
\Delta Z \leftarrow \Delta Z_c + \Delta Z_o + \Delta Z_p
\end{cases}
\tag{7}
$$

5. **Loss.** 由 §3.2.4 的 **Compositional SDS** 提供，模块本身不引入额外 loss。
6. **Training procedure.** 只在测试时迭代优化（per-sample）：
   - **Optimizer**：Adam，lr = 5e-3，weight decay = 1e-2；
   - **Steps**：500（每个测试样本独立优化）；
   - **Hardware**：单卡 RTX 3090 Ti，**约 1 小时 / 样本**；
   - **Batch / sequence length / warmup / lr schedule**：Not specified in the paper；
   - **Initialization**：从随机权重开始（沿用 Live-Sketch 的 per-sample optimization 习惯）。
7. **Inference procedure.** 优化收敛后，最后一次 forward 给出 ΔZ；可微渲染成 16 帧视频。论文未提 SDS 内部 timestep schedule（**Not specified**），按 Live-Sketch 默认。
8. **External models used.** 仅渲染时用 [DiffVG](https://github.com/BachiLi/diffvg) 或类似可微 vector 渲染器（论文未点名，**Not specified**），以及冻结的 ModelScope T2V 给 SDS 信号。
9. **Design choices.**
   - 用 Transformer 取代 Live-Sketch 的纯 MLP：Transformer 通过自注意力自然建模"对象之间的相对关系"，是把 motion modeling 由 sketch-level 升到 object-level 的关键。
   - 给每个对象专属 MLP head：避免不同对象（人 / 球 / 篮筐）的几何分布相互干扰；ablation Setup #4 (w/o Object-aware) 显示移除该设计后 dynamic motion 严重不足（球未被发射 → "shell does not explode"）。
   - 显式引入 ΔZ_c（即 LLM 粗运动的残差结构）：让 Transformer 只学习"如何修正 LLM"，比从零回归整段 ΔZ 容易得多；ablation Setup #1 显示移除 ΔZ_c 后 Dynamic Degree 从 0.283 暴跌到 0.083。

#### 3.2.4 Module 4 — Compositional SDS

**Method Deep-Dive Checklist:**

1. **Purpose & placement.** 解决"complex motion optimization"挑战：T2V 模型对"两个篮球运动员对抗 + 篮球飞向篮筐 + 篮网晃动"这类复合事件不可靠，但对每个单独的子事件（"运动员投篮"、"球飞向篮筐"）很擅长。Compositional SDS 让每个子运动单独享有 SDS 监督。
2. **Inputs / outputs.**
   - Input: ΔZ ∈ ℝ^{n×f×2}（当前候选动画），Y（原指令），{Y_k}（分解后的子指令，r 条），点-对象分配 {P_k}。
   - Output: 复合 SDS 损失 L_CSDS（标量）。
3. **Architecture.** 没有新参数，只是一个 loss 编排器：
   - 对每条子指令 Y_k，从 ΔZ 中**抽取**一个子动画 ΔZ_k——只保留 Y_k 涉及的对象的控制点位移（其他对象冻结在初始 sketch）。这就是 Eq. 8 中的 `decompose(ΔZ, Y_k)`。
   - 用同一个冻结 T2V 模型对 (ΔZ_k, Y_k) 计算一个独立的 SDS loss L_{SDS-k}。
   - 把 r 个子 SDS loss 与原 SDS loss 累加。
4. **Mathematical formulation (Eq. 8):**

$$
\begin{cases}
\{\Delta Z_k\}_{k=1}^{r} \leftarrow \{\text{decompose}(\Delta Z, Y_k)\}_{k=1}^{r} \\
\{\mathcal L_{SDS\text{-}k}\}_{k=1}^{r} \leftarrow \{\text{SDS}(\Delta Z_k, Y_k)\}_{k=1}^{r} \\
\mathcal L_{SDS} \leftarrow \text{SDS}(\Delta Z, Y) \\
\mathcal L_{CSDS} \leftarrow \mathcal L_{SDS} + \sum_{k=1}^{r} \mathcal L_{SDS\text{-}k}
\end{cases}
\tag{8}
$$

   注意：所有项**等权相加**（**未引入额外的权重超参**，论文未做 weight sweep）。这是简化设计也是潜在改进点。

5. **Loss.** 即 L_CSDS，最终被反传到 Motion Refinement Network 的所有可学习参数。
6. **Training.** 同 §3.2.3 的 500 步 Adam 优化。
7. **Inference.** 不参与推理，仅用于训练阶段反向梯度计算。
8. **External models.** [ModelScope T2V (Wang et al., 2023)](https://arxiv.org/abs/2308.06571)（论文 ref [17]/[34]），冻结。
9. **Design choices.**
   - 用 sub-video 而非 sub-sketch：保留所有非 Y_k 涉及的对象在画面里（只是不动），这样 T2V 看到的还是一个"完整场景"，避免 prompt 与画面对象数量错配；
   - 对每条子指令独立 SDS：取自 [Comp4D](https://arxiv.org/abs/2403.16993) / [Trans4D](https://arxiv.org/abs/2410.07155) / [GraphDreamer](https://arxiv.org/abs/2312.00093) 的 compositional SDS 思想；
   - 不衰减 / 不 schedule 各子项权重：后续工作可探索（论文未做）。

10. **Pseudocode**:

    ```python
    for step in range(500):
        ΔZ_o, ΔZ_p = MotionRefinementNet(P, B)
        ΔZ = ΔZ_c + ΔZ_o + ΔZ_p
        L = SDS(render(P + ΔZ), Y)
        for k in range(r):
            ΔZ_k = decompose(ΔZ, Y_k)  # 只保留 Y_k 涉及对象的位移
            L += SDS(render(P + ΔZ_k), Y_k)
        L.backward()
        Adam.step()
    ```

### 3.3 Intuitive Explanation

可以把 MoSketch 类比为一个**剧组**：

- **导演（GPT-4 in Scene Decomp）** 阅读剧本（Y）后，先把"群戏"分成几条单人 / 小组镜头（{Y_k}），并指认每个角色是谁、各占舞台哪块（B_0 与 P_k）。
- **编舞（GPT-4 in Motion Planning）** 在脑里跑一遍物理（重力、惯性、相对位置），给每个角色画一份**走位图**（B），决定从第 1 帧到第 16 帧每个角色的中心和身高/身宽如何变化（这就是 4 维 bbox 序列）。
- **演员训练师（Motion Refinement Net）** 拿走位图当蓝本，让每个角色的肢体细节、布料晃动等内部运动 ΔZ_p 与外部细化 ΔZ_o 都符合走位。
- **多位评分官（Compositional SDS）** 分别评分整组镜头与每个分镜：评分官（T2V 模型）擅长打分小镜头，所以分镜评分会更准；总评 + 分镜评分共同反传给训练师，让训练师在 500 次试演后给出最终成片。

---

## 4. Data Construction

### 4.1 Data Sources

MoSketch **不需要训练数据**（per-sample SDS 优化，模型权重每次从零开始）。论文唯一构建的"数据"是**评测集**——60 张多对象矢量草图 + 配套文本描述。

数据源（§4.1）：

| 来源 | 用途 | 规模 | 备注 |
|---|---|---|---|
| 自然图像（人 / 动物 / 物体三类） | sketch 转化原图 | 至少 60 张，每张含 ≥ 2 个对象 | "随机选自" — 论文未指明具体图像 corpus，**Not specified in the paper** |
| GPT-4 | 为每张草图生成隐式 motion 指令 | 60 条文本 | prompt 模板**未公开**（**Not specified**） |

### 4.2 Pipeline Step-by-Step

完整收集管线（§4.1 "Testing Data Creation"）：

```
[Step 1] 从 {human, animal, object} 随机选含 ≥2 对象的自然图像 (yield: 60 张)
                       │
                       ▼
[Step 2] CLIPasso (ToG'22) 将图像转为矢量草图  (yield: 60 张矢量 sketch)
   ── tool: CLIPasso，开源
   ── 选择该工具因其专门为"semantically-aware object sketching"设计
                       │
                       ▼
[Step 3] GPT-4 给每张 sketch 生成隐式包含动作的文本 (yield: 60 条文本指令)
   ── 例如："A basketball player takes a jump shot, aiming for
            the hoop, with the basketball mid-air and heading
            towards the hoop."
                       │
                       ▼
[Step 4] 论文作者人工把 60 张/条 数据分到 sports / dining / transportation / work 等子场景
   ── 用于定性展示，未用于定量评测分桶
```

每一步**没有过滤**（论文未提丢弃比例），**yield = 60**。

### 4.3 Annotation Methodology

- 没有人工标注的运动 ground-truth（multi-object sketch animation 没有公认 GT）。
- 文本指令由 GPT-4 自动生成，**未做人工校对**（论文未声明 IAA 或质量复核步骤，**Not specified in the paper**）。
- 评测全程依赖 [VBench (Huang et al., CVPR 2024)](https://arxiv.org/abs/2311.17982) / [VBench++ (Huang et al., 2024)](https://arxiv.org/abs/2411.13503) 的 4 个自动指标，没有 human evaluation。

### 4.4 Synthetic / Model-Generated Data

- **CLIPasso (Vinker et al., ToG 2022)**：[arxiv 2202.05822](https://arxiv.org/abs/2202.05822)。把自然图像 → 矢量 sketch 的开源工具。
- **GPT-4 文本指令**：论文未公开 prompt（**Not specified in the paper**）；可推测形如 "Describe the most likely action happening between these objects, in one sentence."
- **去污染**：因为评测不依赖训练集，所以 train/test contamination 不是问题；但**多个 baseline（如 FlipSketch）确实在 Live-Sketch 合成数据上 fine-tune 过**，论文未声明这些数据是否包含 test 重合的草图（**Not specified**）。

### 4.5 Final Statistics

| 项 | 值 |
|---|---|
| 测试集大小 | 60 张多对象矢量草图 |
| 类别分布 | human / animal / object（各类比例未给） |
| 场景分桶 | sports, dining, transportation, work 等（具体每桶数量未给） |
| 草图分辨率 | 224×224（CLIPasso 默认；论文未明示，**Not specified**） |
| 动画帧数 f | 16 |
| 每张草图控制点数 n | 不固定（CLIPasso 输出，每条 stroke 4 个控制点） |
| 训练集 | 无 |
| 验证集 | 无 |

补充材料图 12-17（论文 p10–15）展示了若干测试草图样例，分别属于 "object" / "animal" / "human" 三类。

### 4.6 Benchmark Protocol

- **Task**：给定 (P, Y) 输出 16 帧动画。
- **Scoring**：从 VBench / VBench++ 直接搬运 4 个自动指标，下节详述。
- **Judge model**：VBench 自带（CLIP-based 文本-视频对齐 + DINO-based 主体一致性 + 光流-based 运动平滑 + 帧间差异-based 动态度）；论文未单独指定 judge，沿用 VBench 默认。
- **Prompt for judge**：直接用 Y 与 P；论文未给定专有 prompt（**Not specified**）。
- **Seeds / Randomization**：未声明（**Not specified**）。SDS 含随机扩散 timestep 采样，但论文未提是否用固定 seed。

### 4.7 Known Biases / Limitations of Test Set

作者未在论文里专章讨论，但合理推测：
- 60 张样本量偏小，统计显著性弱；
- 全部由 GPT-4 写指令，对 "fight"、"dance" 等细粒度动作覆盖不全；
- CLIPasso 主要针对静物 sketch 表现好，对"复杂背景中前景对象"召回的 stroke 可能漏拆；
- 三大类（human/animal/object）混合体现了多样性，但每张都至少 2 个对象，单对象 sketch（图 8 的鸽子滑行）只是定性展示，没有进入定量评测。

---

## 5. Experiments & Evaluation

### 5.1 Setup

| 项 | 值 |
|---|---|
| 测试集 | 自建 60 张多对象矢量草图（§4） |
| 帧数 f | 16 |
| 优化步 | 500 |
| Optimizer | Adam, lr = 5e-3, wd = 1e-2 |
| 隐藏维 d | 128 |
| Transformer | 2 层 |
| 硬件 | 单卡 RTX 3090 Ti，约 1 h / 样本 |
| T2V 引导模型 | [ModelScope T2V (Wang et al., 2023)](https://arxiv.org/abs/2308.06571) |
| LLM | GPT-4（具体版本未指明） |
| 视觉 grounding | [Grounding DINO (ECCV 2024)](https://arxiv.org/abs/2303.05499) |
| 矢量化工具 | [CLIPasso (ToG 2022)](https://arxiv.org/abs/2202.05822) |

**Baselines（4 个）**：

| Baseline | 类别 | 链接 |
|---|---|---|
| Live-Sketch (CVPR 2024) | Vector sketch animation, SDS-based | [arxiv 2311.13608](https://arxiv.org/abs/2311.13608) |
| FlipSketch (CVPR 2025) | Raster sketch animation, fine-tune T2V | [arxiv 2412.16209](https://arxiv.org/abs/2412.16209) |
| CogVideoX (2024) | T2V / I2V，DiT-based | [arxiv 2408.06072](https://arxiv.org/abs/2408.06072) |
| DynamiCrafter (ECCV 2024) | I2V, dual-stream image injection | [arxiv 2310.12190](https://arxiv.org/abs/2310.12190) |

**Metrics（4 个，全部来自 [VBench](https://arxiv.org/abs/2311.17982) / [VBench++](https://arxiv.org/abs/2411.13503)）**：

1. **Text-to-Video Alignment** = VBench 的 *Overall Consistency*（基于 CLIP / ViCLIP，0-1）。
2. **Sketch-to-Video Alignment** = VBench 的 *I2V Subject*（基于 DINO 主体一致性）。
3. **Motion Smoothness** = VBench 的 *Motion Smoothness*（基于 [AMT](https://arxiv.org/abs/2304.01410) 的光流帧插值平滑度）。
4. **Dynamic Degree** = VBench 的 *Dynamic Degree*（基于 RAFT 光流的运动幅度）。

### 5.2 Main Results

**Table 2 — Quantitative comparisons for multi-object sketch animation**

![Table 2](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_tab2_main.png)

| Method | Text-to-Video Alignment ↑ | Sketch-to-Video Alignment ↑ | Motion Smoothness ↑ | Dynamic Degree ↑ |
|---|---|---|---|---|
| CogVideoX | 0.141 | 0.610 | 0.747 | — |
| DynamiCrafter | 0.184 | 0.771 | 0.868 | — |
| FlipSketch | 0.199 | 0.704 | 0.839 | — |
| Live-Sketch | 0.207 | 0.897 | 0.956 | 0.266 |
| **MoSketch** | **0.218** | **0.914** | **0.977** | **0.283** |

> 论文备注：CogVideoX / DynamiCrafter / FlipSketch 因为不能保留外观（Sketch-to-Video Alignment 已经低），其 Dynamic Degree 没有可解释意义，因此用 "—" 表示。

**Per-benchmark commentary**：

- **Text-to-Video Alignment**：MoSketch 0.218 vs. Live-Sketch 0.207（+0.011，相对 +5.3%）。这一项考察整段动画是否符合 Y——MoSketch 的 LLM 分解+CSDS 直接受益于细分子指令的强约束。
- **Sketch-to-Video Alignment**：MoSketch 0.914 vs. Live-Sketch 0.897（+0.017）。这一项考察"主体不变形"——矢量表示天然有利，MoSketch 的 object-aware MLP 进一步避免对象失去识别度（FlipSketch 0.704 因为 raster 表征在 fine-tune 后会改色彩 / 几何）。
- **Motion Smoothness**：MoSketch 0.977 是最高；论文解释加入 LLM 粗运动 ΔZ_c 后帧间过渡更连贯（粗运动等价于一个低频先验，抑制 SDS 高频抖动）。
- **Dynamic Degree**：Live-Sketch 0.266 → MoSketch 0.283（+0.017）。MoSketch 在更平滑的同时仍然有更大的动态——这是 LLM 粗运动注入"长程位移"的功劳，避免 Live-Sketch 那种"在原地颤抖"的退化。
- **CogVideoX / DynamiCrafter** 两个 I2V baseline 在 Sketch-to-Video Alignment 上崩溃（0.610 / 0.771），说明把 sketch 当 image 喂给 I2V 模型不可行。

### 5.3 Qualitative Comparison

![Fig 9 — Qualitative comparisons](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig9_compare.png)

主要现象：
- **CogVideoX / DynamiCrafter**：完全不像 sketch，输出更像自然图像（domain gap 致命）。
- **FlipSketch**：输出在 sketch domain 内，但人物外观大幅漂移（raster + DDIM inversion 信息丢失）。
- **Live-Sketch**：保持外观但"几乎不动"（ shaky / static）。
- **MoSketch**：保持外观且产生合理的多对象交互（hurdle race 中的几个跑者动作连贯）。

### 5.4 Qualitative Showcase（论文图 6）

![Fig 6 — Qualitative results across scenarios](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig6_qualitative.png)

论文展示了 3 个典型场景：rock climbing、basketball playing、tank target shooting。每行都同时显示 LLM 给出的 ΔZ_c（粗 bbox 轨迹）和最终 ΔZ（控制点级动画）。值得注意的细节：
- **篮球场景**：球离手抛物线 → 进网；篮筐保持静止；手臂细节 ΔZ_p 有摆动。
- **坦克场景**：炮弹离开炮口、变大撞击目标、爆炸扩散——这种"shell explodes when it collides with the target"是 ΔZ_o 把 LLM 给的"炮弹 bbox 沿水平方向变大"再细化为爆炸辐射状散开的效果。
- **robust to local errors**：如 "smoke" 没被 Grounding DINO 框出 / 炮弹 stroke assignment 不准时，最终结果仍然合理。这说明系统对单点错误有冗余度（多 SDS loss 互相约束）。

### 5.5 Diversity & Single-Object Generalization

![Fig 7 / Fig 8 — Diverse and single-object animations](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig7_8_diversity.png)

- **Fig 7**：同一张"公交车 + 自行车"草图，仅改 Y，可以得到"公交车与自行车并排前进"、"自行车超过公交车"、"公交车跟随自行车"三种结果；说明 LLM 分解 + CSDS 真正反映 Y。
- **Fig 8**：在单对象草图（企鹅滑行 / 冲浪手）上，MoSketch 利用 LLM 粗运动 ΔZ_c 实现 Live-Sketch 难做到的"大幅位移"——这是 motion plan 模块给出的副产品（即便 m=1 时它仍提供轨迹先验）。

### 5.6 Ablation Studies

![Fig 10 — Qualitative ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig10_ablation.png)

**Table 3 — Quantitative ablation of MoSketch**

![Table 3](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_tab3_ablation.png)

| # | Setup | Text-to-Video ↑ | Sketch-to-Video ↑ | Motion Smoothing ↑ | Dynamic Degree ↑ |
|---|---|---|---|---|---|
| 0 | **Full** | **0.218** | 0.914 | **0.977** | **0.283** |
| 1 | w/o ΔZ_c | 0.212 | 0.955 | 0.959 | 0.083 |
| 2 | w/o ΔZ_o | 0.212 | 0.909 | 0.964 | 0.266 |
| 3 | w/o ΔZ_p | 0.203 | **0.971** | 0.971 | 0.200 |
| 4 | w/o Object-aware | 0.205 | 0.932 | 0.968 | 0.266 |
| 5 | w/o CSDS | 0.207 | 0.911 | 0.966 | 0.267 |

> 论文备注：Setup #1 / #3 / #4 在 Sketch-to-Video Alignment 上"超过 Full"是因为对象几乎没动（外观 100% 保留），属于退化解，需结合 Dynamic Degree 一起看。

逐行解读：

- **#1 (w/o ΔZ_c)**：去掉 LLM 粗运动后，Dynamic Degree 直接从 0.283 → 0.083（暴跌 71%）。意味着 SDS 自己很难驱动多对象大幅运动，LLM 粗运动是 dynamic 的关键来源。
- **#2 (w/o ΔZ_o)**：少了对象级仿射细化 → 炮弹未被放大爆炸，Sketch-to-Video Alignment 0.909（与 Full 相近），但 Text-to-Video 微降。
- **#3 (w/o ΔZ_p)**：去掉点级位移 → 对象内部完全静止；Sketch-to-Video 反而最高（0.971，因为外观 100% 保留），但 Dynamic Degree 显著下降到 0.200。
- **#4 (w/o Object-aware)**：把 Refinement Net 替换回 Live-Sketch 那套不分对象的 MLP；炮弹能发射但不爆炸（细节缺失），Text-to-Video 0.205。
- **#5 (w/o CSDS)**：去掉 Σ_k SDS_k 子项 → Text-to-Video 0.207、Dynamic Degree 0.267，相对 Full 都明显下滑，说明 compositional 监督对复杂场景细节至关重要。

### 5.7 Failure Cases & Limitations

![Fig 11 — Limitations](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mosketch_fig11_limitations.png)

论文承认三类失败模式：
1. **错误的 point assignment**：例 Godzilla + 城市，Godzilla 的尾巴被错分给 "city"，Refinement Net 也救不回来。
2. **错误的 coarse motion plan**：例足球场景，LLM 安排守门员朝错误方向走 → ΔZ_o 不能纠正方向性大错。
3. **T2V 模型自身的语义盲区**：例 "fight" 这种语义在 ModelScope T2V 中不可靠，整体输出失败。

### 5.8 Statistical Reliability / Cost / Human Eval

| 项 | 状态 |
|---|---|
| Multi-seed averaging | **未声明**（论文未提是否多 seed 平均，**Not specified**） |
| Confidence interval / std | **未给出** |
| Train cost | 0（per-sample 优化） |
| Inference / 优化 cost | **~1 hour / 样本，单 RTX 3090 Ti** |
| Memory | 未公开（**Not specified**） |
| Human evaluation | **未做** |

### 5.9 Per-Benchmark Commentary（汇总）

- 60 张测试集 + 4 个自动指标，统计可靠度 **较弱**；定量优势在 Text-to-Video 上 +0.011 / Dynamic Degree +0.017，幅度小但方向一致。
- 主要价值在定性结果：成功生成 Live-Sketch 完全做不到的 large-displacement 多对象交互（如篮球进网 / 炮弹爆炸 / 多角色赛跑）。
- 缺乏 human evaluation 是一个明显短板——sketch animation 的"vivid / plausible"高度主观，VBench 自动指标只能部分代表观感。

---

## 6. Strengths

1. **Method-level 真正吃到了 LLM 的运动先验**——把 GPT-4 的 reasoning + bbox 轨迹规划与 SDS 优化结合（§3.2.2 + Eq. 5），这一组合是从 Comp4D / VideoDirectorGPT 思想成功迁移到 vector sketch 任务的范例；ablation Setup #1 显示 ΔZ_c 是 dynamic motion 的核心来源。
2. **Compositional SDS 的工程化处理**优雅有效——把"复杂场景"拆解后让冻结的 ModelScope T2V 只为它擅长的子场景打分（§3.2.4 + Eq. 8），不引入额外训练；ablation Setup #5（w/o CSDS）证明该项是细节质量的关键来源。
3. **保持 zero-training-data 范式**——整个系统唯一可训练的就是 Refinement Net 的少量参数，每张 sketch 1 小时优化即可，使方法对学术资源 / 商业 GIF 工具都很友好；这与 Live-Sketch 的成本量级一致，但能力扩展到了多对象。
4. **Robustness to local errors**——Fig. 6 & §4.2 表明 Grounding DINO / point assignment 中等程度的错误不会摧毁结果，因为 Compositional SDS 提供多重监督互相纠正。

## 7. Weaknesses & Limitations

1. **测试集规模太小（60 张）且无 human evaluation**——VBench 4 项的提升幅度大多在 ±0.02 量级，置信度不足；建议至少做几十人的 A/B 评测来佐证视觉质量优势。
2. **方法对 LLM 的能力强依赖**——若 GPT-4 输出物理错误（Fig 11 案例 2），下游模块无法自纠；论文未做 LLM 替换实验（如 GPT-3.5 / Llama-3 / Qwen），不知方法对 LLM 强度的下界在哪。
3. **Compositional SDS 各项等权累加**（Eq. 8）—— L_SDS 与 r 个 L_{SDS-k} 直接相加，r 越大整体梯度幅度越大，可能让多对象样本过早饱和；缺少 weight tuning / annealing 实验。
4. **不公开 prompt 模板**（Scene Decomposition 与 Motion Planning 两步对 GPT-4 的 prompt 未在论文中给出），复现时必须重写 prompt，难复现 paper 的具体数字。
5. **优化时间昂贵**——1 小时 / 样本，无法支持互动 GIF 创作；与 FlipSketch（fine-tuned T2V，1 次 forward 即出动画）相比劣势明显。Discussion 应增加 efficiency 比较。
6. **代码尚未释放**（"The code will be released."），权重不会有（per-sample 优化），数据集仅在补充材料附图。

## 8. Comparison with Concurrent / Related Work

| Work | 任务定义 | Sketch 表征 | 是否多对象 | 是否需 fine-tune | LLM 用途 | Compositional SDS | 链接 |
|---|---|---|---|---|---|---|---|
| Live-Sketch (CVPR 2024) | 单对象 sketch animation | Vector | 否 | 否 | 无 | 否 | [2311.13608](https://arxiv.org/abs/2311.13608) |
| FlipSketch (CVPR 2025) | 单对象 sketch animation | Raster | 否 | 是 | 无 | 否 | [2412.16209](https://arxiv.org/abs/2412.16209) |
| **MoSketch (本文)** | **多对象 sketch animation** | **Vector** | **是** | **否** | **Scene Decomp + Motion Plan** | **是** | [2503.19351](https://arxiv.org/abs/2503.19351) |
| Comp4D (2024) | 多对象 4D 场景生成 | 3D Gaussian | 是 | 否（per-scene 优化） | Layout & Trajectory plan | 是 | [2403.16993](https://arxiv.org/abs/2403.16993) |
| LLM-Grounded Video Diffusion (ICLR 2024) | T2V，多对象布局 | 像素 | 是 | 否 | Layout planning + reasoning | — | [2309.17444](https://arxiv.org/abs/2309.17444) |
| Trans4D (2024) | 多对象 4D 转场生成 | 3D Gaussian | 是 | 是 | Stage planning | 是 | [2410.07155](https://arxiv.org/abs/2410.07155) |

MoSketch 在表中是**第一个把"LLM-as-planner + Compositional SDS"组合搬到 multi-object vector sketch animation 的工作**，与 Comp4D / Trans4D 在 4D / 3D 场景里同思想路径。

---

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code | ❌（截至 v1） | 论文承诺将开源（"The code will be released."），但 v1 未给链接 |
| Weights | N/A | 没有预训练权重，每次 per-sample 优化 |
| Training data | ❌ | 没有训练集 |
| Test data | ⚠️ 部分 | 60 张 sketch 只在补充材料贴图（Fig 12-17 有 6 张样例），未公开完整 .svg / .png 数据包 |
| Hyperparameters | ⚠️ 部分 | 给了 d=128, L=2, f=16, Adam lr=5e-3, wd=1e-2, 500 steps；缺 batch size、SDS timestep schedule、CFG scale |
| LLM prompt | ❌ | Scene Decomposition / Motion Planning 的 GPT-4 prompt **未公开** |
| Eval / judge prompts | ❌ | 沿用 VBench 默认，但未明确指定具体 metric 调用配置（如 ViCLIP 版本） |
| Hardware | ✅ | 单 RTX 3090 Ti，1 h / 样本 |

**Verdict**：MoSketch 提出的范式概念清晰、模块独立、伪代码可写，但**实际复现门槛偏高**——主要卡在 (i) GPT-4 prompt 模板缺失，(ii) 60 张测试 sketch 的官方版本未释放，(iii) SDS 内部细节（timestep schedule、CFG scale）继承自 Live-Sketch 但没明确指定，(iv) 代码尚未公布。读者若拿一份自己的多对象 sketch 测试，能复刻"分解 + 规划 + 优化"管线，但要复现 paper 报告的精确数字几乎不可能。

---

> 备注：本解读基于论文 v1（arXiv:2503.19351v1, 2025-03-25）。如后续作者发布 v2 或补充材料、释放代码与 prompt，应回到本档案补充。
