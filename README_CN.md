[Read in English](./README.md)

<div align="center">

<img src="./assets/movieroll-logo.png" alt="MovieRoll: And... Action!" width="800">

### A Systematic Framework for Rolling Out Long-Form Videos

<p align="center">
  <a href="https://scholar.google.com/citations?user=bbgjlg0AAAAJ&amp;hl=zh-CN&amp;oi=ao"><b>Yunyang&nbsp;Ge</b></a><sup>1,2,&ast;,‡</sup>, <a href="https://scholar.google.com/citations?user=gsOklKAAAAAJ&amp;hl=zh-CN&amp;oi=ao"><b>Weijie&nbsp;Kong</b></a><sup>2,&ast;</sup>, <a href="https://scholar.google.com/citations?user=MDrO_twAAAAJ&amp;hl=zh-CN&amp;oi=ao"><b>Ryan&nbsp;Xu</b></a><sup>2,&ast;</sup>, <a href="https://scholar.google.com/citations?view_op=list_works&amp;hl=en&amp;user=1xTR6qoAAAAJ"><b>Yue&nbsp;Wu</b></a><sup>2</sup>, <a href="https://scholar.google.com/citations?user=Ypu45nIAAAAJ&amp;hl=zh-CN"><b>Qi&nbsp;Tian</b></a><sup>2</sup>, <a href="https://scholar.google.com/citations?user=lHbXg_0AAAAJ"><b>Jiangfeng&nbsp;Xiong</b></a><sup>2</sup><br>
  <a href="https://scholar.google.com/citations?user=GCOVDKoAAAAJ&amp;hl=zh-CN"><b>Bin&nbsp;Lin</b></a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?user=U6bikksAAAAJ&amp;hl=zh-CN&amp;oi=ao"><b>Mingzhe&nbsp;Zheng</b></a><sup>2,3</sup>, <a href="https://scholar.google.com/citations?user=kcgdO0sAAAAJ&amp;hl=zh-CN&amp;oi=ao"><b>Shenghai&nbsp;Yuan</b></a><sup>1</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&amp;user=vtO33vQAAAAJ"><b>Zheyuan&nbsp;Liu</b></a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&amp;authuser=1&amp;user=nF_klRIAAAAJ"><b>Jianwei&nbsp;Zhang</b></a><sup>2,§</sup>, <a href="https://scholar.google.com/citations?user=-5juAR0AAAAJ&amp;hl=en"><b>Li&nbsp;Yuan</b></a><sup>1,†</sup><br>
  <sup>1</sup> Peking University &nbsp;·&nbsp; <sup>2</sup> Tencent Hunyuan &nbsp;·&nbsp; <sup>3</sup> The Hong Kong University of Science and Technology<br>
  <sub><sup>&ast;</sup> Equal contribution &nbsp;·&nbsp; <sup>§</sup> Project leader &nbsp;·&nbsp; <sup>†</sup> Corresponding author<br>
  <sup>‡</sup> Work done during internship at Tencent Hunyuan</sub>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2611.21339-b31b1b?logo=arxiv)](https://arxiv.org/abs/2611.21339)
[![Project Page](https://img.shields.io/badge/Page-Project_Page-2ea44f)](https://tencent-hunyuan.github.io/MovieRoll/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97_Models-Hugging_Face-yellow)](https://huggingface.co/yunyangge/MovieRoll)
[![SGLang](https://img.shields.io/badge/Backend-SGLang_Diffusion-orange)](./sglang_movieroll/README_CN.md)
[![vLLM-Omni](https://img.shields.io/badge/Backend-vLLM--Omni-blue)](./vllm_omni_movieroll/README_CN.md)
[![Reactor Runtime](https://img.shields.io/badge/Backend-Reactor_Runtime-7B61FF)](./reactor/README.md)
[![License](https://img.shields.io/badge/License-Tencent_Hunyuan_Community-green)](./LICENSE)

</div>

## 🎬 视频演示

https://github.com/user-attachments/assets/2d4287cd-ae78-4870-8cd9-fec0edb21ecf

## 🔥 最新动态！

- **[2026.11.18]** 🚀 支持 [Reactor Runtime](./reactor/README.md)，提供持续的文生视频、图生视频流式推理和动态 prompt 更新。
- **[2026.11.16]** 🚀 Day-0 支持 [SGLang Diffusion](./sglang_movieroll/README_CN.md) 和 [vLLM-Omni](./vllm_omni_movieroll/README_CN.md)。
- **[2026.11.16]** 🎉 MovieRoll 代码正式开源！

<a id="todo"></a>

## ✅ TODO List

- [x] 发布训练与推理代码
- [x] 发布模型权重
- [x] 支持 SGLang 和 vLLM-Omni
- [ ] 支持消费级 GPU（如 RTX 4090、RTX 5090）
- [ ] 支持 Gradio
- [ ] 优化推理效率
- [ ] 支持在线播放模式：边生成、边解码并输出视频

<a id="overview"></a>

## 🔎 方法概览

MovieRoll 是一个从 Base 到少步数 Distilled 模型的长视频生成框架，通过训练与推理的系统化设计，缓解持续生成过程中的质量退化。

- **从 Base 到 Distilled**：使用窗口级 Teacher Forcing 训练 MovieRoll-Base，再通过 ODE 初始化和累积长序列调优（CuLT-DMD）得到 MovieRoll-Distilled。
- **长序列持续生成**：结合窗口联合去噪、分层 KV 缓存和缓存重计算；Base 模型还通过 AnnealCFG 在接近去噪终点时降低引导强度。
- **单事件与多事件控制**：支持单条 prompt 或按顺序切换的事件 prompt，并提供文生视频与图生视频权重。

<p align="center">
  <img src="./assets/movieroll-overview.webp" alt="MovieRoll 方法概览：窗口级训练、分层 KV 缓存、累积长序列调优与滚动生成" width="100%">
</p>

<a id="models"></a>

## 🎁 模型选择

| 模型 | 支持任务 | 分辨率（高 × 宽） | 去噪步数 | 引导方式 | 下载 |
|---|---|---|---|---|---|
| MovieRoll-Base | T2V（文生视频）、I2V（图生视频） | 480p（任意宽高比） | 50 | AnnealCFG | [base](https://huggingface.co/yunyangge/MovieRoll/tree/main/base) |
| MovieRoll-Distilled（T2V） | T2V（文生视频） | 480 × 864 / 864 × 480 | 5 | 不使用 CFG | [distilled/t2v](https://huggingface.co/yunyangge/MovieRoll/tree/main/distilled/t2v) |
| MovieRoll-Distilled（I2V） | I2V（图生视频） | 480 × 864 / 864 × 480 | 5 | 不使用 CFG | [distilled/i2v](https://huggingface.co/yunyangge/MovieRoll/tree/main/distilled/i2v) |

<a id="setup"></a>

## 🛠️ 环境准备

### 运行环境与 GPU 配置

以下命令使用 Python 3.12、支持 CUDA 的 PyTorch 和已安装的 FlashAttention 内核。FlashAttention-3 从其 `hopper/` 目录编译，需要 Hopper（sm90）架构的 GPU（如 H100）；开始长视频生成前，请确认内核编译配置支持当前 GPU。安装完成后，以下命令均在仓库根目录执行。

| 启动配置 | 用途 | 参数 |
|---|---|---|
| 8 张 GPU | 可选的多卡推理 | `torchrun --nproc_per_node=8`；`--sp-size 8`、`--shard-world 8` |
| 1 张 GPU | 默认 Quick Start | `python`；`--sp-size 1`；整条流水线需要能放入单卡显存（96 GB 显存的 GPU 可以放下） |

使用其他 GPU 数量时，`--sp-size` 必须同时整除总卡数和模型的 16 个注意力头；启用 FSDP 时，`--shard-world` 也必须整除总卡数，并且是 `--sp-size` 的倍数。单进程推理不使用 FSDP。示例采用 BF16 transformer 计算。

**当前原生推理入口暂不支持 CPU offload。** 后续将增加支持，以便在消费级 GPU 上进行推理。

```bash
git clone https://github.com/Tencent-Hunyuan/MovieRoll.git && cd MovieRoll
conda create -n movieroll python=3.12 -y && conda activate movieroll
pip install -r requirements.txt
pip install flash-attn --no-build-isolation        # FlashAttention-2，必装

# FlashAttention-3，同样必装（需要 Hopper / sm90 GPU）
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && pip install . --no-build-isolation && cd ../..
python -c "import flash_attn, flash_attn_interface"   # 没有任何输出，说明两者都已装好
```

<details>
<summary>⚡ 可选：SageAttention</summary>

SageAttention 能让采样快约 1.4 倍。请安装下面这个 commit（`v2.2.0` 版本有 kernel bug），然后在推理命令里加上
`--attention-backend sage`。生成的视频和默认的 FlashAttention-3 非常接近，但不是逐位一致。

```bash
git clone https://github.com/thu-ml/SageAttention.git && cd SageAttention
git checkout d1a57a5 && pip install . --no-build-isolation && cd ..
```

</details>

### 🧱 下载权重

MovieRoll 权重位于 [yunyangge/MovieRoll](https://huggingface.co/yunyangge/MovieRoll)。请先同意仓库的访问条款，并通过 `hf auth login` 登录后下载 gated 权重。

**1. 下载一个 MovieRoll 模型。** 将 `MODEL` 设为 `base`、`distilled/t2v` 或 `distilled/i2v`：

```bash
MODEL=distilled/t2v
hf download yunyangge/MovieRoll --include "$MODEL/*" --local-dir ./ckpts/movieroll
```

**2. 下载共用的 VAE 和文本编码器。** 三个模型的推理都需要这些组件：

```bash
hf download tencent/HunyuanVideo-1.5 --include 'vae/*' --local-dir ./ckpts
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2
```

```text
ckpts/
├── vae/
├── text_encoder/
│   ├── llm/
│   ├── byt5-small/
│   └── Glyph-SDXL-v2/
└── movieroll/
    ├── base/                    # 按需下载
    └── distilled/
        ├── t2v/                 # 按需下载
        └── i2v/                 # 按需下载
```

所选模型目录应包含 `config.json` 和 `model.safetensors`。配置文件提供该模型的默认采样参数；仓库内的参考配置见 [`movieroll/configs/models/`](./movieroll/configs/models)。训练所需的初始化权重见 [训练文档](./docs/TRAINING_CN.md)。

<a id="for-agents"></a>

## 🤖 For Agents

你可以直接让能够访问本仓库的编程 agent 读取 [`agent/`](./agent) 下对应任务的文件夹：先阅读 `SKILL_CN.md`（或英文版 `SKILL.md`），再按需阅读其中引用的文件，按照指引执行所需任务。

| 需要做的事情 | 任务文件夹 |
|---|---|
| 改写文生视频、图生视频或多事件 prompt | [streaming-video-prompt-rewrite](./agent/streaming-video-prompt-rewrite) |
| 运行推理或各阶段训练 | [movieroll-run](./agent/movieroll-run) |
| 将 MovieRoll 的方法迁移到其他代码库 | [movieroll-port](./agent/movieroll-port) |
| 使用 SGLang Diffusion 部署和推理 | [movieroll-sglang](./agent/movieroll-sglang) |
| 使用 vLLM-Omni 部署和推理 | [movieroll-vllm-omni](./agent/movieroll-vllm-omni) |

例如，可以直接对 agent 说：

> 请阅读 `agent/movieroll-run/SKILL_CN.md` 及其引用的相关文件，然后使用 Quick Start 中的 prompt，帮我在单张 GPU 上运行 MovieRoll-Distilled 文生视频推理。

更多说明见 [`agent/README.md`](./agent/README.md)。

<a id="quick-start"></a>

## 🚀 推理快速开始

下载模型权重后，在仓库根目录运行以下命令即可生成一条视频。

```bash
python generate_movieroll.py \
  --model ckpts/movieroll/distilled/t2v --sp-size 1 \
  --prompt "In a softly lit, modern living room, a surreal event unfolds as a single orange and white clownfish swims gracefully through the air, continuously circling the head of a young man. The man, in his early twenties with dark, curly hair, sits calmly at a wooden desk, wearing a comfortable cream-colored knit sweater. His expression is one of quiet awe, his eyes subtly tracking the fish's rhythmic, looping path around him. The clownfish moves as if through water, its fins gently propelling it forward in a steady, circular motion. Its scales glisten with a wet sheen under the warm, diffused light coming from a large window to the left, and a few stray, glistening water droplets seem to trail in its wake, hanging in the air for a moment before vanishing. In the background, intentionally out of focus, sits a rectangular glass fish tank filled with clear blue water and swaying green aquatic plants, the source of this airborne marvel. The camera remains static in a medium shot, maintaining an eye-level perspective with the seated man. This fixed viewpoint captures both the man's serene reaction and the fish's complete, unending orbit around his head, emphasizing the dreamlike and continuous nature of the scene. Cinematic photography realistic style." \
  --seed 4890 --video-length 721 \
  --output-dir ./outputs
```

输出文件：`./outputs/samples/default/rank0_sample_0.mp4`。

<a id="inference-options"></a>

## ⚙️ 推理参数

以下保留常用推理参数。`--model`（或 `--dit-weight`）必须指定；`--context-noise` 等模型相关默认值来自所选模型配置。

```bash
--model ckpts/movieroll/base       # 必填（或用 --dit-weight 传入单个 .safetensors），模型目录：base（50 步 + AnnealCFG），distilled/t2v 或 distilled/i2v（5 步）
--prompt "..."                     # 整段视频用一条 prompt
--prompt-list "..." "..." "..."    # 多条 prompt，在同一段视频里依次播放（多事件）
--prompt-csv prompts.csv           # 批量生成：列为 index, seed, prompt（或 event_1..event_N），可选 input_image
--input-image first_frame.jpg      # 图生视频：这张图缩放并居中裁剪到视频尺寸后作为第一帧（配合 distilled/i2v 效果最好）
--video-length 721                 # 帧数，需满足 16k+1：241 = 10 秒，721 = 30 秒，1441 = 60 秒
--height 480 --width 864           # 分辨率，16 的倍数
--seed 930                         # --prompt / --prompt-list 的噪声种子（默认取 --global-seed 930）；CSV 每行使用自己的 seed 列
--context-noise 0.15               # 给缓存的历史帧加的噪声，默认取自模型的 config：base 0.05，distilled/t2v 0.15，distilled/i2v 0
--attention-backend fa3            # fa3，或 sage（SageAttention，快约 1.4 倍）
--vae-parallel-mode none           # 设为 sp 时，VAE 解码（PS-VAE）会分摊到每个序列并行组的各张卡上
--sp-size 8                        # 序列并行组大小；八卡推理设为 8，单卡设为 1
--shard-world 8                    # FSDP 分片组大小；必须整除卡数（不足 8 张卡时设成卡数）
--num-videos-per-prompt 1          # 每条 prompt 生成几个视频，每个视频的噪声都不同
--output-type video                # video（mp4）、pt（解码后的张量）或 latent（仅 distilled）
--output-dir ./outputs             # 视频的输出目录
```

base 与 distilled 模型的单事件、多事件 `--prompt-csv` 示例文件见 [`example/`](./example)。

<a id="prompt-rewriting"></a>

### ✍️ Prompt 改写

MovieRoll 更适合详细的长 prompt：镜头保持静止或跟随主体，动作可以一直延续下去，多个事件之间的场景保持一致。
[`agent/`](./agent) 下的 prompt 改写 skill 可以把一句简短的想法或一串事件改写成这样的 prompt：

- [`SKILL_CN.md`](./agent/streaming-video-prompt-rewrite/SKILL_CN.md)：中文 prompt
- [`SKILL.md`](./agent/streaming-video-prompt-rewrite/SKILL.md)：英文 prompt

两份都在 [`agent/streaming-video-prompt-rewrite`](./agent/streaming-video-prompt-rewrite) 目录下，编程 agent 加载哪一份都可以
（在 Claude Code 中，把整个文件夹复制到 `.claude/skills/` 下即可）。配套的 `system_prompt_CN.md` / `system_prompt.md`
也可以作为任意大模型的 system prompt 使用（取基础部分加上某一种模式的部分，或用 `scripts/validate_prompt.py` 里的
`build_system_prompt(mode)` 拼好），同一个脚本两种语言都能校验。

<a id="backends"></a>

## ⚡ 推理后端

以上示例使用 MovieRoll 原生推理入口。集成推理框架时，请按照相应后端文档中的环境准备、模型组装和推理命令操作。

| 后端 | 文档 |
|---|---|
| SGLang Diffusion | [部署与推理](./sglang_movieroll/README_CN.md) |
| vLLM-Omni | [部署与推理](./vllm_omni_movieroll/README_CN.md) |
| Reactor Runtime | [部署与推理](./reactor/README.md) |

<a id="training"></a>

## 🎓 训练

MovieRoll 的训练分为四个阶段：

```
HunyuanVideo-1.5 480p T2V
  └─① 流式 SFT，单事件 ──▶ MovieRoll-Base
       └─② 流式 SFT，多事件 ──▶ 多事件 Base
            └─③ ODE 初始化 ──▶ 少步 student
                 ├─④ CuLT-DMD ──▶ MovieRoll-Distilled（文生视频）
                 └─④ CuLT-DMD + ACS ──▶ MovieRoll-Distilled（图生视频）
```

完整的训练数据准备、各阶段配置和启动命令见 [训练文档](./docs/TRAINING_CN.md)。

## 🙏 致谢

- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- [LongLive](https://github.com/NVlabs/LongLive)
- [Self Forcing](https://github.com/guandeh17/Self-Forcing)
- [Causal Forcing](https://github.com/thu-ml/Causal-Forcing)
- [Helios](https://github.com/PKU-YuanGroup/Helios)
- [Rolling Forcing](https://github.com/TencentARC/RollingForcing)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Glyph-ByT5](https://github.com/AIGText/Glyph-ByT5)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tencent-Hunyuan/MovieRoll&type=Date)](https://star-history.com/#Tencent-Hunyuan/MovieRoll&Date)

<a id="license"></a>

## 📄 License

MovieRoll 的代码从 [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) 的 `60783e7` commit 分叉而来，
并以相同的 [TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT](./LICENSE) 发布。

<a id="citation"></a>

## 📚 Citation

如果我们的工作对你有帮助，欢迎引用：

```bibtex
@misc{ge2026movierollsystematicframeworkrolling,
      title={MovieRoll: A Systematic Framework for Rolling Out Long-Form Videos},
      author={Yunyang Ge and Weijie Kong and Ryan Xu and Yue Wu and Qi Tian and Jiangfeng Xiong and Bin Lin and Mingzhe Zheng and Shenghai Yuan and Zheyuan Liu and Jianwei Zhang and Li Yuan},
      year={2026},
      eprint={2611.21339},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.21339},
}
```

<a id="contact"></a>

## 📬 Contact

如有问题或合作意向，请联系 [yunyang@stu.pku.edu.cn](mailto:yunyang@stu.pku.edu.cn)。
