# Credits and licenses · 致谢与许可

The included workflow files, installer and supporting code are distributed under the [GPL-3.0 license](LICENSE). Third-party model weights and runtime packages follow their own upstream terms. The installer obtains the pinned releases directly from the sources in [manifest.json](manifest.json).

本仓库随附的工作流、安装脚本和配套代码采用 [GPL-3.0](LICENSE)。第三方权重与运行环境沿用其上游许可。安装器从 [清单](manifest.json) 列出的原始仓库取得指定版本。

| Project or author · 项目或作者 | Contribution · 用途 |
| --- | --- |
| [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3) | MiniMax H3 FL2VA model family · 基础音视频模型 |
| [Comfy Org](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.45.0) and [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) | Local graph runtime, H3 nodes and optimized CUDA execution · 本地运行与算子 |
| [Comfy Org's H3 weight repository](https://huggingface.co/Comfy-Org/MiniMax-H3) | INT8 diffusion and video VAE, Qwen vision-text encoder, FP32 audio VAE · 扩散权重、图文编码器及音视频 VAE |
| [Zironic's H3-Optimizations](https://github.com/Zironic/H3-Optimizations/commit/f195ca494c876007ad7fa04561c30af49ed2b0c2) | Memory optimization and sparse attention · 显存调度与视频注意力 |
| [Alibaba TaoLive AIGC Team](https://huggingface.co/TaoLiveAIGC/TaoMate-H3-FL2VA), [Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental), [austinit](https://huggingface.co/austinit/TaoMate-H3-FL2VA-ComfyUI) | TaoMate three-step distillation, adapter conversion and ComfyUI recipe · 三步蒸馏、格式转换和采样配方 |
| [Lovis Odin at fal](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA) | Realism People adapter · 人物质感 LoRA |
| [sepiablue-ai](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream/tree/eb148a3d0f91a273668bbe027bfcbafb85f2c5e8) | Upstream streaming delivery adapter used as the basis of the adapted native-size delivery node · 原生尺寸流式交付节点的上游来源 |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes/blob/ab72eda7ffad6280422d71ab7d54d3fb87c0ba18/nodes/minimax_nodes.py) | Source of the feed-forward chunking adaptation in the workflow helper package · 工具节点的前馈分块实现来源 |
| [PyAV](https://github.com/PyAV-Org/PyAV) and [FFmpeg](https://ffmpeg.org/) | MP4, H.264 and AAC output · 视频编码与封装 |

The source packages preserve their included LICENSE and NOTICE files. The original AI-generated reference stills and prompts are included for reproducibility. For the H3 model and its generated media, review the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and [NOTICE](NOTICE).

节点包保留对应 LICENSE 与 NOTICE。示例首图为 AI 生成图，用于复现对应场景。H3 权重和生成视频的条款见 [MiniMax H3 社区许可](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) 与 [NOTICE](NOTICE)。

