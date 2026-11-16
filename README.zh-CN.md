# ComfyUI H3-08 · 原生 1080P 文戏

[English](README.md) · **简体中文**

两套 MiniMax H3 图生视频工作流，用于十秒对白镜头，带原生音频。均在 1920×1088 的模型网格上采样，一次运行输出 **1920×1080、24 fps、240 帧 MP4** 和 AV 潜空间存档。

| 工作流 | 三步视频注意力 | 用途 |
| --- | --- | --- |
| [H3-08](ComfyUI/user/default/workflows/05-图生视频/H3-人像图生视频/H3-08.json) | 29%／7%／7% | 日常对白。默认打开打印店场景。 |
| [H3-08B](ComfyUI/user/default/workflows/05-图生视频/H3-人像图生视频/H3-08B.json) | 29%／15%／7% | 动态镜头备选。默认打开 2K 雨后球场。 |

## 一套环境复现两条工作流

1. 解压 [ComfyUI v0.45.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.45.0) 的 Windows x64 NVIDIA 便携版。在 GitHub 点 **Code → Download ZIP**，解压到单独目录。也可以克隆仓库。
2. 退出 ComfyUI，在含 `python_embeded` 的便携版根目录打开 PowerShell，把下面的示例路径换成仓库实际位置。安装脚本按清单校验文件、安装两套画布和节点源码、下载固定版本的注意力插件，并可从上游下载六份权重。权重合计约 **37.61 GiB**。使用 `--download-models` 前请核对模型许可及适用地域。 [MiniMax H3 社区许可](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

```powershell
.\python_embeded\python.exe -s -X utf8 "D:\comfyui-h3-08-workflows\install_h3.py" --comfyui .\ComfyUI --download-models
```

3. 用同一套 ComfyUI 启动。下列参数对应实测环境。

```powershell
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --port 8188 --use-ck-attention --offline --disable-pinned-memory --vram-headroom 2.0 --reserve-vram 2.0 --cache-none --disable-fast-disk
```

4. 在 **工作流 → 05-图生视频 → H3-人像图生视频** 选择 H3-08 或 H3-08B，点一次「运行」。MP4 和 AV 存档分别写到 `ComfyUI/output/H3/H3-08/` 或 `ComfyUI/output/H3/H3-08B/`。

如果已有六份对应版本的模型，可以省略 `--download-models`。加 `--check` 可只读检查已安装文件。通过 `--optimizer-zip` 可使用从作者处下载的固定提交 ZIP。遇到同名文件内容不一致，安装器会报告对应路径。

### 复现三条 AI 生成样片

| 样片 | 画布和提示词 | 生成耗时 | 整卡显存峰值 | 整机内存峰值 |
| --- | --- | ---: | ---: | ---: |
| 01 · 打印店相亲 | H3-08 默认画布，[提示词](prompts/print-shop.txt) | 192.30 秒 | 8.32 GiB | 14.51 GiB |
| 02 · 雨后球场大幅横移 | H3-08，[共用横移提示词](prompts/court-dolly.txt) | 196.10 秒 | 9.04 GiB | 15.96 GiB |
| 03 · 雨后球场大幅横移 | H3-08B，[共用横移提示词](prompts/court-dolly.txt) | 203.21 秒 | 9.12 GiB | 15.85 GiB |

复现 02 和 03 时，两套都在 **#9** 选同一张 [2048×1152 球场首图](ComfyUI/input/H3/native1080/drama/riverside-court-native2k.png)，在 **#11** 粘贴整段横移提示词，种子固定为 **411064**。女方问「明天还来吗？」，男方答「来。球我带。」。镜头横移，两人保持同框。

到 4.50 秒时，篮板分别向右移动约 **862 和 878 像素**。快移时，两版都有短时间的发缘与背景细线软化，抽查帧中的脸部轮廓和背景主结构衔接连续。H3-08B 在这条镜头里的细节差异较小，本轮多用 7.12 秒。

实测为 Windows x64、RTX 4080 SUPER（16 GiB 显存）、整机可见物理内存 31.84 GiB。表里是含桌面与后台的整卡、整机采样峰值。两次横移起跑整机约 14.9 GiB。8 GiB 显卡、16 GiB 整机的复现能力仍需对应硬件实测。

## 节点参数

| 节点 | 设置 |
| --- | --- |
| #9、#10 · 首图 | 加载 PNG，Lanczos 居中裁切到 1920×1088。随包首图保留原始像素。 |
| #8、#11 · 条件编码 | INT8 ConvRot 视频 VAE 编码首帧，Qwen3-VL 32B NVFP4 AWQ 做图文编码。视觉宽度 1280，调度预留 12.5 GiB，模型时长 243 帧，首帧条件落在 0 秒。 |
| #41、#42、#2 · 模型加载 | H3 FL2VA 剪枝 INT8 ConvRot。首帧条件完成后释放视频 VAE 并回收驻留 RAM，再加载扩散模型。 |
| #4、#25 · LoRA | 标准 Model-only 加载。TaoMate FL2VA 三步平均 Rank 19 BF16 为 1.0，fal Realism People 为 0.6。 |
| #5 · 音画时间映射 | 视频 shift 12，音频 shift 3。 |
| #3 · 内存优化 | Activation chunk rows 4096。Fused QKV、MLP 优化、精度、QKV 流式和嵌入走 Auto，V 路径选 Lower VRAM (slower)。 |
| #6 · 稀疏注意力共用设置 | Kitchen INT8 64Q×128KV Faster，视频 token 排列 1×8×8，Hold，Early steps 1，Early KV 0.29。 |
| #6 · H3-08 | Video attention budget 0.07，Late steps 0。保存的 Late KV 0.20 在该档不参与计算。 |
| #6 · H3-08B | Video attention budget 0.15，Late steps 1，Late KV 0.07，确保末步使用 7%。 |
| #1、#12、#13、#14、#15 · 去噪 | 固定种子 411064，BasicGuider 正向 CFG 1，Euler，sigmas 为 1.0／0.961165／0.853333／0.0。采样预留 13.25 GiB，三次更新。 |
| #16 · AV 存档 | 将音频和视频 latent 保存为 safetensors，原 latent 传给出片节点。 |
| #44、#45、#19、#20 · 声音 | 写完 AV 后释放扩散模型。加载 FP32 音频 VAE，解码后卸载。 |
| #22 · 视频 | 共用 INT8 视频 VAE，实测 FP16 计算。空间块 256，至少重叠 64 像素，时间窗 17 帧、重叠 5 帧，每批最多 4 块。上下各裁 4 像素，输出前 240 帧、24 fps，libx264 CRF 16、yuv420p、AAC。 |
| #40 · 画布说明 | 用法和实测数据。 |

节点里的预留显存属于调度参数。实测运行环境为 ComfyUI **0.45.0**、Python **3.13.12**、PyTorch **2.12.0+cu130**、comfy-kitchen **0.2.41**、comfy-aimdo **0.6.0**、PyAV **17.0.1**、前端 **1.58.5**。六份权重的来源、固定版本、目标路径、字节数和 SHA-256 见 [安装文件清单](manifest.json)。

## 感谢与许可

感谢 MiniMax、ComfyUI、Comfy Kitchen、Zironic、TaoLive AIGC Team、Kijai、austinit、Lovis Odin、fal 和 sepiablue-ai。节点源码随附对应开源许可和来源说明。 [完整致谢与来源](THIRD_PARTY_NOTICES.md)

仓库公开画布、配套源码、复现首图、提示词和实测数据。以上三条为本地生成的 AI 样片，符合使用条件的用户可在本地复现。生成视频的对外分发适用 MiniMax H3 社区许可的地域条款。 [MiniMax H3 社区许可第 I 节与第 V.4 条](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

Powered by MiniMax H3.

