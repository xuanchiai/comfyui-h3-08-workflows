# ComfyUI H3-08 · Native 1080p dialogue

**English** · [简体中文](README.zh-CN.md)

Two ten-second MiniMax H3 image-to-video presets with native audio. Both sample on the 1920 × 1088 model grid and produce a **1920 × 1080, 24 fps, 240-frame MP4** and an AV latent archive.

| Workflow | Video attention by step | Default example | Benchmarked scenes |
| --- | --- | --- | --- |
| [H3-08](ComfyUI/user/default/workflows/05-图生视频/H3-人像图生视频/H3-08.json) | 29% / 7% / 7% | Print-shop dialogue | Print shop and basketball court (small movements and wide dolly) |
| [H3-08B](ComfyUI/user/default/workflows/05-图生视频/H3-人像图生视频/H3-08B.json) | 29% / 15% / 7% | 2K basketball court | Basketball court (small movements and wide dolly) |

Both presets accept a different first frame and prompt. The benchmarks below cover the sample scenes listed above.

## Install and run

1. Extract the Windows x64 NVIDIA Portable build of [ComfyUI v0.45.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.45.0). Use **Code → Download ZIP** or clone this repository into its own directory.
2. Close ComfyUI. From the portable installation directory containing `python_embeded`, run this command with the actual path to the downloaded repository. The installer checks hashes, installs the two canvases and node source, fetches the pinned attention plugin, and can fetch the six weights from their original Hugging Face repositories. The weights take about **37.61 GiB**. Check the model license and regional eligibility before choosing `--download-models`. [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

```powershell
.\python_embeded\python.exe -s -X utf8 "D:\comfyui-h3-08-workflows\install_h3.py" --comfyui .\ComfyUI --download-models
```

3. Start the same ComfyUI installation. These are the flags used for the published measurements.

```powershell
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --port 8188 --use-ck-attention --offline --disable-pinned-memory --vram-headroom 2.0 --reserve-vram 2.0 --cache-none --disable-fast-disk
```

4. Open **Workflow → 05-图生视频 → H3-人像图生视频**. Choose H3-08 or H3-08B and click **Run** once. The MP4 and AV archive appear in `ComfyUI/output/H3/H3-08/` or `ComfyUI/output/H3/H3-08B/`.

If you already have the six exact weights, omit `--download-models`. Use `--check` to inspect a local installation. Use `--optimizer-zip` with the pinned upstream ZIP for offline plugin installation. The installer reports a file conflict when an existing file differs from the manifest.

### Reproduce the three AI-generated demonstrations

[Watch the demo on X](https://x.com/Xmaxkoi/status/2122289135780343809)

| Demonstration | Canvas and prompt | Time | Whole-GPU VRAM | System RAM |
| --- | --- | ---: | ---: | ---: |
| 01 · Print-shop dialogue | H3-08 defaults, [prompt](prompts/print-shop.txt) | 192.30 s | 8.32 GiB | 14.51 GiB |
| 02 · Basketball court, wide dolly | H3-08, [shared dolly prompt](prompts/court-dolly.txt) | 196.10 s | 9.04 GiB | 15.96 GiB |
| 03 · Basketball court, wide dolly | H3-08B, [shared dolly prompt](prompts/court-dolly.txt) | 203.21 s | 9.12 GiB | 15.85 GiB |

For 02 and 03, select the [same 2048 × 1152 first image](ComfyUI/input/H3/native1080/drama/riverside-court-native2k.png) in node **#9**, paste the entire dolly prompt into **#11**, and keep the fixed seed **411064**. The woman asks “明天还来吗？” and the man replies “来。球我带。” The camera slides sideways while keeping both actors in frame.

The backboard moved about **862 px** with H3-08 and **878 px** with H3-08B by 4.50 s. Moving hair and fine background lines softened briefly in both clips. Faces and the main background structures stayed continuous in the inspected frames. H3-08B showed a small difference in detail in this scene and took 7.12 s longer.

Measurements come from one Windows x64 PC with an RTX 4080 SUPER (16 GiB) and 31.84 GiB visible physical RAM. Each number is the observed peak including desktop use. The two dolly runs started at about 14.9 GiB system RAM. Reproduction on 8 GiB GPUs and 16 GiB systems remains unverified.

## All workflow settings

| Nodes | Settings |
| --- | --- |
| #9, #10 · First frame | Load the first PNG. Lanczos, centered crop to 1920 × 1088. The two included inputs preserve original image pixels. |
| #8, #11 · Conditioning | INT8 ConvRot video VAE for the first-frame latent. Qwen3-VL 32B NVFP4 AWQ for text and vision, vision width 1280, reserve 12.5 GiB, model length 243 frames, first-frame conditioning at 0 s. |
| #41, #42, #2 · Model load | Select H3 FL2VA pruned INT8 ConvRot. Release the first-frame VAE and reclaim resident RAM before diffusion model load. |
| #4, #25 · LoRAs | Model-only TaoMate FL2VA 3-step average rank 19 BF16 at 1.0, then fal Realism People at 0.6. |
| #5 · AV time mapping | Video shift 12, audio shift 3. |
| #3 · Memory optimization | Activation chunk rows 4096. Fused QKV, MLP optimization, precision, QKV streaming and embedding memory on Auto. V memory uses Lower VRAM (slower). |
| #6 · Sparse attention, shared | Kitchen INT8 64Q × 128KV Faster, video token order 1×8×8, Hold, Early steps 1, Early KV 0.29. |
| #6 · H3-08 | Video attention budget 0.07, Late steps 0. Late KV 0.20 is inactive in this preset. |
| #6 · H3-08B | Video attention budget 0.15, Late steps 1, Late KV 0.07. |
| #1, #12, #13, #14, #15 · Sampling | Fixed seed 411064, positive BasicGuider (CFG 1), Euler, sigmas 1.0 / 0.961165 / 0.853333 / 0.0, reserve 13.25 GiB. Three updates. |
| #16 · Archive | Save audio and video latents as safetensors and forward the original latents to output. |
| #44, #45, #19, #20 · Audio | Release the diffusion model after archiving. Load the FP32 audio VAE, decode and release. |
| #22 · Video output | Shared INT8 video VAE, FP16 compute on the tested runtime, 256 spatial tiles, at least 64 px overlap, 17-frame windows with 5-frame overlap, at most four tiles per batch. Crop 4 px above and below. Keep 240 frames at 24 fps. H.264 libx264 CRF 16, yuv420p and AAC. |
| #40 · Canvas note | Usage and benchmark notes. |

The per-node reserves are scheduler inputs. Benchmark peaks are measured separately. Tested runtime versions are ComfyUI **0.45.0**, Python **3.13.12**, PyTorch **2.12.0+cu130**, comfy-kitchen **0.2.41**, comfy-aimdo **0.6.0**, PyAV **17.0.1** and frontend **1.58.5**. The complete file sizes, SHA-256 hashes, destination paths and exact model revisions are in the [installation manifest](manifest.json).

## Dependencies and credits

Thanks to MiniMax, ComfyUI and Comfy Kitchen, Zironic, the TaoLive AIGC Team, Kijai and austinit, Lovis Odin and fal, and sepiablue-ai. The included source carries its upstream notices. [Attribution and license details](THIRD_PARTY_NOTICES.md)

The repository publishes the workflow JSON, supporting source, example input images, prompts and measurements. The three locally generated MP4s are identified above so eligible users can reproduce them. Distribution of H3-generated media is governed by the regional scope in the H3 community license. [MiniMax H3 Community License, § I and V.4](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

Powered by MiniMax H3.

