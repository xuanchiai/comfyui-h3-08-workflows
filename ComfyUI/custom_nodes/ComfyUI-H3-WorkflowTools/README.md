# H3 Workflow Tools

正式 H3 工作流使用的前馈分块、AV 存档和潜空间尺寸适配节点。前馈分块改编自 `MiniMaxChunkFeedForward`，来源为 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes/blob/ab72eda7ffad6280422d71ab7d54d3fb87c0ba18/nodes/minimax_nodes.py)，GPL-3.0，见 LICENSE。

界面分类为 H3/显存优化、H3/存档和 H3/潜空间。节点内部 ID 沿用已保存画布的名称，避免旧画布缺节点。前馈分块仅对连接的模型副本生效，保留现有计算逻辑。

AV 存档写入 ComfyUI/output/H3。同尺寸适配直接传递视频和音频潜空间，改为更大尺寸时调用已安装的 ComfyUI-H3-Video-Upsampler。正常工作流使用同尺寸路径。

使用说明位于 ComfyUI/user/default/参考资料/H3/使用说明.md。日常通过根目录 run_nvidia_gpu.bat 启动正式 8188。
