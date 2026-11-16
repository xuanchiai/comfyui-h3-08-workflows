"""MiniMax H3 workflow memory and AV storage tools.

Adapted from kijai/ComfyUI-KJNodes nodes/minimax_nodes.py, commit
ab72eda7ffad6280422d71ab7d54d3fb87c0ba18, GPL-3.0. Node ID is
namespaced to avoid collision with an independently updated KJNodes install.
"""
import logging
import types

import torch

import comfy.ops
from comfy_api.latest import ComfyExtension, io
from .direct_latent import H3ResearchDirectLatentUpscale, H3ResearchArchiveAVLatent


def _chunked_forward(self, x):
    # SwiGLU projections process each token row independently.
    if x.shape[0] > self.h3_research_seq_threshold and self.h3_research_num_chunks > 1:
        out = torch.empty_like(x)
        offset = 0
        for chunk in torch.chunk(x, self.h3_research_num_chunks, dim=0):
            out[offset:offset + chunk.shape[0]] = comfy.ops.linear_input_act(
                self.fc2, self.fc1(chunk), "swiglu"
            )
            offset += chunk.shape[0]
        return out
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")


class _ChunkPatch:
    def __init__(self, num_chunks, seq_threshold):
        self.num_chunks = num_chunks
        self.seq_threshold = seq_threshold

    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.h3_research_num_chunks = self.num_chunks
            self_module.h3_research_seq_threshold = self.seq_threshold
            return _chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)


class H3ResearchChunkFeedForward(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ResearchChunkFeedForward",
            display_name="H3 前馈分块",
            category="H3/显存优化",
            description="按视频token行分块计算前馈层。1块关闭分块，增加块数可以降低中间张量峰值，实际耗时和显存随任务变化。",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("chunks", default=4, min=1, max=64, step=1),
                io.Int.Input("seq_threshold", default=4096, min=256, max=262144, step=256),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, chunks, seq_threshold) -> io.NodeOutput:
        if chunks == 1:
            return io.NodeOutput(model)
        patched_model = model.clone()
        diffusion_model = patched_model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks or not hasattr(blocks[0], "mlp") or not hasattr(blocks[0].mlp, "fc1"):
            raise ValueError("H3 Research Chunk Feed Forward requires the supported MiniMax H3 model")
        for idx, block in enumerate(blocks):
            patched = _ChunkPatch(chunks, seq_threshold).__get__(block.mlp, block.mlp.__class__)
            patched_model.add_object_patch(f"diffusion_model.blocks.{idx}.mlp.forward", patched)
        logging.info("H3ResearchChunkFeedForward: patched %d H3 feed-forward blocks, chunks=%d threshold=%d", len(blocks), chunks, seq_threshold)
        return io.NodeOutput(patched_model)


class H3WorkflowToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3ResearchChunkFeedForward, H3ResearchDirectLatentUpscale, H3ResearchArchiveAVLatent]


async def comfy_entrypoint():
    return H3WorkflowToolsExtension()
