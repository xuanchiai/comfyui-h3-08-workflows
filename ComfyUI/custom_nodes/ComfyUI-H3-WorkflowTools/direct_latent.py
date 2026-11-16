"""H3 AV archive and direct latent size adaptation."""
import logging
from pathlib import Path
import sys
import time

import torch
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
import folder_paths
import safetensors.torch

MODEL = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
LOG = logging.getLogger(__name__)


def split_av(latent):
    """Accept sampled AV or an explicitly loaded H3 motion-context archive."""
    if latent.get("noise_mask") is not None:
        raise ValueError("Direct H3 research upscale does not support masked latents")
    source = latent.get("samples")
    parts = source.tensors if getattr(source, "is_nested", False) else source
    if not isinstance(parts, (list, tuple)) or len(parts) != 2:
        raise ValueError("Expected an H3 video/audio latent pair")
    video, audio = parts
    if not (isinstance(video, torch.Tensor) and video.ndim == 5
            and video.shape[0] == 1 and video.shape[1] == 24):
        raise ValueError("Expected H3 video [1,24,T,H,W]")
    if not (isinstance(audio, torch.Tensor) and audio.ndim == 4
            and audio.shape[0] == 1 and audio.shape[1:3] == (32, 2)):
        raise ValueError("Expected H3 audio [1,32,2,T]")
    return video, audio


def installed_upscaler():
    """Reuse the already loaded dependency, including its trained normalization."""
    suffix = "/ComfyUI-H3-Video-Upsampler/tiled_upsampler.py"
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and str(filename).replace("\\", "/").endswith(suffix):
            return module
    raise RuntimeError("ComfyUI-H3-Video-Upsampler must be installed and loaded")


def upscale_av(samples, upscale_model, width, height):
    video, audio = split_av(samples)
    width, height = int(width), int(height)
    if width % 32 or height % 32 or min(width, height) < 32:
        raise ValueError("H3 target dimensions must be positive multiples of32")
    src_h, src_w = video.shape[-2] * 16, video.shape[-1] * 16
    if width < src_w or height < src_h:
        raise ValueError("This node upscales; the target must not shrink either axis")
    started = time.perf_counter()
    if (width, height) == (src_w, src_h):
        enlarged, mode = video, "passthrough"
    else:
        enlarged = installed_upscaler().upscale_chunks_batch(
            [video], upscale_model, width, height
        )[0]
        mode = "learned3d_whole_clip"
    expected = (*video.shape[:-2], height // 16, width // 16)
    if tuple(enlarged.shape) != expected or not torch.isfinite(enlarged).all():
        raise RuntimeError("H3 learned upscaler returned an invalid video latent")
    result = dict(samples)
    result["samples"] = NestedTensor((enlarged, audio))
    # The upscaler receives only video: audio retains the same tensor object.
    LOG.info("H3DirectLatent: %s %sx%s -> %sx%s, %s latent steps, %.3fs; audio unchanged",
             mode, src_w, src_h, width, height, video.shape[2], time.perf_counter() - started)
    return result


class H3ResearchDirectLatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ResearchDirectLatentUpscale",
            display_name="H3 潜空间尺寸适配",
            category="H3/潜空间",
            description="Sampled H3 AV directly to the installed learned3D upscaler. No image decode/encode, no diffusion redraw. Audio untouched. Equal dimensions explicitly bypass the network for a reference decode.",
            inputs=[io.Latent.Input("samples"),
                    io.Combo.Input("upscale_model", options=[MODEL], default=MODEL),
                    io.Int.Input("width", default=1888, min=32, max=4096, step=32),
                    io.Int.Input("height", default=1088, min=32, max=4096, step=32)],
            outputs=[io.Latent.Output(display_name="AV latent")],
        )

    @classmethod
    def execute(cls, samples, upscale_model, width, height):
        return io.NodeOutput(upscale_av(samples, upscale_model, width, height))


class H3ResearchArchiveAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ResearchArchiveAVLatent", display_name="H3 保存AV存档",
            category="H3/存档", is_output_node=True,
            description="Save video/audio separately as safetensors and pass the original AV latent onward. This archive can be loaded by H3 Motion Context Load then explicitly converted by the Direct3D node.",
            inputs=[io.Latent.Input("samples"),
                    io.String.Input("filename_prefix", default="H3/av")],
            outputs=[io.Latent.Output(display_name="unchanged AV latent"), io.String.Output(display_name="archive path")],
        )

    @classmethod
    def execute(cls, samples, filename_prefix):
        video, audio = split_av(samples)
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        path = Path(folder) / f"{filename}_{counter:05d}_.safetensors"
        safetensors.torch.save_file(
            {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().contiguous()},
            str(path), metadata={"format": "h3_motion_context_av_v1"})
        LOG.info("H3DirectLatent: archived sampled AV latent to %s", path)
        return io.NodeOutput(samples, str(path))
