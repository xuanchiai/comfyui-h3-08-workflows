"""Native 1080P H3 nodes for the locally validated workflow."""
import gc
import threading
from contextlib import contextmanager
from pathlib import Path

import folder_paths
import node_helpers
import nodes
from comfy import memory_management, model_management, storage
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize
from comfy_extras.nodes_audio import VAEDecodeAudio

from .conditioning import qwen_host_embedding, retain_mapped_nvfp4_scales, stage_nvfp4_scales
from .native_rgb_stream import save_native_video

CATEGORY = "H3/原生1080P"
LOCK = threading.RLock()


@contextmanager
def reserve_vram(amount):
    if not memory_management.aimdo_enabled:
        yield
        return
    from comfy_aimdo import control

    before = control.get_simple_vram_headroom()
    control.set_simple_vram_headroom(max(before, int(amount * 2**30)))
    try:
        yield
    finally:
        control.set_simple_vram_headroom(before)


def use_local_storage(patcher, path):
    if not patcher.is_dynamic() or storage.fast_storage(path) is not True:
        return
    pin_state = patcher.model.dynamic_pins[patcher.load_device]
    if pin_state["hostbufs_initialized"] and not pin_state["fast_disk"]:
        return
    patcher.fast_disk = True
    pin_state["fast_disk"] = True


class H3NativeUNETLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3NativeUNETLoader", display_name="H3 原生｜扩散模型", category=CATEGORY,
            inputs=[io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models"))],
            outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, unet_name):
        model, = nodes.UNETLoader().load_unet(unet_name, "default")
        use_local_storage(model, folder_paths.get_full_path_or_raise("diffusion_models", unet_name))
        return io.NodeOutput(model)


class H3NativeVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3NativeVAELoader", display_name="H3 原生｜音频或视频 VAE", category=CATEGORY,
            inputs=[io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae"))],
            outputs=[io.Vae.Output()])

    @classmethod
    def execute(cls, vae_name):
        vae, = nodes.VAELoader().load_vae(vae_name)
        use_local_storage(vae.patcher, folder_paths.get_full_path_or_raise("vae", vae_name))
        return io.NodeOutput(vae)


class H3NativeConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3NativeConditioning", display_name="H3 原生｜首尾图与低内存条件", category=CATEGORY,
            inputs=[io.Vae.Input("vae"),
                    io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Int.Input("vision_width", default=1280, min=352, max=1920, step=32),
                    io.Float.Input("reserve_gib", default=12.5, min=0, max=24, step=0.25),
                    io.Image.Input("first_frame"), io.Image.Input("last_frame", optional=True),
                    io.Int.Input("length", default=124, min=5, max=3600, step=17, optional=True,
                                 tooltip="原生帧数，十秒预设为 243。")],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()])

    @classmethod
    def execute(cls, vae, clip_name, prompt, vision_width, reserve_gib, first_frame, last_frame=None, length=124):
        if clip_name.split("/")[-1].split("\\")[-1] != "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors":
            raise ValueError("请选择配套的 H3 NVFP4 文本编码器。")
        width, height = 1920, 1088
        latent, frames = _empty_av_latent(width, height, length)
        keys = [{"resolved_frame_index": 0, "image": _resize(first_frame[:1], width, height, "disabled")}]
        if last_frame is not None:
            keys.append({"resolved_frame_index": frames - 1, "image": _resize(last_frame[:1], width, height, "center")})
        vision_height = max(32, round(height * vision_width / width / 32) * 32)
        stats = {"reads": 0, "bytes": 0, "quantized_requests": 0, "mapped_detaches": 0}
        with LOCK:
            with retain_mapped_nvfp4_scales(stats):
                clip, = nodes.CLIPLoader().load_clip(clip_name, type="minimax", device="default")
            try:
                use_local_storage(clip.patcher, folder_paths.get_full_path_or_raise("text_encoders", clip_name))
                images = [_resize(key["image"], vision_width, vision_height, "disabled") for key in keys]
                with reserve_vram(reserve_gib), qwen_host_embedding(clip), stage_nvfp4_scales(stats):
                    tokens = clip.tokenize(prompt, images=images)
                    conditioning = clip.encode_from_tokens_scheduled(tokens)
                    del tokens, images
            finally:
                model_management.unload_model_and_clones(clip.patcher, unload_additional_models=True)
                del clip
                gc.collect()
            with reserve_vram(reserve_gib):
                for key in keys:
                    key["latent"] = vae.encode(key.pop("image"))
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_keyframes": keys})
        return io.NodeOutput(conditioning, latent)


class H3NativeSampler(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3NativeSampler"
        schema.display_name = "H3 原生｜采样后释放模型"
        schema.category = CATEGORY
        schema.inputs.append(io.Float.Input("reserve_gib", default=11, min=0, max=24, step=0.25))
        schema.inputs.append(io.Vae.Input("keyframe_vae", optional=True))
        return schema

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, reserve_gib, keyframe_vae=None):
        with LOCK:
            if keyframe_vae is not None:
                model_management.unload_model_and_clones(keyframe_vae.patcher, unload_additional_models=False)
            with reserve_vram(reserve_gib):
                result = super().execute(noise, guider, sampler, sigmas, latent_image)
            model_management.unload_model_and_clones(guider.model_patcher, unload_additional_models=False)
        return result


class H3NativeDecodeAudio(VAEDecodeAudio):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3NativeDecodeAudio"
        schema.display_name = "H3 原生｜音频解码"
        schema.category = CATEGORY
        return schema

    @classmethod
    def execute(cls, vae, samples):
        with LOCK:
            result = super().execute(vae=vae, samples=samples)
            model_management.unload_model_and_clones(vae.patcher, unload_additional_models=False)
        return result


class H3NativeSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3NativeSave", display_name="H3 原生｜1080P 流式输出", category=CATEGORY,
            is_output_node=True,
            inputs=[io.Latent.Input("samples"), io.Vae.Input("vae"), io.Audio.Input("audio"),
                    io.String.Input("filename_prefix", default="H3/native1080/video"),
                    io.Int.Input("cq", default=16, min=0, max=51),
                    io.Combo.Input("video_codec", options=["h264_nvenc", "libx264"]),
                    io.Int.Input("output_frames", default=0, min=0, max=3600, step=1, optional=True,
                                 tooltip="写出前 N 帧。0 使用完整原片。240 帧在 24 fps 下是 10 秒。")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo])

    @classmethod
    def execute(cls, samples, vae, audio, filename_prefix, cq, video_codec, output_frames=0):
        value = samples["samples"]
        if value.is_nested:
            value = value.unbind()[0]
        output, name, number, folder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 1920, 1080)
        filename = f"{name}_{number:05}_.mp4"
        with LOCK:
            save_native_video(value, vae, Path(output) / filename, cq=cq, audio=audio,
                metadata={"prompt": cls.hidden.prompt, "workflow": cls.hidden.extra_pnginfo},
                loading_policy="stock_tiled", video_codec=video_codec, output_frames=output_frames)
        return io.NodeOutput.from_dict({"ui": {"images": [{"filename": filename, "subfolder": folder, "type": "output"}],
                                             "animated": [True]}})


class H3NativeExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3NativeUNETLoader, H3NativeVAELoader, H3NativeConditioning,
                H3NativeSampler, H3NativeDecodeAudio, H3NativeSave]


async def comfy_entrypoint():
    return H3NativeExtension()
