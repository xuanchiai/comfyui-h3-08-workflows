"""Experimental native H3 RGB streaming delivery, GPL-3.0.

Adapted from local ComfyUI-H3-X2-Delivery/streaming.py; see NOTICE.md.
No ComfyUI imports occur until save_native_video() is explicitly called.
This module does not change a decoder's weights, channels, tiling or precision.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import gc
import json
import math
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import time

import av
import torch


def color_srgb(target):
    target.color_primaries = 1
    target.color_trc = 13
    target.colorspace = 1
    target.color_range = 1


def precision_state(vae):
    return {"vae_dtype": str(vae.vae_dtype),
            "fp16_accumulation": getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None)}


def decode_memory_request(vae, video, policy):
    if policy == "full_shape":
        bounded = tuple(video.shape)
    elif policy == "stock_tiled":
        spatial = vae.spacial_compression_decode()
        temporal = vae.temporal_compression_decode()
        tile_t = max(2, 64 // temporal) if temporal is not None else None
        bounded = vae._tile_bounded_shape(video.shape, 512 // spatial, 512 // spatial, tile_t)
    else:
        raise ValueError("Unknown VAE loading policy")
    return int(vae.memory_used_decode(bounded, vae.vae_dtype)), tuple(bounded)


def validate_shape(shape, vertical_crop):
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 3:
        raise ValueError("Expected native RGB [1,3,T,H,W]; packed X2 RGB is unsupported")
    _, _, frames, height, width = shape
    if min(frames, height, width) <= 0:
        raise ValueError("Empty output shape")
    if vertical_crop < 0 or vertical_crop % 2 or 2 * vertical_crop >= height:
        raise ValueError("Crop must be nonnegative, even and leave positive height")
    if width % 2 or (height - 2 * vertical_crop) % 2:
        raise ValueError("H.264 yuv420p requires even dimensions")
    return int(width), int(height - 2 * vertical_crop)


def native_uint8_tensor(part, vertical_crop):
    """Only native layout conversion, center crop and video uint8 quantization."""
    if part.ndim != 5 or part.shape[0] != 1 or part.shape[1] != 3 or part.dtype != torch.float32:
        raise ValueError("Finalized native H3 chunks must be float32 [1,3,T,H,W]")
    pixels = part[0].permute(1, 2, 3, 0)
    if vertical_crop:
        pixels = pixels[:, vertical_crop:-vertical_crop]
    return pixels.mul(255).clamp_(0, 255).to(torch.uint8).contiguous()


def native_uint8(part, vertical_crop):
    return native_uint8_tensor(part, vertical_crop).cpu().numpy()


@contextmanager
def decode_tile_batch(inner, cap):
    if cap == 4:
        yield
        return
    from comfy import model_management

    def decode_row(z_row, x_idx, x_len):
        free = model_management.get_free_memory(z_row.device)
        batch = int(max(1, min(cap, free // (128 * 2**20 * z_row.shape[0]))))
        tiles = [z_row[..., start // inner.vae_ratio:(start + width) // inner.vae_ratio]
                 for start, width in zip(x_idx, x_len)]
        for offset in range(0, len(tiles), batch):
            group = tiles[offset:offset + batch]
            yield from inner._decode_pixels(torch.cat(group)).chunk(len(group))

    if '_decode_tile_row' in vars(inner):
        raise RuntimeError('Unexpected existing native VAE tile override')
    inner._decode_tile_row = decode_row
    try:
        yield
    finally:
        del inner._decode_tile_row


class NativeRGBSink:
    """Native finalized temporal chunks, bounded to one encoder future."""
    def __init__(self, output, stream, shape, vertical_crop=4, interrupt=lambda: None, audit=None, output_frames=0):
        validate_shape(shape, vertical_crop)
        self.output, self.stream, self.shape = output, stream, tuple(shape)
        self.vertical_crop, self.interrupt = vertical_crop, interrupt
        self.audit = audit
        self.output_frames = output_frames or self.shape[2]
        self.written = 0
        self.pending = None
        self.future = None
        self.rgb_buffer = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-native-nvenc")
        self.conversion_s = 0.0
        self.encoder_wait_s = 0.0

    def __getitem__(self, key):
        if self.pending is not None:
            raise ValueError("A pending output slice was not written")
        if not isinstance(key, tuple) or len(key) != 5 or any(key[i] != slice(None) for i in (0, 1, 3, 4)):
            raise ValueError("Native H3 output-buffer contract changed")
        time_slice = key[2]
        if not isinstance(time_slice, slice) or time_slice.step is not None or time_slice.start != self.written:
            raise ValueError("Finalized chunks must be written sequentially")
        if time_slice.stop is None or not self.written < time_slice.stop <= self.shape[2]:
            raise ValueError("Output slice extends beyond expected frames")
        self.pending = time_slice.stop - time_slice.start
        return self

    def copy_(self, part):
        self.interrupt()
        if part.ndim != 5 or tuple(part.shape) != (1, 3, self.pending, self.shape[3], self.shape[4]):
            raise ValueError("Finalized chunk dimensions do not match the reserved slice")
        if self.audit is not None:
            self.audit.record(part, self.written, self.vertical_crop)
        if self.future is not None:
            started = time.perf_counter()
            self.future.result()
            self.encoder_wait_s += time.perf_counter() - started
        started = time.perf_counter()
        pixels = native_uint8_tensor(part, self.vertical_crop)
        if self.rgb_buffer is None or self.rgb_buffer.shape[0] < pixels.shape[0]:
            self.rgb_buffer = torch.empty_like(pixels, device='cpu')
        view = self.rgb_buffer[:pixels.shape[0]]
        view.copy_(pixels)
        self.conversion_s += time.perf_counter() - started
        self.future = self.worker.submit(self.encode, view.numpy(), self.written)
        self.written += self.pending
        self.pending = None
        return self

    def encode(self, pixels, first_index):
        rate = self.stream.average_rate
        count = max(0, self.output_frames - first_index)
        for offset, pixels_frame in enumerate(pixels[:count]):
            frame = av.VideoFrame.from_ndarray(pixels_frame, format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace=1)
            color_srgb(frame)
            frame.pts = first_index + offset
            frame.time_base = Fraction(rate.denominator, rate.numerator)
            self.output.mux(self.stream.encode(frame))

    def finish(self):
        if self.future is not None:
            started = time.perf_counter()
            self.future.result()
            self.encoder_wait_s += time.perf_counter() - started
        if self.pending is not None or self.written != self.shape[2]:
            raise RuntimeError(f"Incomplete decode: {self.written}/{self.shape[2]} frames")
        self.output.mux(self.stream.encode(None))

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)


def validate_decoder(vae, video):
    vae.throw_exception_if_invalid()
    inner = vae.first_stage_model
    if inner.__class__.__name__ != "MiniMaxH3VideoVAE" or getattr(inner, "noncausal", True):
        raise ValueError("Requires the stock causal MiniMaxH3VideoVAE, not Sol/Image/X2")
    if getattr(inner.decoder, "out_channels", None) != 3:
        raise ValueError("Requires original native RGB decoder output channels")
    if (inner.tiling, inner.tile_size, inner.tile_overlap_min) != (True, 256, 64):
        raise ValueError("Decoder must retain its reference tiling=True, tile_size=256, overlap=64")
    if any(getattr(inner, key).shape[1] != 3 for key in ("pixel_mean", "pixel_std")):
        raise ValueError("Native RGB normalization buffers are required")
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError("Expected video latent [1,24,T,H,W]")
    if video.shape[2] < 2 or (video.shape[2] - 2) % 5:
        raise ValueError("Use H3 latent length 2+5*k, corresponding to video length 5+17*k")
    shape = tuple(inner.decode_output_shape(video.shape))
    if shape[-2:] != (video.shape[-2] * 16, video.shape[-1] * 16):
        raise ValueError("Native decoder must preserve its spatial 16x ratio")
    return inner, shape


def save_native_video(video, vae, target, *, fps=24, vertical_crop=4, cq=16, nvenc_gpu=0, audio=None, metadata=None, audit=None, loading_policy="full_shape", tile_batch=4, video_codec="h264_nvenc", output_frames=0):
    """Call explicitly with a loaded original VAE; no service or registration needed.

    Timings cover VAE loading to GPU, decode and output encoding. They exclude
    checkpoint construction, audio decoding, diffusion and source loading.
    """
    import comfy.model_management as mm
    import comfy.model_prefetch

    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    if video_codec not in ("h264_nvenc", "libx264") or video_codec not in av.codecs_available:
        raise RuntimeError(f"Unsupported H.264 encoder: {video_codec}")
    if vae.device.type != "cuda":
        raise ValueError("The native decoder needs a CUDA VAE")
    if not math.isfinite(fps) or fps <= 0 or not 0 <= cq <= 51:
        raise ValueError("Invalid FPS or NVENC CQ")
    inner, shape = validate_decoder(vae, video)
    output_frames = output_frames or shape[2]
    if not 1 <= output_frames <= shape[2]:
        raise ValueError(f"Output frames must be between 1 and {shape[2]}")
    memory_required, estimated_shape = decode_memory_request(vae, video, loading_policy)
    width, height = validate_shape(shape, vertical_crop)
    rate = Fraction(str(fps)).limit_denominator(100000)
    waveform, layout, sample_rate = None, None, None
    if audio is not None:
        sample_rate = int(audio["sample_rate"])
        data = audio["waveform"]
        if sample_rate <= 0 or data.ndim != 3 or data.shape[0] != 1 or data.shape[1] not in (1, 2):
            raise ValueError("Expected one mono/stereo [1,C,T] audio waveform")
        waveform = data[0, :, :math.ceil(sample_rate / rate * output_frames)].float().cpu().contiguous().numpy()
        if not waveform.shape[1]:
            raise ValueError("Empty audio")
        layout = "mono" if waveform.shape[0] == 1 else "stereo"
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".h3-native-", suffix=".part.mp4", dir=target.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    started = time.perf_counter()
    timings = {}
    try:
        with av.open(str(temporary), "w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
            output.metadata["h3_native1080"] = json.dumps(metadata or {}, ensure_ascii=False)
            stream = output.add_stream(video_codec, rate=rate)
            stream.width, stream.height, stream.pix_fmt, stream.bit_rate = width, height, "yuv420p", 0
            if video_codec == "libx264":
                stream.codec_context.thread_count = 2
                stream.options = {"preset": "veryfast", "tune": "zerolatency", "crf": str(cq)}
            else:
                stream.options = {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": str(cq), "gpu": str(nvenc_gpu)}
            color_srgb(stream.codec_context)
            audio_stream = output.add_stream("aac", rate=sample_rate, layout=layout) if waveform is not None else None
            if audio_stream is not None:
                audio_stream.bit_rate = 192000
            output.start_encoding()
            with mm.cuda_device_context(vae.device):
                load_started = time.perf_counter()
                with comfy.model_prefetch.pause_malloc_graph():
                    mm.load_models_gpu([vae.patcher], memory_required=memory_required, force_full_load=vae.disable_offload)
                timings["vae_gpu_load_s"] = time.perf_counter() - load_started
                validate_decoder(vae, video)
                sink = NativeRGBSink(output, stream, shape, vertical_crop, mm.throw_exception_if_processing_interrupted,
                                     audit, output_frames=output_frames)
                decode_started = time.perf_counter()
                try:
                    with decode_tile_batch(inner, tile_batch):
                        inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=sink)
                    sink.finish()
                    timings.update(decode_and_video_encode_s=time.perf_counter() - decode_started,
                                   chunk_uint8_and_download_s=sink.conversion_s,
                                   encoder_wait_s=sink.encoder_wait_s)
                finally:
                    sink.close()
            if audio_stream is not None:
                frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
                frame.sample_rate, frame.pts = sample_rate, 0
                output.mux(audio_stream.encode(frame))
                output.mux(audio_stream.encode(None))
        mm.throw_exception_if_processing_interrupted()
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    timings["delivery_wall_s"] = time.perf_counter() - started
    if audit is not None:
        audit.finish()
        timings["rgb_audit_overhead_s"] = audit.elapsed_s
    return {"target": str(target), "width": width, "height": height, "frames": output_frames, "decoded_frames": shape[2], "fps": float(rate),
            "codec": video_codec, "cq": cq, "native_decoder": True, "pixel_shuffle": False,
            "tile_size": 256, "tile_overlap": 64, "tile_batch_cap": tile_batch, "float32_full_buffer_avoided_gib": math.prod(shape)*4/1024**3,
            "loading_policy": loading_policy, "estimated_latent_shape": estimated_shape, "memory_required_gib": memory_required/1024**3,
            "timings": timings, "precision": precision_state(vae), "measurement_scope": "decode-delivery only, not full generation"}

