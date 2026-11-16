"""Read-only mapped Qwen weights and exact NVFP4 scale transfer."""
import dataclasses
from contextlib import contextmanager
from threading import get_ident


@contextmanager
def retain_mapped_nvfp4_scales(stats):
    import torch
    from comfy_kitchen.tensor.base import _DISPATCH_TABLE
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    op = torch.ops.aten.detach.default
    original = _DISPATCH_TABLE[op]
    owner = get_ident()

    def detach(weight, args, kwargs):
        params = weight._params
        if (get_ident() == owner and type(params) is TensorCoreNVFP4Layout.Params and params.block_scale.device.type == 'cpu'
                and getattr(params.block_scale.untyped_storage(), '_comfy_tensor_file_slice', None) is not None):
            stats['mapped_detaches'] += 1
            params = dataclasses.replace(params, scale=params.scale.detach(), block_scale=params.block_scale.detach())
            return weight._copy_with(qdata=weight._qdata.detach(), params=params, clone_params=False)
        return original(weight, args, kwargs)

    _DISPATCH_TABLE[op] = detach
    try:
        yield
    finally:
        _DISPATCH_TABLE[op] = original


@contextmanager
def stage_nvfp4_scales(stats):
    from comfy import memory_management
    from comfy.quant_ops import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    original = memory_management.read_tensor_file_slice_into
    owner = get_ident()

    def read_compatible(tensor, destination, stream=None, destination2=None):
        if not (get_ident() == owner and isinstance(tensor, QuantizedTensor)
                and type(tensor._params) is TensorCoreNVFP4Layout.Params):
            return original(tensor, destination, stream, destination2)
        stats['quantized_requests'] += 1
        recipients = [value for value in (destination, destination2) if value is not None]
        if not recipients or any(not isinstance(value, QuantizedTensor)
                                 or type(value._params) is not TensorCoreNVFP4Layout.Params for value in recipients):
            return original(tensor, destination, stream, destination2)
        source = tensor._params
        block = source.block_scale
        outputs = [value._params.block_scale for value in recipients]
        file_slice = getattr(block.untyped_storage(), '_comfy_tensor_file_slice', None)
        if not (file_slice and file_slice.file_ref is not None
                and block.device.type == 'cpu' and block.storage_offset() == 0 and block.is_contiguous()
                and file_slice.size == block.numel() * block.element_size()
                and all(out.dtype == block.dtype and out.shape == block.shape and out.is_contiguous() for out in outputs)):
            return original(tensor, destination, stream, destination2)
        qdata_out = destination._qdata if destination is not None else None
        qdata_out2 = destination2._qdata if destination2 is not None else None
        if not original(tensor._qdata, qdata_out, stream, qdata_out2):
            return original(tensor, destination, stream, destination2)
        host_out = destination._params.block_scale if destination is not None else None
        device_out = destination2._params.block_scale if destination2 is not None else None
        if not original(block, host_out, stream, device_out):
            return original(tensor, destination, stream, destination2)
        for value in recipients:
            target = value._params
            target.scale.copy_(source.scale, non_blocking=target.scale.device.type == 'cuda')
            value._params = dataclasses.replace(target, orig_shape=source.orig_shape, transposed=source.transposed)
        stats['reads'] += 1
        stats['bytes'] += file_slice.size
        return True

    memory_management.read_tensor_file_slice_into = read_compatible
    try:
        yield
    finally:
        memory_management.read_tensor_file_slice_into = original


@contextmanager
def qwen_host_embedding(clip):
    embedding = clip.cond_stage_model.qwen3vl_32b.transformer.get_input_embeddings()
    if (embedding.quant_format != 'int8_tensorwise' or embedding.weight.device.type != 'cpu'
            or embedding.weight_function or 'forward_comfy_cast_weights' in vars(embedding)):
        raise RuntimeError('Expected an unpatched CPU-backed INT8 Qwen embedding')
    embedding.forward_comfy_cast_weights = embedding.host_rows
    try:
        yield
    finally:
        del embedding.forward_comfy_cast_weights


