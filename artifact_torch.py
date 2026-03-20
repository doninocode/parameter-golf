from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(
    name: str,
    t: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    fp32_name_patterns: tuple[str, ...],
    keep_float_store_dtype: torch.dtype,
) -> Tensor:
    if any(pattern in name for pattern in fp32_name_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=keep_float_store_dtype).contiguous()
    return t


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
    fp32_name_patterns: tuple[str, ...],
    keep_float_max_numel: int,
    keep_float_store_dtype: torch.dtype,
    per_row_scale_dtype: torch.dtype,
    clip_q: float,
):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(("baseline_tensor_bytes", "payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= keep_float_max_numel:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes, fp32_name_patterns, keep_float_store_dtype)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
            clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
            s = scale.to(dtype=per_row_scale_dtype).contiguous()
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
            s = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / s), -127, 127).to(torch.int8).contiguous()
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def quantize_state_dict_bitnet(
    state_dict: dict[str, Tensor],
    bitlinear_names: set[str],
    fp32_name_patterns: tuple[str, ...],
    keep_float_max_numel: int,
    keep_float_store_dtype: torch.dtype,
    scale_eps: float,
):
    packed: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = dict.fromkeys(("baseline_tensor_bytes", "payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if name in bitlinear_names and t.ndim == 2 and t.is_floating_point():
            scale = t.float().abs().mean(dim=1).clamp_min(scale_eps).to(dtype=torch.float16).contiguous()
            bits = (t.float() >= 0).to(torch.uint8).reshape(-1).cpu().numpy()
            packed_arr = np.packbits(bits, bitorder="little")
            packed_t = torch.from_numpy(packed_arr.copy())
            packed[name] = packed_t
            scales[name] = scale
            shapes[name] = tuple(t.shape)
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["payload_bytes"] += tensor_nbytes(packed_t) + tensor_nbytes(scale)
            continue
        kept = keep_float_tensor(name, t, passthrough_orig_dtypes, fp32_name_patterns, keep_float_store_dtype)
        passthrough[name] = kept
        stats["payload_bytes"] += tensor_nbytes(kept)
    obj: dict[str, object] = {
        "__quant_format__": "bitnet_sign_rowpack_v1",
        "packed": packed,
        "scales": scales,
        "shapes": shapes,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_bitnet(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, packed in obj["packed"].items():
        shape = tuple(obj["shapes"][name])
        numel = int(np.prod(shape))
        bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=numel, bitorder="little")
        signs = torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0).view(shape)
        scale = obj["scales"][name].float().view(shape[0], *([1] * (len(shape) - 1)))
        dtype = getattr(torch, obj["dtypes"][name])
        out[name] = (signs * scale).to(dtype=dtype).contiguous()
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def quantize_state_dict_artifact(
    state_dict: dict[str, Tensor],
    *,
    bitlinear_names: set[str],
    fp32_name_patterns: tuple[str, ...],
    keep_float_max_numel: int,
    keep_float_store_dtype: torch.dtype,
    per_row_scale_dtype: torch.dtype,
    clip_q: float,
    scale_eps: float,
):
    if bitlinear_names:
        return quantize_state_dict_bitnet(
            state_dict,
            bitlinear_names,
            fp32_name_patterns,
            keep_float_max_numel,
            keep_float_store_dtype,
            scale_eps,
        )
    return quantize_state_dict_int8(
        state_dict,
        fp32_name_patterns,
        keep_float_max_numel,
        keep_float_store_dtype,
        per_row_scale_dtype,
        clip_q,
    )


def dequantize_state_dict_artifact(obj: dict[str, object]) -> dict[str, Tensor]:
    fmt = obj.get("__quant_format__")
    if fmt == "bitnet_sign_rowpack_v1":
        return dequantize_state_dict_bitnet(obj)
    if fmt == "int8_clean_per_row_v1":
        return dequantize_state_dict_int8(obj)
    raise ValueError(f"Unsupported artifact format: {fmt}")
