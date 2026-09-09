from __future__ import annotations

import gc
import math


def load_t5_encoder_streaming(gguf_path, *, device="cpu", gguf_loader=None, torch_module=None):
    import gguf
    import numpy as np
    import torch
    from gguf import GGUFReader
    from transformers import T5Config, T5EncoderModel

    torch = torch_module or torch
    if gguf_loader is None:
        raise ValueError("gguf_loader is required for T5 GGUF dequantization.")

    key_map = {
        "enc.": "encoder.",
        ".blk.": ".block.",
        "token_embd": "shared",
        "output_norm": "final_layer_norm",
        "attn_q": "layer.0.SelfAttention.q",
        "attn_k": "layer.0.SelfAttention.k",
        "attn_v": "layer.0.SelfAttention.v",
        "attn_o": "layer.0.SelfAttention.o",
        "attn_norm": "layer.0.layer_norm",
        "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
        "ffn_up": "layer.1.DenseReluDense.wi_1",
        "ffn_down": "layer.1.DenseReluDense.wo",
        "ffn_gate": "layer.1.DenseReluDense.wi_0",
        "ffn_norm": "layer.1.layer_norm",
    }

    config = T5Config.from_pretrained("google/flan-t5-xl")
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            model = T5EncoderModel(config)
        model = model.to(dtype=torch.float16)
        model.to_empty(device=device)
    except Exception as exc:
        raise RuntimeError(
            "Streaming T5 loading requires accelerate and empty-weight support. "
            "Restart the runtime and rerun dependency installation."
        ) from exc

    model.eval()
    model.requires_grad_(False)
    parameter_refs = dict(model.named_parameters())
    buffer_refs = dict(model.named_buffers())
    reader = GGUFReader(str(gguf_path))
    quantized_types = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}
    loaded = 0
    skipped = []

    print(f"[Streaming T5 load] {gguf_path} -> {device}")
    with torch.inference_mode():
        for tensor in reader.tensors:
            name = tensor.name
            for old_key, new_key in key_map.items():
                name = name.replace(old_key, new_key)
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            raw = torch.from_numpy(np.array(tensor.data))
            is_quantized = tensor.tensor_type not in quantized_types
            if is_quantized:
                quant_param = gguf_loader.GGUFParameter(raw, quant_type=tensor.tensor_type)
                value = gguf_loader.dequantize_gguf_tensor(quant_param, target_dtype=torch.float16)
            else:
                value = raw.to(dtype=torch.float16)
            if value.numel() != math.prod(shape):
                skipped.append((name, "numel mismatch"))
                del raw, value
                continue
            value = value.reshape(shape)
            target = parameter_refs.get(name)
            if target is None:
                target = buffer_refs.get(name)
            if target is None or tuple(target.shape) != tuple(shape):
                skipped.append((name, "missing or shape mismatch"))
                del raw, value
                continue
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded += 1
            del raw, value

    del reader, parameter_refs, buffer_refs
    gc.collect()
    if str(next(model.parameters()).device) != str(torch.device(device)):
        model.to(device)
    model.eval()
    model.requires_grad_(False)
    print(f"[Streaming T5 load complete] tensors loaded: {loaded}, skipped: {len(skipped)}")
    if skipped:
        print("First skipped tensors:", skipped[:5])
    return model
