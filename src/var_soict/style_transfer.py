from __future__ import annotations

import gc
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps
from infinity.models.basic import CrossAttnBlock, apply_rotary_emb, slow_attn
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_

from .config import ExperimentConfig, ModelBundle


def phi_svd(feature, alpha: float = 1.0, rank: int | None = None):
    """Paper Eq. (5), generalized from VAR [B,C,H,W] to Infinity [B,C,T,H,W]."""
    original_dtype = feature.dtype
    batch, channels = feature.shape[:2]
    spatial_shape = feature.shape[2:]
    outputs = []

    for batch_id in range(batch):
        matrix = feature[batch_id].detach().float().reshape(channels, -1)
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        available_rank = singular_values.numel()
        used_rank = available_rank if rank is None else min(int(rank), available_rank)
        weights = torch.exp(-float(alpha) * torch.arange(used_rank, device=matrix.device, dtype=matrix.dtype))
        weighted_s = singular_values[:used_rank] * weights
        reconstructed = (u[:, :used_rank] * weighted_s.unsqueeze(0)) @ vh[:used_rank]
        outputs.append(reconstructed.reshape(channels, *spatial_shape))

    return torch.stack(outputs).to(dtype=original_dtype)


def principal_feature_blend(generation_feature, style_feature, alpha=1.0, rank=None, strength=1.0):
    """PFB with optional strength multiplier; strength=1 is the paper method."""
    if generation_feature.shape != style_feature.shape:
        raise ValueError(f"PFB shape mismatch: {generation_feature.shape} vs {style_feature.shape}")
    style_feature = style_feature.to(generation_feature)
    style_component = phi_svd(style_feature, alpha=alpha, rank=rank)
    generation_component = phi_svd(generation_feature, alpha=alpha, rank=rank)
    return generation_feature + float(strength) * (style_component - generation_component)


def apply_feature_edit(generation_feature, style_feature, mode, alpha=1.0, rank=None, strength=1.0):
    if mode == "none":
        return generation_feature
    if mode == "replace":
        return style_feature.to(generation_feature)
    if mode == "pfb":
        return principal_feature_blend(generation_feature, style_feature, alpha=alpha, rank=rank, strength=strength)
    raise ValueError(f"Unknown edit mode: {mode}")


class SACController:
    def __init__(self, base_batch=1):
        self.base_batch = base_batch
        self.active = False
        self.sac_strength = 1.0
        self.total_calls = 0
        self.max_q_copy_error = 0.0
        self.max_k_copy_error = 0.0

    def reset_statistics(self):
        self.total_calls = 0
        self.max_q_copy_error = 0.0
        self.max_k_copy_error = 0.0


def _infinity_sac_attention_forward(
    attention,
    x,
    attn_bias_or_two_vector,
    attn_fn=None,
    scale_schedule=None,
    rope2d_freqs_grid=None,
    scale_ind=0,
):
    batch4, length, channels = x.shape
    if attention.using_flash:
        raise RuntimeError("SAC patch expects customized_flash_attn=False.")

    qkv = attention.mat_qkv(x)
    qkv = qkv + torch.cat((attention.q_bias, attention.zero_k_bias, attention.v_bias)).to(qkv)
    qkv = qkv.view(batch4, length, 3, attention.num_heads, attention.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

    if attention.cos_attn:
        scale_mul = attention.scale_mul_1H11.clamp_max(attention.max_scale_mul).exp()
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
        v = v.contiguous()
    else:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    if rope2d_freqs_grid is not None:
        q, k = apply_rotary_emb(
            q,
            k,
            scale_schedule,
            rope2d_freqs_grid,
            attention.pad_to_multiplier,
            attention.rope2d_normalized_by_hw,
            scale_ind,
        )

    controller = getattr(attention, "_paper_sac_controller", None)
    if controller is not None and controller.active:
        b = controller.base_batch
        if batch4 != 4 * b:
            raise RuntimeError(f"SAC expected joint batch {4 * b}, received {batch4}.")

        q_content = torch.cat((q[:b], q[:b], q[2 * b : 3 * b], q[2 * b : 3 * b]), dim=0)
        k_content = torch.cat((k[:b], k[:b], k[2 * b : 3 * b], k[2 * b : 3 * b]), dim=0)
        if controller.sac_strength >= 1.0:
            q, k = q_content, k_content
        else:
            q = q + float(controller.sac_strength) * (q_content - q)
            k = k + float(controller.sac_strength) * (k_content - k)

        controller.total_calls += 1
        controller.max_q_copy_error = max(
            controller.max_q_copy_error,
            float((q[b : 2 * b] - q[:b]).abs().max().detach().cpu()),
            float((q[3 * b : 4 * b] - q[2 * b : 3 * b]).abs().max().detach().cpu()),
        )
        controller.max_k_copy_error = max(
            controller.max_k_copy_error,
            float((k[b : 2 * b] - k[:b]).abs().max().detach().cpu()),
            float((k[3 * b : 4 * b] - k[2 * b : 3 * b]).abs().max().detach().cpu()),
        )

    if attention.caching:
        if attention.cached_k is None:
            attention.cached_k, attention.cached_v = k, v
        else:
            attention.cached_k = torch.cat((attention.cached_k, k), dim=2)
            attention.cached_v = torch.cat((attention.cached_v, v), dim=2)
        k, v = attention.cached_k, attention.cached_v

    if attention.use_flex_attn and attn_fn is not None:
        output = attn_fn(q, k, v, scale=attention.scale).transpose(1, 2).reshape(batch4, length, channels)
    else:
        output = slow_attn(
            query=q.to(v.dtype),
            key=k.to(v.dtype),
            value=v,
            scale=attention.scale,
            attn_mask=attn_bias_or_two_vector,
            dropout_p=0,
        ).transpose(1, 2).reshape(batch4, length, channels)
    return attention.proj_drop(attention.proj(output))


class PaperSACPatch:
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        self.original_forwards = []

    def __enter__(self):
        for block in self.model.unregistered_blocks:
            if not isinstance(block, CrossAttnBlock):
                continue
            attention = block.sa
            self.original_forwards.append((attention, attention.forward))
            attention._paper_sac_controller = self.controller
            attention.forward = types.MethodType(_infinity_sac_attention_forward, attention)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attention, original_forward in self.original_forwards:
            attention.forward = original_forward
            if hasattr(attention, "_paper_sac_controller"):
                delattr(attention, "_paper_sac_controller")
        return False


class StyleTransferEngine:
    def __init__(self, bundle: ModelBundle, config: ExperimentConfig):
        if not bundle.scale_schedule:
            raise ValueError("ModelBundle.scale_schedule must be populated before creating StyleTransferEngine.")
        self.bundle = bundle
        self.config = config
        self.device = bundle.device
        self.text_tokenizer = bundle.text_tokenizer
        self.text_encoder = bundle.text_encoder
        self.vae = bundle.vae
        self.model = bundle.infinity_model
        self.scale_schedule = bundle.scale_schedule
        self.patch_nums = tuple(h for (_, h, _) in self.scale_schedule)
        self.style_image_cache = {}
        self.style_feature_cache = {}
        if not hasattr(self.vae.quantizer, "lfq"):
            self.vae.quantizer.lfq = self.vae.quantizer.bsq
        print("Backend: Infinity-2B GGUF | device:", self.device, "| image size:", (512, 512))

    def load_reference_image(self, path, size=512):
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        image_01 = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(self.device)
        return image_01.mul(2).sub(1), image_01, image

    @torch.no_grad()
    def extract_multiscale_style_features(self, image_m11):
        with torch.amp.autocast("cuda", enabled=False):
            _, _, _, all_bit_indices, _, _ = self.vae.encode(image_m11.float(), scale_schedule=self.scale_schedule)

        summed_codes = None
        features = []
        final_size = self.scale_schedule[-1]
        num_scales = len(self.scale_schedule)
        for step_id, bit_indices in enumerate(all_bit_indices):
            codes = self.vae.quantizer.lfq.indices_to_codes(bit_indices, label_type="bit_label")
            if step_id != num_scales - 1:
                codes = F.interpolate(codes, size=final_size, mode=self.vae.quantizer.z_interplote_up)
            summed_codes = codes if summed_codes is None else summed_codes + codes
            features.append(summed_codes.detach().float().clone())
        assert len(features) == len(self.patch_nums)
        return features

    def get_style_image(self, style_path):
        key = str(Path(style_path))
        if key not in self.style_image_cache:
            _, style_01, _ = self.load_reference_image(style_path)
            self.style_image_cache[key] = style_01.detach().float().cpu()
        return self.style_image_cache[key]

    def get_style_features(self, style_path):
        key = str(Path(style_path))
        if key not in self.style_feature_cache:
            style_m11, _, _ = self.load_reference_image(style_path)
            with torch.inference_mode():
                features = self.extract_multiscale_style_features(style_m11)
            self.style_feature_cache[key] = [feature.detach().float().cpu() for feature in features]
            del style_m11, features
            gc.collect()
            torch.cuda.empty_cache()
        return [feature.to(self.device) for feature in self.style_feature_cache[key]]

    def encode_prompts(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.text_tokenizer(
            text=list(prompts), max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = tokens.input_ids.to(self.device, non_blocking=True)
        mask = tokens.attention_mask.to(self.device, non_blocking=True)
        with torch.no_grad():
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        max_seqlen_k = max(lens)
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        return kv_compact, lens, cu_seqlens_k, max_seqlen_k

    def _sample_bit_labels(self, logits_bl2d, rng, top_k, top_p):
        batch, seq_len = logits_bl2d.shape[:2]
        logits = logits_bl2d.reshape(batch, -1, 2).clone()
        sampled = sample_with_top_k_top_p_also_inplace_modifying_logits_(
            logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
        )[:, :, 0]
        return sampled.reshape(batch, seq_len, -1)

    def _bit_labels_to_codes(self, idx_bld, pn):
        idx = idx_bld.reshape(idx_bld.shape[0], pn[1], pn[2], -1)
        idx = idx.unsqueeze(1)
        return self.vae.quantizer.lfq.indices_to_codes(idx, label_type="bit_label")

    def _next_raw_from_summed_codes(self, summed_codes, next_scale):
        last_stage = F.interpolate(summed_codes, size=next_scale, mode=self.vae.quantizer.z_interplote_up)
        last_stage = last_stage.squeeze(-3)
        if self.model.apply_spatial_patchify:
            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)
        return last_stage

    def _decode_summed_codes_to_image_01(self, summed_codes):
        image = self.vae.decode(summed_codes.squeeze(-3))
        return image.add(1).mul(0.5).clamp(0, 1)

    @torch.no_grad()
    def paper_dual_path_generate(
        self,
        prompt,
        style_features,
        *,
        seed=None,
        cfg=None,
        tau=None,
        top_k=None,
        top_p=None,
        pfb_feature_index=None,
        pfb_feature_indices=None,
        sac_prediction_start=None,
        edit_mode="pfb",
        alpha=None,
        rank=None,
        style_strength=1.0,
        style_decay=1.0,
        style_strength_by_step=None,
        sac_strength=1.0,
        enable_sac=True,
    ):
        cfg = self.config.cfg if cfg is None else cfg
        tau = self.config.tau if tau is None else tau
        top_k = self.config.top_k if top_k is None else top_k
        top_p = self.config.top_p if top_p is None else top_p
        seed = self.config.seed if seed is None else seed
        alpha = self.config.paper_alpha if alpha is None else alpha
        pfb_feature_index = self.config.paper_pfb_feature_index if pfb_feature_index is None else pfb_feature_index
        sac_prediction_start = (
            self.config.paper_sac_prediction_start if sac_prediction_start is None else sac_prediction_start
        )

        if cfg < 1.0:
            raise ValueError("CFG must be >= 1.0 for this dual-stream experiment.")
        if pfb_feature_indices is None:
            pfb_feature_indices = [pfb_feature_index]
        else:
            pfb_feature_indices = sorted(set(int(index) for index in pfb_feature_indices))
        if not pfb_feature_indices or any(index < 0 or index >= len(self.scale_schedule) for index in pfb_feature_indices):
            raise ValueError("Invalid PFB feature indices.")
        if style_strength_by_step is not None:
            style_strength_by_step = {int(step): float(strength) for step, strength in style_strength_by_step.items()}
            expected_steps = set(pfb_feature_indices)
            if set(style_strength_by_step) != expected_steps or any(
                strength < 0 for strength in style_strength_by_step.values()
            ):
                raise ValueError("style_strength_by_step must provide one non-negative strength for every PFB scale.")
        if enable_sac and not 0 <= sac_prediction_start < len(self.scale_schedule):
            raise ValueError("Invalid SAC prediction start.")

        self.model.eval()
        base_batch = 1
        condition_batch = 2
        content_rng = torch.Generator(device=self.device).manual_seed(seed)
        generation_rng = torch.Generator(device=self.device).manual_seed(seed)

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = self.encode_prompts([prompt, prompt])
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total : total + le] = self.model.cfg_uncond[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
        bs = 4

        kv_compact = self.model.text_norm(kv_compact)
        sos = cond_bd = self.model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
        kv_compact = self.model.text_proj_for_ca(kv_compact)
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.model.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast("cuda", enabled=False):
            cond_bd_or_gss = self.model.shared_ada_lin(cond_bd.float()).float().contiguous()

        final_size = self.scale_schedule[-1]
        content_summed = last_stage.new_zeros(base_batch, self.model.d_vae, *final_size)
        generation_summed = torch.zeros_like(content_summed)
        content_trace, generation_trace = [], []

        controller = SACController(base_batch)
        controller.sac_strength = float(sac_strength)
        controller.reset_statistics()

        for block in self.model.unregistered_blocks:
            block.sa.kv_caching(True)

        pre_pfb_max_difference = 0.0
        pfb_relative_change_by_step = {}
        try:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                sac_context = PaperSACPatch(self.model, controller) if enable_sac else nullcontext()
                with sac_context:
                    for step_id, pn in enumerate(self.scale_schedule):
                        controller.active = enable_sac and step_id >= sac_prediction_start
                        need_to_pad = 0
                        attn_fn = None
                        if self.model.use_flex_attn:
                            attn_fn = self.model.attn_fn_compile_dict.get(tuple(self.scale_schedule[: step_id + 1]), None)

                        for block_idx, block_chunk in enumerate(self.model.block_chunks):
                            if self.model.add_lvl_embeding_only_first_block and block_idx == 0:
                                last_stage = self.model.add_lvl_embeding(
                                    last_stage, step_id, self.scale_schedule, need_to_pad=need_to_pad
                                )
                            if not self.model.add_lvl_embeding_only_first_block:
                                last_stage = self.model.add_lvl_embeding(
                                    last_stage, step_id, self.scale_schedule, need_to_pad=need_to_pad
                                )

                            for block in block_chunk.module:
                                last_stage = block(
                                    x=last_stage,
                                    cond_BD=cond_bd_or_gss,
                                    ca_kv=ca_kv,
                                    attn_bias_or_two_vector=None,
                                    attn_fn=attn_fn,
                                    scale_schedule=self.scale_schedule,
                                    rope2d_freqs_grid=self.model.rope2d_freqs_grid,
                                    scale_ind=step_id,
                                )

                        logits = self.model.get_logits(last_stage, cond_bd).mul(1 / float(tau))
                        logits = float(cfg) * logits[:condition_batch] + (1 - float(cfg)) * logits[condition_batch:]
                        content_idx = self._sample_bit_labels(logits[:1], content_rng, top_k, top_p)
                        generation_idx = self._sample_bit_labels(logits[1:2], generation_rng, top_k, top_p)

                        content_codes = self._bit_labels_to_codes(content_idx, pn)
                        generation_codes = self._bit_labels_to_codes(generation_idx, pn)
                        if step_id != len(self.scale_schedule) - 1:
                            content_codes = F.interpolate(
                                content_codes, size=final_size, mode=self.vae.quantizer.z_interplote_up
                            )
                            generation_codes = F.interpolate(
                                generation_codes, size=final_size, mode=self.vae.quantizer.z_interplote_up
                            )

                        content_summed = content_summed + content_codes
                        generation_summed = generation_summed + generation_codes

                        if step_id < min(pfb_feature_indices):
                            pre_pfb_max_difference = max(
                                pre_pfb_max_difference,
                                float((content_summed - generation_summed).abs().max().detach().cpu()),
                            )

                        if step_id in pfb_feature_indices and edit_mode != "none":
                            injection_order = pfb_feature_indices.index(step_id)
                            effective_strength = (
                                style_strength_by_step[step_id]
                                if style_strength_by_step is not None
                                else float(style_strength) * float(style_decay) ** injection_order
                            )
                            generation_before_edit = generation_summed.clone()
                            generation_summed = apply_feature_edit(
                                generation_summed,
                                style_features[step_id],
                                mode=edit_mode,
                                alpha=alpha,
                                rank=rank,
                                strength=effective_strength,
                            )
                            pfb_relative_change_by_step[step_id] = float(
                                (generation_summed - generation_before_edit).norm()
                                / generation_before_edit.norm().clamp_min(1e-8)
                            )

                        content_trace.append(content_summed.detach().float().clone())
                        generation_trace.append(generation_summed.detach().float().clone())

                        if step_id != len(self.scale_schedule) - 1:
                            next_scale = self.scale_schedule[step_id + 1]
                            content_next = self._next_raw_from_summed_codes(content_summed, next_scale)
                            generation_next = self._next_raw_from_summed_codes(generation_summed, next_scale)
                            two_streams = torch.cat((content_next, generation_next), dim=0)
                            last_stage = self.model.word_embed(self.model.norm0_ve(two_streams))
                            last_stage = last_stage.repeat(bs // condition_batch, 1, 1)

            content_image = self._decode_summed_codes_to_image_01(content_summed)
            generation_image = self._decode_summed_codes_to_image_01(generation_summed)
            return {
                "content_image_01": content_image,
                "stylized_image_01": generation_image,
                "content_features": content_trace,
                "generation_features": generation_trace,
                "pre_pfb_max_difference": pre_pfb_max_difference,
                "sac_calls": controller.total_calls,
                "max_q_copy_error": controller.max_q_copy_error,
                "max_k_copy_error": controller.max_k_copy_error,
                "pfb_relative_change_by_step": pfb_relative_change_by_step,
            }
        finally:
            controller.active = False
            for block in self.model.unregistered_blocks:
                block.sa.kv_caching(False)

    def generate_content_image(self, prompt):
        with torch.inference_mode():
            result = self.paper_dual_path_generate(prompt, [], edit_mode="none", enable_sac=False)
        image = result["content_image_01"].detach().float().cpu()
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return image

    def generate_variant_image(
        self,
        prompt,
        style_path,
        *,
        pfb_feature_indices,
        style_decay,
        style_strength,
        enable_sac,
        rank=None,
        style_strength_by_step=None,
    ):
        style_features = self.get_style_features(style_path)
        with torch.inference_mode():
            result = self.paper_dual_path_generate(
                prompt,
                style_features,
                pfb_feature_indices=pfb_feature_indices,
                edit_mode="pfb",
                rank=rank,
                style_strength=style_strength,
                style_decay=style_decay,
                style_strength_by_step=style_strength_by_step,
                sac_strength=1.0,
                enable_sac=enable_sac,
            )
        image = result["stylized_image_01"].detach().float().cpu()
        del style_features, result
        gc.collect()
        torch.cuda.empty_cache()
        return image

