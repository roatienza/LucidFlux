#!/usr/bin/env python3
"""
Minimal Gradio UI for LucidFlux image restoration.

Loads the Flux backbone, LucidFlux conditioner weights, and SwinIR prior once,
then exposes a single-image restoration endpoint.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import gradio.routes
import numpy as np
import math
import torch
from PIL import Image

from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder  # noqa: F401
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipVisionModel

from inference import (
    DualConditionBranch,
    Modulation,
    load_redux_image_encoder,
    prepare_with_embeddings,
)
from src.flux.align_color import wavelet_reconstruction
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor
from src.flux.modules.layers import IPDoubleStreamBlockProcessor, ImageProjModel
from src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack
from src.flux.swinir import SwinIR
from src.flux.util import load_ae, load_checkpoint, load_flow_model, load_single_condition_branch

# Patch Gradio API schema helpers to tolerate boolean schema nodes (see gradio#11059).
try:  # pragma: no cover
    import gradio_client.utils as grc_utils  # type: ignore

    _orig_json_schema_to_python_type = grc_utils._json_schema_to_python_type  # type: ignore
    _orig_get_type = grc_utils.get_type  # type: ignore

    def _safe_json_schema_to_python_type(schema, defs):  # type: ignore
        if isinstance(schema, bool):
            return "typing.Any"
        return _orig_json_schema_to_python_type(schema, defs)

    def _safe_get_type(schema):  # type: ignore
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    grc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore
    grc_utils.get_type = _safe_get_type  # type: ignore
except Exception:  # pragma: no cover
    pass

# Work around a Gradio 4.44 schema regression: fall back gracefully if API schema
# generation fails, but keep the normal behaviour otherwise.
if hasattr(gradio.routes, "App") and hasattr(gradio.routes.App, "get_api_info"):
    _original_app_get_api_info = gradio.routes.App.get_api_info

    def _safe_app_get_api_info(self):
        try:
            return _original_app_get_api_info(self)
        except TypeError as exc:  # pragma: no cover
            print(f"[WARN] Falling back from App.get_api_info due to: {exc}", flush=True)
            return {}

    gradio.routes.App.get_api_info = _safe_app_get_api_info

if hasattr(gr.blocks.Blocks, "get_api_info"):
    _original_blocks_get_api_info = gr.blocks.Blocks.get_api_info

    def _safe_blocks_get_api_info(self):
        try:
            return _original_blocks_get_api_info(self)
        except TypeError as exc:  # pragma: no cover
            print(f"[WARN] Falling back from Blocks.get_api_info due to: {exc}", flush=True)
            return {}

    gr.blocks.Blocks.get_api_info = _safe_blocks_get_api_info


def _load_env_exports(env_path: Path) -> None:
    """Populate os.environ with entries from weights/env.sh if present."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            key, _, value = line[7:].partition("=")
            os.environ[key.strip()] = value.strip()


class LucidFluxEngine:
    """Wraps LucidFlux inference so we can reuse a single loaded model."""

    def __init__(
        self,
        weights_root: Path,
        device: Optional[str] = None,
        offload: bool = False,
    ) -> None:
        self.weights_root = weights_root
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.offload = offload and self.device.type == "cuda"
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._lock = threading.Lock()
        ip_env_path = os.getenv("LUCIDFLUX_IP_ADAPTER_PATH")
        self.ip_adapter_name = os.getenv("LUCIDFLUX_IP_ADAPTER_NAME", "flux-ip-adapter.safetensors")
        self.ip_adapter_repo = os.getenv("LUCIDFLUX_IP_ADAPTER_REPO", "XLabs-AI/flux-ip-adapter")
        self.ip_image_encoder_id = os.getenv("LUCIDFLUX_IP_IMAGE_ENCODER", "openai/clip-vit-large-patch14")
        self.ip_adapter_path = Path(ip_env_path).expanduser() if ip_env_path else (self.weights_root / "ip_adapter" / self.ip_adapter_name)
        self._ip_context_dim = 4096
        self._ip_hidden_dim = 3072
        self._ip_extra_tokens = 4
        self._ip_setup = False
        self._ip_attn_processors: dict[str, torch.nn.Module] = {}
        self._ip_image_encoder: Optional[CLIPVisionModelWithProjection] = None
        self._ip_clip_processor: Optional[CLIPImageProcessor] = None
        self._ip_proj_module: Optional[ImageProjModel] = None
        self._load_backbone()
        self._load_conditioners()
        self._load_aux_modules()
        self.log_path = (self.weights_root.parent / "logs" / "inference_log.csv").resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text(
                "timestamp,input_width,input_height,output_width,output_height,max_side,steps,guidance,control_weight,guidance_multiplier,prior_blend,reference_strength,reference_applied,seed,duration_sec,xformers_enabled\n"
            )

    def _load_backbone(self) -> None:
        """Load Flux backbone (flow + autoencoder)."""
        name = "flux-dev"
        load_device = "cpu" if self.offload else str(self.device)
        self.model = load_flow_model(name, device=load_device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._base_attn_processors = {k: v for k, v in self.model.attn_processors.items()}
        self._attn_state = "base"
        self.xformers_enabled = False
        try:
            import xformers  # noqa: F401

            if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                self.model.enable_xformers_memory_efficient_attention()
                self.xformers_enabled = True
        except Exception:
            self.xformers_enabled = False

        self.ae = load_ae(name, device=load_device)
        self.ae = self.ae.to(self.device)
        self.ae.eval()

    def _load_conditioners(self) -> None:
        """Load LucidFlux dual-condition branches + modulation weights."""
        ckpt_path = self.weights_root / "lucidflux" / "lucidflux.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing LucidFlux checkpoint at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        self.condition_lq = load_single_condition_branch("flux-dev", str(self.device)).to(torch.bfloat16)
        self.condition_lq.load_state_dict(checkpoint["condition_lq"], strict=False)
        self.condition_lq = self.condition_lq.to(self.device)

        self.condition_ldr = load_single_condition_branch("flux-dev", str(self.device)).to(torch.bfloat16)
        self.condition_ldr.load_state_dict(checkpoint["condition_ldr"], strict=False)
        self.condition_ldr = self.condition_ldr.to(self.device)

        self.modulation_lq = Modulation(dim=3072).to(torch.bfloat16)
        self.modulation_lq.load_state_dict(checkpoint["modulation_lq"], strict=False)
        self.modulation_lq = self.modulation_lq.to(self.device)

        self.modulation_ldr = Modulation(dim=3072).to(torch.bfloat16)
        self.modulation_ldr.load_state_dict(checkpoint["modulation_ldr"], strict=False)
        self.modulation_ldr = self.modulation_ldr.to(self.device)

        self.dual_condition_branch = DualConditionBranch(
            self.condition_lq,
            self.condition_ldr,
            modulation_lq=self.modulation_lq,
            modulation_ldr=self.modulation_ldr,
        ).to(self.device)
        self.dual_condition_branch.eval()

        embed_path = self.weights_root / "lucidflux" / "prompt_embeddings.pt"
        if not embed_path.exists():
            raise FileNotFoundError(f"Missing prompt embeddings at {embed_path}")
        embeddings = torch.load(embed_path, map_location="cpu")
        self.precomputed_txt = embeddings["txt"].to(self.device, dtype=self.dtype)
        self.precomputed_vec = embeddings["vec"].to(self.device, dtype=self.dtype)
        self.precomputed_prompt = embeddings.get("prompt", "LucidFlux")

        self.connector_state = checkpoint["connector"]

    def _load_aux_modules(self) -> None:
        """Load SwinIR prior, SigLIP encoder, and Redux image encoder."""
        swinir_path = self.weights_root / "swinir.pth"
        if not swinir_path.exists():
            raise FileNotFoundError(f"Missing SwinIR prior at {swinir_path}")
        swinir_ckpt = torch.load(swinir_path, map_location="cpu")
        state_dict = swinir_ckpt.get("state_dict", swinir_ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.swinir = SwinIR(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
            window_size=8,
            mlp_ratio=2,
            sf=8,
            img_range=1.0,
            upsampler="nearest+conv",
            resi_connection="1conv",
            unshuffle=True,
            unshuffle_scale=8,
        )
        self.swinir.load_state_dict(state_dict, strict=False)
        self.swinir = self.swinir.to(self.device)
        self.swinir.eval()
        for param in self.swinir.parameters():
            param.requires_grad_(False)

        self.siglip_model = SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch16-512")
        self.siglip_model = self.siglip_model.to(self.device, dtype=self.dtype)
        self.siglip_model.eval()

        self.redux_image_encoder = load_redux_image_encoder(
            self.device,
            self.dtype,
            self.connector_state,
        )

    @staticmethod
    def _clone_attn_dict(source: dict[str, torch.nn.Module]) -> dict[str, torch.nn.Module]:
        return {key: value for key, value in source.items()}

    def _setup_ip_adapter(self) -> None:
        if self._ip_setup:
            return

        checkpoint = None
        if self.ip_adapter_path and self.ip_adapter_path.exists():
            checkpoint = load_checkpoint(str(self.ip_adapter_path), None, None)
        else:
            try:
                checkpoint = load_checkpoint(None, self.ip_adapter_repo, self.ip_adapter_name)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Unable to load IP-Adapter weights from {self.ip_adapter_repo}/{self.ip_adapter_name}"
                ) from exc

        proj_state = {
            k[len("ip_adapter_proj_model.") :]: v
            for k, v in checkpoint.items()
            if k.startswith("ip_adapter_proj_model.")
        }
        if not proj_state:
            raise RuntimeError("IP-Adapter checkpoint missing projection weights.")

        proj_module = ImageProjModel(self._ip_context_dim, 768, self._ip_extra_tokens)
        proj_module.load_state_dict(proj_state, strict=False)
        proj_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._ip_proj_module = proj_module.to(self.device, dtype=proj_dtype)

        encoder_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.ip_image_encoder_id
        ).to(self.device, dtype=encoder_dtype)
        self._ip_image_encoder.eval()
        self._ip_clip_processor = CLIPImageProcessor()

        ip_attn_procs: dict[str, torch.nn.Module] = {}
        for name, base_proc in self._base_attn_processors.items():
            prefix = f"{name}."
            ip_state = {
                k[len(prefix) :]: v
                for k, v in checkpoint.items()
                if k.startswith(prefix)
            }
            if ip_state:
                proc = IPDoubleStreamBlockProcessor(self._ip_context_dim, self._ip_hidden_dim)
                proc.load_state_dict(ip_state, strict=False)
                ip_attn_procs[name] = proc.to(self.device, dtype=torch.bfloat16 if self.device.type == "cuda" else self.dtype)
            else:
                ip_attn_procs[name] = base_proc

        self._ip_attn_processors = ip_attn_procs
        self._ip_setup = True

    def _enable_ip_adapter(self) -> None:
        self._setup_ip_adapter()
        if self._attn_state != "ip":
            self.model.set_attn_processor(self._clone_attn_dict(self._ip_attn_processors))
            self._attn_state = "ip"

    def _disable_ip_adapter(self) -> None:
        if self._attn_state == "ip":
            self.model.set_attn_processor(self._clone_attn_dict(self._base_attn_processors))
            self._attn_state = "base"

    def _encode_reference_image(self, reference: Image.Image) -> torch.Tensor:
        if not self._ip_setup or not self._ip_proj_module or not self._ip_image_encoder or not self._ip_clip_processor:
            raise RuntimeError("IP-Adapter is not initialised.")

        ref_rgb = reference.convert("RGB")
        clip_inputs = self._ip_clip_processor(images=ref_rgb, return_tensors="pt").pixel_values
        clip_inputs = clip_inputs.to(self._ip_image_encoder.device, dtype=self._ip_image_encoder.dtype)
        with torch.inference_mode():
            embeds = self._ip_image_encoder(clip_inputs).image_embeds
        tokens = self._ip_proj_module(embeds.to(self.device, dtype=torch.bfloat16 if self.device.type == "cuda" else self.dtype))
        return tokens


    @torch.inference_mode()
    def restore(
        self,
        image: Image.Image,
        *,
        max_side: int = 1024,
        steps: int = 20,
        guidance: float = 4.0,
        seed: int = -1,
        true_gs: float = 1.0,
        control_weight: float = 1.0,
        prior_blend: float = 0.3,
        reference_image: Optional[Image.Image] = None,
        reference_strength: float = 0.0,
    ) -> tuple[Image.Image, dict]:
        if image is None:
            raise ValueError("Input image is required.")
        with self._lock:
            return self._restore_impl(
                image=image,
                max_side=max_side,
                steps=steps,
                guidance=guidance,
                seed=seed,
                true_gs=true_gs,
                control_weight=control_weight,
                prior_blend=prior_blend,
                reference_image=reference_image,
                reference_strength=reference_strength,
            )

    def _restore_impl(
        self,
        image: Image.Image,
        max_side: int,
        steps: int,
        guidance: float,
        seed: int,
        true_gs: float,
        control_weight: float,
        prior_blend: float,
        reference_image: Optional[Image.Image],
        reference_strength: float,
    ) -> tuple[Image.Image, dict]:
        start_time = time.perf_counter()
        orig_w, orig_h = image.size
        if orig_w == 0 or orig_h == 0:
            raise ValueError("Input image has zero width or height.")

        max_side = max(64, int(max_side))
        longest = max(orig_w, orig_h)
        scale = min(1.0, max_side / float(longest)) if longest > 0 else 1.0
        scaled_w = max(orig_w * scale, 1.0)
        scaled_h = max(orig_h * scale, 1.0)
        base = 64
        width = max(base, int(math.ceil(scaled_w / base)) * base)
        height = max(base, int(math.ceil(scaled_h / base)) * base)

        resampling = getattr(Image, "Resampling", Image)
        lanczos = resampling.LANCZOS if hasattr(resampling, "LANCZOS") else Image.LANCZOS
        lq_image = image.convert("RGB").resize((width, height), lanczos)
        lq_np = np.array(lq_image, dtype=np.float32)
        condition_cond = torch.from_numpy((lq_np / 127.5) - 1.0)
        condition_cond = condition_cond.permute(2, 0, 1).unsqueeze(0)
        condition_cond = condition_cond.to(self.device, dtype=torch.bfloat16)

        ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
        ci_pre = self.swinir(ci_01.to(self.device)).float().clamp(0.0, 1.0)
        condition_cond_ldr = ((ci_pre * 2.0 - 1.0) * control_weight).to(torch.bfloat16)

        reference_strength = max(0.0, float(reference_strength))
        image_proj: Optional[torch.Tensor] = None
        ip_scale = 0.0
        reference_used = False
        if reference_image is not None and reference_strength > 0.0:
            self._enable_ip_adapter()
            image_proj = self._encode_reference_image(reference_image)
            ip_scale = reference_strength
            reference_used = True
        else:
            if self._attn_state == "ip":
                self._disable_ip_adapter()

        if seed is None or seed < 0:
            seed = torch.Generator(device="cpu").seed()
        torch.manual_seed(seed)
        x = get_noise(
            1,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed,
        )

        inp_cond = prepare_with_embeddings(
            img=x,
            precomputed_txt=self.precomputed_txt,
            precomputed_vec=self.precomputed_vec,
        )

        siglip_size = getattr(getattr(self.siglip_model, "config", None), "image_size", 512)
        siglip_pixel_values = siglip_from_unit_tensor(ci_pre, size=(siglip_size, siglip_size))
        siglip_inputs = {"pixel_values": siglip_pixel_values.to(device=self.device, dtype=self.dtype)}
        siglip_feats = self.siglip_model(**siglip_inputs).last_hidden_state.to(dtype=self.dtype)
        image_embeds = self.redux_image_encoder(siglip_feats)["image_embeds"]

        txt = inp_cond["txt"].to(device=self.device, dtype=torch.bfloat16)
        txt_ids = inp_cond["txt_ids"].to(device=self.device, dtype=torch.bfloat16)
        vec = inp_cond["vec"].to(device=self.device, dtype=torch.bfloat16)
        img = inp_cond["img"].to(device=self.device, dtype=torch.bfloat16)
        img_ids = inp_cond["img_ids"].to(device=self.device, dtype=torch.bfloat16)

        siglip_txt = torch.cat([txt, image_embeds.to(dtype=torch.bfloat16)], dim=1)
        extra_ids = torch.zeros(
            (siglip_txt.shape[0], 1024, txt_ids.shape[-1]),
            device=txt_ids.device,
            dtype=torch.bfloat16,
        )
        siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1)

        timesteps = get_schedule(
            steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )

        guidance_eff = guidance * true_gs
        x = denoise_lucidflux(
            self.model,
            dual_condition_model=self.dual_condition_branch,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            siglip_txt=siglip_txt,
            siglip_txt_ids=siglip_txt_ids,
            vec=vec,
            timesteps=timesteps,
            guidance=guidance_eff,
            condition_cond_lq=condition_cond,
            condition_cond_ldr=condition_cond_ldr,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )

        x = unpack(x.float(), height, width)
        x = self.ae.decode(x)
        x1 = x.clamp(-1, 1)
        x1 = x1[-1]

        hq = wavelet_reconstruction((x1 + 1.0) / 2.0, ci_pre.squeeze(0))
        hq = hq.clamp(0, 1).cpu()
        output = (hq.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        restored = Image.fromarray(output)
        if restored.size != (orig_w, orig_h):
            restored = restored.resize((orig_w, orig_h), lanczos)
        prior_blend = max(0.0, min(1.0, prior_blend))
        if prior_blend > 0:
            blended = restored.copy()
            prior_img = Image.fromarray((ci_pre.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            if prior_img.size != blended.size:
                prior_img = prior_img.resize(blended.size, lanczos)
            restored = Image.blend(prior_img, blended, prior_blend)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        metrics = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_width": orig_w,
            "input_height": orig_h,
            "output_width": restored.size[0],
            "output_height": restored.size[1],
            "max_side": max_side,
            "steps": steps,
            "guidance": guidance,
            "control_weight": control_weight,
            "guidance_multiplier": true_gs,
            "prior_blend": prior_blend,
            "reference_strength": ip_scale,
            "reference_applied": int(reference_used),
            "seed": seed,
            "duration_sec": duration,
            "xformers_enabled": int(self.xformers_enabled),
        }
        self._log_inference(metrics)
        return restored, metrics

    def _log_inference(self, metrics: dict) -> None:
        line = ",".join(
            str(metrics[key])
            for key in [
                "timestamp",
                "input_width",
                "input_height",
                "output_width",
                "output_height",
                "max_side",
                "steps",
                "guidance",
                "control_weight",
                "guidance_multiplier",
                "prior_blend",
                "reference_strength",
                "reference_applied",
                "seed",
                "duration_sec",
                "xformers_enabled",
            ]
        )
        with self.log_path.open("a") as fh:
            fh.write(f"{line}\n")


def build_interface(engine: LucidFluxEngine) -> gr.Interface:
    """Build a minimal Gradio Interface around the LucidFlux engine."""

    def _predict(
        img,
        reference_img,
        step_count,
        guidance_scale,
        seed_value,
        max_side,
        control_weight,
        guidance_multiplier,
        prior_blend,
        reference_strength,
    ):
        if img is None:
            raise gr.Error("Please upload an input image.")
        image, metrics = engine.restore(
            img,
            max_side=int(max_side),
            steps=int(step_count),
            guidance=float(guidance_scale),
            seed=int(seed_value),
            true_gs=float(guidance_multiplier),
            control_weight=float(control_weight),
            prior_blend=float(prior_blend),
            reference_image=reference_img,
            reference_strength=float(reference_strength),
        )
        status = (
            f"⏱️ {metrics['duration_sec']:.2f}s · "
            f"{metrics['input_width']}×{metrics['input_height']} → {metrics['output_width']}×{metrics['output_height']} · "
            f"control {metrics['control_weight']:.2f} · guidance ×{metrics['guidance_multiplier']:.2f} · blend {metrics['prior_blend']:.2f}"
        )
        if metrics.get("reference_applied"):
            status += f" · ref ×{metrics['reference_strength']:.2f}"
        return image, status

    return gr.Interface(
        fn=_predict,
        inputs=[
            gr.Image(
                label="Low-quality input",
                type="pil",
                image_mode="RGB",
                height=512,
            ),
            gr.Image(
                label="Reference face (optional)",
                type="pil",
                image_mode="RGB",
                height=512,
            ),
            gr.Slider(10, 40, value=30, step=1, label="Sampling steps"),
            gr.Slider(2.0, 6.0, value=3.0, step=0.1, label="CFG guidance"),
            gr.Slider(
                minimum=-1,
                maximum=2_147_483_647,
                value=-1,
                step=1,
                label="Seed (-1 for random)",
            ),
            gr.Slider(
                minimum=256,
                maximum=2048,
                value=1280,
                step=16,
                label="Max long side (px)",
            ),
            gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=1.1,
                step=0.05,
                label="Control weight",
            ),
            gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=0.9,
                step=0.05,
                label="Guidance multiplier",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Blend with prior (higher = safer, lower = crisper)",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.85,
                step=0.05,
                label="Reference strength (0 = off)",
            ),
        ],
        outputs=[
            gr.Image(
                label="Restored output",
                type="pil",
                image_mode="RGB",
            ),
            gr.Markdown(label="Summary", value=""),
        ],
        title="LucidFlux Restoration",
        description="Upload a degraded photo to restore it with the LucidFlux diffusion transformer.",
        concurrency_limit=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch LucidFlux Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    parser.add_argument("--listen", default="0.0.0.0", help="Server host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860).")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights"),
        help="Path to LucidFlux weights directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (defaults to CUDA when available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_exports(args.weights / "env.sh")
    engine = LucidFluxEngine(args.weights.resolve(), device=args.device)
    demo = build_interface(engine)
    app = demo.launch(
        share=bool(args.share),
        server_name=args.listen,
        server_port=args.port,
        inline=False,
        show_api=False,
    )
    share_url = getattr(app, "share_url", None)
    local_url = getattr(app, "local_url", None)
    if local_url:
        print(f"[INFO] Local URL: {local_url}", flush=True)
    if share_url:
        print(f"[INFO] Share URL: {share_url}", flush=True)
    elif args.share:
        print("[WARN] Share URL requested but not available.", flush=True)


if __name__ == "__main__":
    main()
