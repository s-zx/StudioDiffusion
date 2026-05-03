"""IP-Adapter v2 SDXL wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from adapters.ip_adapter.layers_v2 import ImageProjModelV2, IPAttnProcessor2_0V2


class IPAdapterSDXLV2(nn.Module):
    """Wrapper that wires IP-Adapter v2 into an SDXL UNet."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        image_encoder_id: str = "openai/clip-vit-large-patch14-336",
        num_tokens: int = 16,
        adapter_scale: float = 1.0,
        proj_hidden_size: int | None = None,
        image_encoder_dtype: torch.dtype | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.image_encoder_id = image_encoder_id
        self.num_tokens = num_tokens
        self.adapter_scale = adapter_scale
        self.proj_hidden_size = proj_hidden_size

        image_encoder_kwargs: dict[str, object] = {"local_files_only": local_files_only}
        if image_encoder_dtype is not None:
            image_encoder_kwargs["torch_dtype"] = image_encoder_dtype

        # Frozen CLIP vision encoder. Do not train this component.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_id,
            **image_encoder_kwargs,
        )
        try:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                image_encoder_id,
                local_files_only=local_files_only,
            )
        except OSError:
            self.feature_extractor = None

        clip_embed_dim = self.image_encoder.config.projection_dim
        cross_attention_dim = self.unet.config.cross_attention_dim

        # Trainable projection from CLIP image embedding to IP tokens.
        self.image_proj_model = ImageProjModelV2(
            clip_embed_dim=clip_embed_dim,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
            hidden_size=proj_hidden_size,
        )

        self._inject_ip_processors()
        self.freeze_base_model()

    def _hidden_size_for_processor(self, processor_name: str) -> int:
        """Return the UNet hidden size for one attention processor name."""
        if processor_name.startswith("mid_block"):
            return self.unet.config.block_out_channels[-1]
        elif processor_name.startswith("down_blocks."):
            block_i = int(processor_name.split(".")[1])
            return self.unet.config.block_out_channels[block_i]
        elif processor_name.startswith("up_blocks."):
            block_i = int(processor_name.split(".")[1])
            return self.unet.config.block_out_channels[-1 - block_i]
        raise ValueError(f"Unknown attention processor name: {processor_name}")

    def _inject_ip_processors(self) -> None:
        """Replace cross-attention processors with IPAttnProcessor2_0V2."""
        from diffusers.models.attention_processor import AttnProcessor2_0
        processors = {}
        for processor in self.unet.attn_processors:
            if processor.endswith("attn1.processor"):
                processors[processor] = AttnProcessor2_0()
            else:
                hidden_size = self._hidden_size_for_processor(processor)
                cross_attention_dim = self.unet.config.cross_attention_dim
                num_tokens = self.num_tokens
                scale = self.adapter_scale
                processors[processor]= IPAttnProcessor2_0V2(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=num_tokens,
                    scale=scale,
                )
        self.unet.set_attn_processor(processors)
            

    def freeze_base_model(self) -> None:
        """Freeze base SDXL + CLIP and unfreeze only adapter parameters."""
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.image_proj_model.parameters():
            param.requires_grad = True
        for processor in self.ip_attention_processors().values():
            for param in processor.parameters():
                param.requires_grad = True

    def ip_attention_processors(self) -> dict[str, IPAttnProcessor2_0V2]:
        """Return only injected IP-Adapter processors from the UNet."""
        processors = {}
        for name, processor in self.unet.attn_processors.items():
            if isinstance(processor, IPAttnProcessor2_0V2):
                processors[name] = processor
        return processors
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the parameters the optimizer should update."""
        return [param for param in self.parameters() if param.requires_grad]

    def trainable_parameter_summary(self) -> str:
        """Return a printable trainable/total parameter summary."""
        return f"Trainable parameters: {sum(p.numel() for p in self.trainable_parameters())} / {sum(p.numel() for p in self.parameters())}"

    def encode_image(self, clip_pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode CLIP pixels into conditional and unconditional IP tokens."""
        image_encoder_param = next(self.image_encoder.parameters())
        clip_pixel_values = clip_pixel_values.to(
        device=image_encoder_param.device,
        dtype=image_encoder_param.dtype,
        )
        with torch.no_grad():
            image_embeds = self.image_encoder(pixel_values=clip_pixel_values).image_embeds
            proj_param = next(self.image_proj_model.parameters())
            image_embeds=image_embeds.to(device=proj_param.device, dtype=proj_param.dtype)
        projected_image_embeds = self.image_proj_model(image_embeds)
        uncond_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return (projected_image_embeds, uncond_embeds)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor],
        clip_pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise with SDXL conditioned on text plus IP image tokens."""
        clip_pixel_values = clip_pixel_values.to(next(self.parameters()).device)
        image_tokens, _ = self.encode_image(clip_pixel_values)
        cross_attention_kwargs = {"ip_hidden_states": image_tokens}
        unet_output = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return unet_output.sample

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save only the adapter weights and small v2 metadata."""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.image_proj_model.state_dict(), Path(save_directory) / "image_proj_model.pt")
        ip_processor_state_dict = {}
        for name, processor in self.ip_attention_processors().items():
            ip_processor_state_dict[name] = processor.state_dict()
        torch.save(ip_processor_state_dict, Path(save_directory) / "ip_attn_processors.pt")
        json_dict = {
            "image_encoder_id": self.image_encoder_id,
            "num_tokens": self.num_tokens,
            "adapter_scale": self.adapter_scale,
            "proj_hidden_size": self.proj_hidden_size,
        }   
        (Path(save_directory) / "ip_adapter_v2_config.json").write_text(json.dumps(json_dict))

    @classmethod
    def load_pretrained(
        cls,
        unet: UNet2DConditionModel,
        load_directory: str | Path,
        **kwargs,
    ) -> "IPAdapterSDXLV2":
        """Create a v2 adapter and load adapter-only weights from disk."""
        config_path = Path(load_directory) / "ip_adapter_v2_config.json"
        if config_path.exists():
            json_dict = json.loads(config_path.read_text())
            kwargs = {**json_dict, **kwargs}
        adapter = cls(unet=unet, **kwargs)
        image_proj_model_path = Path(load_directory) / "image_proj_model.pt"
        adapter.image_proj_model.load_state_dict(torch.load(image_proj_model_path, map_location="cpu"))
        ip_attn_processors_path = Path(load_directory) / "ip_attn_processors.pt"
        ip_processor_state_dict = torch.load(ip_attn_processors_path, map_location="cpu")
        for name, processor in adapter.ip_attention_processors().items():
            processor.load_state_dict(ip_processor_state_dict[name])
        return adapter
