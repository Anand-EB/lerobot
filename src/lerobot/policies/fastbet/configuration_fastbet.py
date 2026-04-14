#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import VQBeTSchedulerConfig


@PreTrainedConfig.register_subclass("fastbet")
@dataclass
class FASTBeTConfig(PreTrainedConfig):
    """Configuration class for FAST-BeT.

    Like VQ-BeT but replaces the Residual VQ-VAE action quantizer with a pretrained FAST tokenizer
    (DCT + BPE), eliminating the two-phase training. The GPT is trained end-to-end to predict
    FAST token IDs directly via cross-entropy.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        n_action_pred_token: Total number of current token and future tokens that FASTBeT predicts.
        action_chunk_size: Action chunk size of each action prediction token.
        fast_tokenizer_name: HuggingFace repo or local path for the pretrained FAST tokenizer.
        max_action_tokens: Number of FAST tokens produced per action chunk (tokenizer-dependent).
        fast_vocab_size: Vocabulary size of the FAST tokenizer (BPE vocab size).
        vision_backbone: Name of the torchvision resnet backbone.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        gpt_block_size: Max block size of minGPT.
        gpt_input_dim: GPT input dimension (also used for observation features).
        gpt_output_dim: GPT output dimension (input to the token prediction head).
        gpt_n_layer: Number of GPT layers.
        gpt_n_head: Number of GPT attention heads.
        gpt_hidden_dim: Size of GPT hidden dimensions.
        dropout: Dropout rate for GPT.
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    action_chunk_size: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # FAST tokenizer (pretrained, frozen during policy training)
    fast_tokenizer_name: str = "physical-intelligence/fast"
    max_action_tokens: int = 8
    fast_vocab_size: int = 2048

    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32

    # GPT
    gpt_block_size: int = 500
    gpt_input_dim: int = 512
    gpt_output_dim: int = 512
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    dropout: float = 0.1

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> VQBeTSchedulerConfig:
        return VQBeTSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_vqvae_training_steps=0,  # No Action Quantization pretraining phase
        )

    def validate_features(self) -> None:
        if not len(self.image_features) == 1:
            raise ValueError("You must provide only one image among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )
        
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, self.n_action_pred_token + self.action_chunk_size - 1))

    @property
    def reward_delta_indices(self) -> None:
        return None