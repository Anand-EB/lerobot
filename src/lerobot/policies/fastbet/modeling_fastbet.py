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

import logging
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from transformers import AutoProcessor

from lerobot.policies.fastbet.configuration_fastbet import FASTBeTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_output_shape, populate_queues
from lerobot.policies.vqbet.vqbet_utils import GPT
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# ruff: noqa: N806

logger = logging.getLogger(__name__)


class FASTBeTPolicy(PreTrainedPolicy):
    """
    VQ-BeT Policy as per "Behavior Generation with Latent Actions"
    """

    config_class = FASTBeTConfig
    name = "fastbet"

    def __init__(
        self,
        config: FASTBeTConfig | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        tokenizer = AutoProcessor.from_pretrained(config.fast_tokenizer_name, trust_remote_code=True)
        self.vqbet = FASTBeTModel(config, tokenizer)

        self.reset()

    def get_optim_params(self) -> dict:
        decay_params, no_decay_params = self.vqbet.policy.configure_parameters()
        decay_params = (
            decay_params
            + list(self.vqbet.rgb_encoder.parameters())
            + list(self.vqbet.state_projector.parameters())
            + list(self.vqbet.rgb_feature_projector.parameters())
            + [self.vqbet.action_token]
            + list(self.vqbet.action_head.lm_head.parameters())
        )

        return [
            {
                "params": decay_params,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        queues are populated during rollout of the policy, they contain the n latest observations and actions
        """
        self._queues = {
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.action_chunk_size),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.vqbet(batch, rollout=True)[:, : self.config.action_chunk_size]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        # NOTE: It's important that this happens after stacking the images into a single key.
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # since the data in the action queue's dimension is (action_chunk_size, batch_size, action_dim), we transpose the action and fill the queue
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        _, loss_dict = self.vqbet(batch, rollout=False)
        loss = loss_dict.pop("loss")

        return loss, loss_dict


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class FASTBeTModel(nn.Module):
    """FASTBeT: The underlying neural network for FASTBeT

    Note: In this code we use the terms `rgb_encoder`, 'policy', `action_head`. The meanings are as follows.
        - The `rgb_encoder` process rgb-style image observations to one-dimensional embedding vectors
        - A `policy` is a minGPT architecture, that takes observation sequences and action query tokens to generate `features`.
        - These `features` pass through the action head, which passes through the code prediction, offset prediction head,
        and finally generates a prediction for the action chunks.

        -------------------------------** legend **-------------------------------
        │   n = n_obs_steps, p = n_action_pred_token, c = action_chunk_size)   │
        │   o_{t} : visual observation at timestep {t}                           │
        │   s_{t} : state observation at timestep {t}                            │
        │   a_{t} : action at timestep {t}                                       │
        │   A_Q : action_query_token                                             │
        --------------------------------------------------------------------------


        Training Phase 1. Discretize action using Residual VQ (for config.n_vqvae_training_steps steps)


        ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
        │                 │            │                 │            │                 │
        │   RVQ encoder   │    ─►      │     Residual    │    ─►      │   RVQ Decoder   │
        │ (a_{t}~a_{t+p}) │            │  Code Quantizer │            │                 │
        │                 │            │                 │            │                 │
        └─────────────────┘            └─────────────────┘            └─────────────────┘

        Training Phase 2.

          timestep {t-n+1}   timestep {t-n+2}                timestep {t}
            ┌─────┴─────┐     ┌─────┴─────┐                 ┌─────┴─────┐

        o_{t-n+1}         o_{t-n+2}           ...         o_{t}
            │                 │                             │
            │ s_{t-n+1}       │ s_{t-n+2}         ...       │   s_{t}           p
            │     │           │     │                       │     │     ┌───────┴───────┐
            │     │    A_Q    │     │    A_Q          ...   │     │    A_Q     ...     A_Q
            │     │     │     │     │     │                 │     │     │               │
        ┌───▼─────▼─────▼─────▼─────▼─────▼─────────────────▼─────▼─────▼───────────────▼───┐
        │                                                                                   │
        │                                       GPT                                         │       =>    policy
        │                                                                                   │
        └───────────────▼─────────────────▼─────────────────────────────▼───────────────▼───┘
                        │                 │                             │               │
                    ┌───┴───┐         ┌───┴───┐                     ┌───┴───┐       ┌───┴───┐
                  code    offset    code    offset                code    offset  code    offset
                    ▼       │         ▼       │                     ▼       │       ▼       │       =>    action_head
               RVQ Decoder  │    RVQ Decoder  │                RVQ Decoder  │  RVQ Decoder  │
                    └── + ──┘         └── + ──┘                     └── + ──┘       └── + ──┘
                        ▼                 ▼                             ▼               ▼
                   action chunk      action chunk                  action chunk     action chunk
                    a_{t-n+1} ~       a_{t-n+2} ~                   a_{t} ~     ...  a_{t+p-1} ~
                     a_{t-n+c}         a_{t-n+c+1}                   a_{t+c-1}        a_{t+p+c-1}

                                                                        ▼
                                                      ONLY this chunk is used in rollout!
    """

    def __init__(self, config: FASTBeTConfig, tokenizer):
        super().__init__()
        self.config = config

        self.rgb_encoder = FASTBeTRgbEncoder(config)
        self.num_images = len(self.config.image_features)
        # This action query token is used as a prompt for querying action chunks. Please refer to "A_Q" in the image above.
        # Note: During the forward pass, this token is repeated as many times as needed. The authors also experimented with initializing the necessary number of tokens independently and observed inferior results.
        self.action_token = nn.Parameter(torch.randn(1, 1, self.config.gpt_input_dim))

        # To input state and observation features into GPT layers, we first project the features to fit the shape of input size of GPT.
        self.state_projector = MLP(
            config.robot_state_feature.shape[0], hidden_channels=[self.config.gpt_input_dim]
        )
        self.rgb_feature_projector = MLP(
            self.rgb_encoder.feature_dim, hidden_channels=[self.config.gpt_input_dim]
        )

        # GPT part of FASTBeT
        self.policy = GPT(config)
        self.action_head = FASTActionHead(config, tokenizer)

        # Action tokens for: each observation step, the current action token, and all future action tokens.
        num_tokens = self.config.n_action_pred_token + self.config.n_obs_steps - 1
        self.register_buffer(
            "select_target_actions_indices",
            torch.row_stack([torch.arange(i, i + self.config.action_chunk_size) for i in range(num_tokens)]),
        )

    def forward(self, batch: dict[str, Tensor], rollout: bool) -> tuple[dict, dict]:
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, OBS_IMAGES})
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(
            img_features, "(b s n) ... -> b s n ...", b=batch_size, s=n_obs_steps, n=self.num_images
        )

        # Arrange prior and current observation step tokens as shown in the class docstring.
        # First project features to token dimension.
        rgb_tokens = self.rgb_feature_projector(
            img_features
        )  # (batch, obs_step, number of different cameras, projection dims)
        input_tokens = [rgb_tokens[:, :, i] for i in range(rgb_tokens.size(2))]
        input_tokens.append(self.state_projector(batch[OBS_STATE]))  # (batch, obs_step, projection dims)
        input_tokens.append(einops.repeat(self.action_token, "1 1 d -> b n d", b=batch_size, n=n_obs_steps))
        # Interleave tokens by stacking and rearranging.
        input_tokens = torch.stack(input_tokens, dim=2)
        input_tokens = einops.rearrange(input_tokens, "b n t d -> b (n t) d")

        len_additional_action_token = self.config.n_action_pred_token - 1
        future_action_tokens = self.action_token.repeat(batch_size, len_additional_action_token, 1)

        # add additional action query tokens for predicting future action chunks
        input_tokens = torch.cat([input_tokens, future_action_tokens], dim=1)

        # get action features (pass through GPT)
        features = self.policy(input_tokens)
        # len(self.config.input_features) is the number of different observation modes.
        # this line gets the index of action prompt tokens.
        historical_act_pred_index = np.arange(0, n_obs_steps) * (len(self.config.input_features) + 1) + len(
            self.config.input_features
        )

        # only extract the output tokens at the position of action query:
        # Behavior Transformer (BeT), and VQ-BeT are both sequence-to-sequence prediction models,
        # mapping sequential observation to sequential action (please refer to section 2.2 in BeT paper https://huggingface.co/papers/2206.11251).
        # Thus, it predicts a historical action sequence, in addition to current and future actions (predicting future actions : optional).
        if len_additional_action_token > 0:
            features = torch.cat(
                [features[:, historical_act_pred_index], features[:, -len_additional_action_token:]], dim=1
            )
        else:
            features = features[:, historical_act_pred_index]
        # pass through action head: (B, T, max_action_tokens, vocab_size)
        action_head_output = self.action_head(features)
        if rollout:
            token_ids = action_head_output[:, n_obs_steps - 1, :, :].argmax(-1)  # (B, max_action_tokens)
            return self.action_head.decode_actions(token_ids)  # (B, action_chunk_size, action_dim)
        else:
            target_actions = batch[ACTION][:, self.select_target_actions_indices]
            loss_dict = self.action_head.loss_fn(action_head_output, target_actions)
            return action_head_output, loss_dict


class FASTActionHead(nn.Module):
    """Replaces VQBeTHead: predicts FAST token IDs via a linear head + cross-entropy loss.

    Training: tokenize ground-truth action chunks with the frozen FAST tokenizer,
              compute cross-entropy between GPT logits and token IDs.
    Inference: argmax over logits -> token IDs -> FAST decode -> continuous actions.
    """

    def __init__(self, config: FASTBeTConfig, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer  # frozen, not a nn.Module
        self.max_action_tokens = config.max_action_tokens
        self.vocab_size = config.fast_vocab_size
        self.action_chunk_size = config.action_chunk_size
        action_feature = next(iter(config.output_features.values()))
        self.action_dim = action_feature.shape[-1]
        self.lm_head = nn.Linear(config.gpt_output_dim, self.max_action_tokens * self.vocab_size)

    def forward(self, features: Tensor) -> Tensor:
        # features: (B, T, gpt_output_dim) -> (B, T, max_action_tokens, vocab_size)
        N, T, _ = features.shape
        logits = self.lm_head(features)
        return logits.reshape(N, T, self.max_action_tokens, self.vocab_size)

    def tokenize_actions(self, action_chunks: Tensor) -> Tensor:
        """Tokenize a batch of action chunks with the frozen FAST tokenizer.
        Args:
            action_chunks: (B, action_chunk_size, action_dim)
        Returns:
            (B, max_action_tokens) long tensor on same device
        """
        token_ids = self.tokenizer(action_chunks.cpu())  # list of variable-length lists
        lengths = [len(t) for t in token_ids]
        logger.debug("FAST token lengths: min=%d max=%d mean=%.1f (max_action_tokens=%d)",
                    min(lengths), max(lengths), sum(lengths) / len(lengths), self.max_action_tokens)
        padded = [(t + [-1] * self.max_action_tokens)[: self.max_action_tokens] for t in token_ids]
        return torch.tensor(padded, dtype=torch.long).to(action_chunks.device)

    def decode_actions(self, token_ids: Tensor) -> Tensor:
        """Decode FAST token IDs back to continuous actions.

        Uses a greedy prefix search to find the shortest token prefix whose BPE-decoded
        string has exactly action_chunk_size * action_dim characters. This avoids passing
        untrained padding positions (which have no gradient during training) to the decoder,
        which would corrupt the BPE string length and cause shape mismatches.

        Args:
            token_ids: (B, max_action_tokens) int tensor
        Returns:
            (B, action_chunk_size, action_dim) float tensor
        """
        target_chars = self.action_chunk_size * self.action_dim
        valid_token_seqs = []
        for tokens in token_ids.tolist():
            for n in range(1, len(tokens) + 1):
                decoded = self.tokenizer.bpe_tokenizer.decode(tokens[:n], skip_special_tokens=True)
                if len(decoded) == target_chars:
                    valid_token_seqs.append(tokens[:n])
                    break
            else:
                valid_token_seqs.append(tokens)  # fallback: use all tokens
        actions = self.tokenizer.decode(
            valid_token_seqs,
            time_horizon=self.action_chunk_size,
            action_dim=self.action_dim,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        return actions.to(token_ids.device)

    def loss_fn(self, logits: Tensor, target_actions: Tensor) -> dict:
        """Cross-entropy over FAST token predictions.
        Args:
            logits: (B, T, max_action_tokens, vocab_size)
            target_actions: (B, T, action_chunk_size, action_dim)
        """
        B, T = logits.shape[:2]
        target_chunks = target_actions.reshape(B * T, self.config.action_chunk_size, -1)
        target_token_ids = self.tokenize_actions(target_chunks)  # (B*T, max_action_tokens)

        logits_flat = logits.reshape(B * T, self.max_action_tokens, self.vocab_size)
        # Mask out the padding tokens so that they do not contribute to the loss.
        # This workaround is because FAST uses 0 as a valid token ID, so we use -1 for padding and then
        # mask it out.
        mask = (target_token_ids != -1).float()  # (B*T, max_action_tokens)

        loss_per_token = F.cross_entropy(
            logits_flat.reshape(-1, self.vocab_size),
            target_token_ids.clamp(min=0).reshape(-1),
            reduction="none",
        ).reshape(B * T, self.max_action_tokens)
        loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
        return {"loss": loss, "token_ce_loss": loss.detach().item()}


class FASTBeTRgbEncoder(nn.Module):
    """Encode an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.

    Same with DiffusionRgbEncoder from modeling_diffusion.py
    """

    def __init__(self, config: FASTBeTConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module





class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1]))

        super().__init__(*layers)
