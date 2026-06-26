# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InputParam, InsertableDict, OutputParam
from .before_denoise import (
    GigaworldAdditionalInputsStep,
    GigaworldAddNoiseToImageLatentsStep,
    GigaworldAddNoiseToVideoLatentsStep,
    GigaworldI2VSeedHistoryStep,
    GigaworldPrepareHistoryStep,
    GigaworldSetTimestepsStep,
    GigaworldTextInputStep,
    GigaworldV2VSeedHistoryStep,
)
from .decoders import GigaworldDecodeStep
from .denoise import GigaworldChunkDenoiseStep, GigaworldI2VChunkDenoiseStep
from .encoders import GigaworldImageVaeEncoderStep, GigaworldTextEncoderStep, GigaworldVideoVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. Vae Encoder
# ====================


# auto_docstring
class GigaworldAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Encoder step that encodes video or image inputs. This is an auto pipeline block.
       - `GigaworldVideoVaeEncoderStep` (video_encoder) is used when `video` is provided.
       - `GigaworldImageVaeEncoderStep` (image_encoder) is used when `image` is provided.
       - If neither is provided, step will be skipped.

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          video (`None`, *optional*):
              Input video for video-to-video generation
          height (`int`, *optional*, defaults to 384):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 640):
              The width in pixels of the generated image.
          num_latent_frames_per_chunk (`int`, *optional*, defaults to 9):
              Number of latent frames per temporal chunk.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.

      Outputs:
          image_latents (`Tensor`):
              The latent representation of the input image.
          video_latents (`Tensor`):
              Encoded video latents (chunked)
          fake_image_latents (`Tensor`):
              Fake image latents for history seeding
    """

    block_classes = [GigaworldVideoVaeEncoderStep, GigaworldImageVaeEncoderStep]
    block_names = ["video_encoder", "image_encoder"]
    block_trigger_inputs = ["video", "image"]

    @property
    def description(self):
        return (
            "Encoder step that encodes video or image inputs. This is an auto pipeline block.\n"
            " - `GigaworldVideoVaeEncoderStep` (video_encoder) is used when `video` is provided.\n"
            " - `GigaworldImageVaeEncoderStep` (image_encoder) is used when `image` is provided.\n"
            " - If neither is provided, step will be skipped."
        )


# ====================
# 2. DENOISE
# ====================


# DENOISE (T2V)
# auto_docstring
class GigaworldCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block that takes encoded conditions and runs the chunk-based denoising process.

      Components:
          transformer (`GigaworldTransformer3DModel`) scheduler (`GigaworldScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          height (`int`, *optional*, defaults to 384):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 640):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 132):
              Total number of video frames to generate.
          num_latent_frames_per_chunk (`int`, *optional*, defaults to 9):
              Number of latent frames per temporal chunk.
          history_sizes (`list`, *optional*, defaults to [16, 2, 1]):
              Sizes of long/mid/short history buffers for temporal context.
          keep_first_frame (`bool`, *optional*, defaults to True):
              Whether to keep the first frame as a prefix in history.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latent_chunks (`list`):
              List of per-chunk denoised latent tensors
    """

    model_name = "gigaworld"
    block_classes = [
        GigaworldTextInputStep,
        GigaworldPrepareHistoryStep,
        GigaworldSetTimestepsStep,
        GigaworldChunkDenoiseStep,
    ]
    block_names = ["input", "prepare_history", "set_timesteps", "chunk_denoise"]

    @property
    def description(self):
        return "Denoise block that takes encoded conditions and runs the chunk-based denoising process."

    @property
    def outputs(self):
        return [OutputParam("latent_chunks", type_hint=list, description="List of per-chunk denoised latent tensors")]


# DENOISE (I2V)
# auto_docstring
class GigaworldI2VCoreDenoiseStep(SequentialPipelineBlocks):
    """
    I2V denoise block that seeds history with image latents and uses I2V-aware chunk preparation.

      Components:
          transformer (`GigaworldTransformer3DModel`) scheduler (`GigaworldScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          fake_image_latents (`Tensor`, *optional*):
              Fake image latents used as history seed for I2V generation.
          image_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for image latent noise.
          image_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for image latent noise.
          video_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for video/fake-image latent noise.
          video_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for video/fake-image latent noise.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_frames (`int`, *optional*, defaults to 132):
              Total number of video frames to generate.
          num_latent_frames_per_chunk (`int`, *optional*, defaults to 9):
              Number of latent frames per temporal chunk.
          history_sizes (`list`, *optional*, defaults to [16, 2, 1]):
              Sizes of long/mid/short history buffers for temporal context.
          keep_first_frame (`bool`, *optional*, defaults to True):
              Whether to keep the first frame as a prefix in history.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latent_chunks (`list`):
              List of per-chunk denoised latent tensors
    """

    model_name = "gigaworld"
    block_classes = [
        GigaworldTextInputStep,
        GigaworldAdditionalInputsStep(
            image_latent_inputs=[InputParam.template("image_latents")],
            additional_batch_inputs=[
                InputParam(
                    "fake_image_latents",
                    type_hint=torch.Tensor,
                    description="Fake image latents used as history seed for I2V generation.",
                ),
            ],
        ),
        GigaworldAddNoiseToImageLatentsStep,
        GigaworldPrepareHistoryStep,
        GigaworldI2VSeedHistoryStep,
        GigaworldSetTimestepsStep,
        GigaworldI2VChunkDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "add_noise_image",
        "prepare_history",
        "seed_history",
        "set_timesteps",
        "chunk_denoise",
    ]

    @property
    def description(self):
        return "I2V denoise block that seeds history with image latents and uses I2V-aware chunk preparation."

    @property
    def outputs(self):
        return [OutputParam("latent_chunks", type_hint=list, description="List of per-chunk denoised latent tensors")]


# DENOISE (V2V)
# auto_docstring
class GigaworldV2VCoreDenoiseStep(SequentialPipelineBlocks):
    """
    V2V denoise block that seeds history with video latents and uses I2V-aware chunk preparation.

      Components:
          transformer (`GigaworldTransformer3DModel`) scheduler (`GigaworldScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          video_latents (`Tensor`, *optional*):
              Encoded video latents for V2V generation.
          num_latent_frames_per_chunk (`int`, *optional*, defaults to 9):
              Number of latent frames per temporal chunk.
          image_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for image latent noise.
          image_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for image latent noise.
          video_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for video latent noise.
          video_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for video latent noise.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_frames (`int`, *optional*, defaults to 132):
              Total number of video frames to generate.
          history_sizes (`list`, *optional*, defaults to [16, 2, 1]):
              Sizes of long/mid/short history buffers for temporal context.
          keep_first_frame (`bool`, *optional*, defaults to True):
              Whether to keep the first frame as a prefix in history.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latent_chunks (`list`):
              List of per-chunk denoised latent tensors
    """

    model_name = "gigaworld"
    block_classes = [
        GigaworldTextInputStep,
        GigaworldAdditionalInputsStep(
            image_latent_inputs=[InputParam.template("image_latents")],
            additional_batch_inputs=[
                InputParam(
                    "video_latents", type_hint=torch.Tensor, description="Encoded video latents for V2V generation."
                ),
            ],
        ),
        GigaworldAddNoiseToVideoLatentsStep,
        GigaworldPrepareHistoryStep,
        GigaworldV2VSeedHistoryStep,
        GigaworldSetTimestepsStep,
        GigaworldI2VChunkDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "add_noise_video",
        "prepare_history",
        "seed_history",
        "set_timesteps",
        "chunk_denoise",
    ]

    @property
    def description(self):
        return "V2V denoise block that seeds history with video latents and uses I2V-aware chunk preparation."

    @property
    def outputs(self):
        return [OutputParam("latent_chunks", type_hint=list, description="List of per-chunk denoised latent tensors")]


# AUTO DENOISE
# auto_docstring
class GigaworldAutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Core denoise step that selects the appropriate denoising block.
       - `GigaworldV2VCoreDenoiseStep` (video2video) for video-to-video tasks.
       - `GigaworldI2VCoreDenoiseStep` (image2video) for image-to-video tasks.
       - `GigaworldCoreDenoiseStep` (text2video) for text-to-video tasks.

      Components:
          transformer (`GigaworldTransformer3DModel`) scheduler (`GigaworldScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          video_latents (`Tensor`, *optional*):
              Encoded video latents for V2V generation.
          num_latent_frames_per_chunk (`int`, *optional*, defaults to 9):
              Number of latent frames per temporal chunk.
          image_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for image latent noise.
          image_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for image latent noise.
          video_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for video latent noise.
          video_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for video latent noise.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_frames (`int`, *optional*, defaults to 132):
              Total number of video frames to generate.
          history_sizes (`list`):
              Sizes of long/mid/short history buffers for temporal context.
          keep_first_frame (`bool`, *optional*, defaults to True):
              Whether to keep the first frame as a prefix in history.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`):
              Custom sigmas for the denoising process.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          fake_image_latents (`Tensor`, *optional*):
              Fake image latents used as history seed for I2V generation.
          height (`int`, *optional*, defaults to 384):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 640):
              The width in pixels of the generated image.

      Outputs:
          latent_chunks (`list`):
              List of per-chunk denoised latent tensors
    """

    block_classes = [GigaworldV2VCoreDenoiseStep, GigaworldI2VCoreDenoiseStep, GigaworldCoreDenoiseStep]
    block_names = ["video2video", "image2video", "text2video"]
    block_trigger_inputs = ["video_latents", "fake_image_latents"]
    default_block_name = "text2video"

    def select_block(self, video_latents=None, fake_image_latents=None):
        if video_latents is not None:
            return "video2video"
        elif fake_image_latents is not None:
            return "image2video"
        return None

    @property
    def description(self):
        return (
            "Core denoise step that selects the appropriate denoising block.\n"
            " - `GigaworldV2VCoreDenoiseStep` (video2video) for video-to-video tasks.\n"
            " - `GigaworldI2VCoreDenoiseStep` (image2video) for image-to-video tasks.\n"
            " - `GigaworldCoreDenoiseStep` (text2video) for text-to-video tasks."
        )


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", GigaworldTextEncoderStep()),
        ("vae_encoder", GigaworldAutoVaeEncoderStep()),
        ("denoise", GigaworldAutoCoreDenoiseStep()),
        ("decode", GigaworldDecodeStep()),
    ]
)

# ====================
# 3. Auto Blocks
# ====================


# auto_docstring
class GigaworldAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-video, image-to-video, and video-to-video tasks using Gigaworld.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `prompt`, `image`
        - `video2video`: requires `prompt`, `video`

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLWan`) video_processor (`VideoProcessor`) transformer (`GigaworldTransformer3DModel`) scheduler
          (`GigaworldScheduler`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.
          video (`None`, *optional*):
              Input video for video-to-video generation
          height (`int`, *optional*, defaults to 384):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 640):
              The width in pixels of the generated image.
          num_latent_frames_per_chunk (`int`, *optional*, defaults to 9):
              Number of latent frames per temporal chunk.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos to generate per prompt.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          video_latents (`Tensor`, *optional*):
              Encoded video latents for V2V generation.
          image_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for image latent noise.
          image_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for image latent noise.
          video_noise_sigma_min (`float`, *optional*, defaults to 0.111):
              Minimum sigma for video latent noise.
          video_noise_sigma_max (`float`, *optional*, defaults to 0.135):
              Maximum sigma for video latent noise.
          num_frames (`int`, *optional*, defaults to 132):
              Total number of video frames to generate.
          history_sizes (`list`):
              Sizes of long/mid/short history buffers for temporal context.
          keep_first_frame (`bool`, *optional*, defaults to True):
              Whether to keep the first frame as a prefix in history.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`):
              Custom sigmas for the denoising process.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          fake_image_latents (`Tensor`, *optional*):
              Fake image latents used as history seed for I2V generation.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "gigaworld"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"prompt": True, "image": True},
        "video2video": {"prompt": True, "video": True},
    }

    @property
    def description(self):
        return "Auto Modular pipeline for text-to-video, image-to-video, and video-to-video tasks using Gigaworld."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
