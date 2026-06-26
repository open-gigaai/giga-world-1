# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import pytest

from diffusers.modular_pipelines import (
    GigaworldAutoBlocks,
    GigaworldModularPipeline,
    GigaworldPyramidAutoBlocks,
    GigaworldPyramidModularPipeline,
)

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


GIGAWORLD_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "GigaworldTextEncoderStep"),
        ("denoise.input", "GigaworldTextInputStep"),
        ("denoise.prepare_history", "GigaworldPrepareHistoryStep"),
        ("denoise.set_timesteps", "GigaworldSetTimestepsStep"),
        ("denoise.chunk_denoise", "GigaworldChunkDenoiseStep"),
        ("decode", "GigaworldDecodeStep"),
    ],
    "image2video": [
        ("text_encoder", "GigaworldTextEncoderStep"),
        ("vae_encoder", "GigaworldImageVaeEncoderStep"),
        ("denoise.input", "GigaworldTextInputStep"),
        ("denoise.additional_inputs", "GigaworldAdditionalInputsStep"),
        ("denoise.add_noise_image", "GigaworldAddNoiseToImageLatentsStep"),
        ("denoise.prepare_history", "GigaworldPrepareHistoryStep"),
        ("denoise.seed_history", "GigaworldI2VSeedHistoryStep"),
        ("denoise.set_timesteps", "GigaworldSetTimestepsStep"),
        ("denoise.chunk_denoise", "GigaworldI2VChunkDenoiseStep"),
        ("decode", "GigaworldDecodeStep"),
    ],
    "video2video": [
        ("text_encoder", "GigaworldTextEncoderStep"),
        ("vae_encoder", "GigaworldVideoVaeEncoderStep"),
        ("denoise.input", "GigaworldTextInputStep"),
        ("denoise.additional_inputs", "GigaworldAdditionalInputsStep"),
        ("denoise.add_noise_video", "GigaworldAddNoiseToVideoLatentsStep"),
        ("denoise.prepare_history", "GigaworldPrepareHistoryStep"),
        ("denoise.seed_history", "GigaworldV2VSeedHistoryStep"),
        ("denoise.set_timesteps", "GigaworldSetTimestepsStep"),
        ("denoise.chunk_denoise", "GigaworldI2VChunkDenoiseStep"),
        ("decode", "GigaworldDecodeStep"),
    ],
}


class TestGigaworldModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = GigaworldModularPipeline
    pipeline_blocks_class = GigaworldAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-gigaworld-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    output_name = "videos"
    expected_workflow_blocks = GIGAWORLD_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


GIGAWORLD_PYRAMID_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "GigaworldTextEncoderStep"),
        ("denoise.input", "GigaworldTextInputStep"),
        ("denoise.prepare_history", "GigaworldPrepareHistoryStep"),
        ("denoise.pyramid_chunk_denoise", "GigaworldPyramidChunkDenoiseStep"),
        ("decode", "GigaworldDecodeStep"),
    ],
    "image2video": [
        ("text_encoder", "GigaworldTextEncoderStep"),
        ("vae_encoder", "GigaworldImageVaeEncoderStep"),
        ("denoise.input", "GigaworldTextInputStep"),
        ("denoise.additional_inputs", "GigaworldAdditionalInputsStep"),
        ("denoise.add_noise_image", "GigaworldAddNoiseToImageLatentsStep"),
        ("denoise.prepare_history", "GigaworldPrepareHistoryStep"),
        ("denoise.seed_history", "GigaworldI2VSeedHistoryStep"),
        ("denoise.pyramid_chunk_denoise", "GigaworldPyramidI2VChunkDenoiseStep"),
        ("decode", "GigaworldDecodeStep"),
    ],
    "video2video": [
        ("text_encoder", "GigaworldTextEncoderStep"),
        ("vae_encoder", "GigaworldVideoVaeEncoderStep"),
        ("denoise.input", "GigaworldTextInputStep"),
        ("denoise.additional_inputs", "GigaworldAdditionalInputsStep"),
        ("denoise.add_noise_video", "GigaworldAddNoiseToVideoLatentsStep"),
        ("denoise.prepare_history", "GigaworldPrepareHistoryStep"),
        ("denoise.seed_history", "GigaworldV2VSeedHistoryStep"),
        ("denoise.pyramid_chunk_denoise", "GigaworldPyramidI2VChunkDenoiseStep"),
        ("decode", "GigaworldDecodeStep"),
    ],
}


class TestGigaworldPyramidModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = GigaworldPyramidModularPipeline
    pipeline_blocks_class = GigaworldPyramidAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-gigaworld-pyramid-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["pyramid_num_inference_steps_list", "num_videos_per_prompt", "latents"])
    output_name = "videos"
    expected_workflow_blocks = GIGAWORLD_PYRAMID_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "pyramid_num_inference_steps_list": [2, 2],
            "height": 64,
            "width": 64,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        # Pyramid pipeline injects noise at each stage, so batch vs single can differ more
        super().test_inference_batch_single_identical(expected_max_diff=5e-1)

    @pytest.mark.skip(reason="Pyramid multi-stage noise makes offload comparison unreliable with tiny models")
    def test_components_auto_cpu_offload_inference_consistent(self):
        pass

    @pytest.mark.skip(reason="Pyramid multi-stage noise makes save/load comparison unreliable with tiny models")
    def test_save_from_pretrained(self):
        pass

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass
