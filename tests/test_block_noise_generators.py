# Copyright 2025 The Gigaworld Team and The HuggingFace Team. All rights reserved.
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

"""Regression tests for per-sample block-noise generators."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_REPO_ROOT = Path(__file__).parents[1]
_PIPELINE_SOURCES = (
    _REPO_ROOT / "gigaworld" / "pipelines" / "pipeline_gigaworld.py",
    _REPO_ROOT / "gigaworld" / "pipelines" / "pipeline_gigaworld_functrl.py",
    _REPO_ROOT / "gigaworld" / "pipelines" / "pipeline_gigaworld_functrl_wan22_5b.py",
    _REPO_ROOT / "gigaworld" / "diffusers_version" / "pipeline_gigaworld_diffusers.py",
)


def _load_sample_block_noise(source_path: Path, method_index: int = 0):
    """Load only the self-contained method, avoiding CUDA-only pipeline imports."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sample_block_noise"
    ]
    method = methods[method_index]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["sample_block_noise"]


def _pipeline(gamma: float):
    return SimpleNamespace(scheduler=SimpleNamespace(config=SimpleNamespace(gamma=gamma)))


def test_list_generators_are_consumed_in_batch_order():
    sample_block_noise = _load_sample_block_noise(_PIPELINE_SOURCES[0])
    batch_size, channel, num_frames, height, width = 2, 1, 2, 4, 4
    gamma = 0.25
    seeds = (17, 29)
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]

    actual = sample_block_noise(
        _pipeline(gamma),
        batch_size,
        channel,
        num_frames,
        height,
        width,
        patch_size=(1, 2, 2),
        device=torch.device("cpu"),
        generator=generators,
    )

    block_height, block_width = 2, 2
    block_size = block_height * block_width
    blocks_per_sample = channel * num_frames * (height // block_height) * (width // block_width)
    expected_generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    expected_z = torch.cat(
        [
            torch.randn(blocks_per_sample, block_size, generator=generator)
            for generator in expected_generators
        ],
        dim=0,
    )
    covariance = (
        torch.eye(block_size) * (1 + gamma)
        - torch.ones(block_size, block_size) * gamma
        + torch.eye(block_size) * 1e-8
    )
    expected = (expected_z @ torch.linalg.cholesky(covariance).T).reshape(
        batch_size,
        channel,
        num_frames,
        height // block_height,
        width // block_width,
        block_height,
        block_width,
    )
    expected = expected.permute(0, 1, 2, 3, 5, 4, 6).reshape(
        batch_size, channel, num_frames, height, width
    )

    torch.testing.assert_close(actual, expected)
    # A single-generator implementation would consume the second sample from
    # the first stream and fail this per-sample identity check.
    assert not torch.equal(actual[0], actual[1])


def test_generator_list_must_match_batch_size():
    sample_block_noise = _load_sample_block_noise(_PIPELINE_SOURCES[0])
    with pytest.raises(ValueError, match="one generator per batch item"):
        sample_block_noise(
            _pipeline(0.0),
            2,
            1,
            1,
            2,
            2,
            patch_size=(1, 1, 1),
            device=torch.device("cpu"),
            generator=[torch.Generator()],
        )


@pytest.mark.parametrize("source_path", _PIPELINE_SOURCES)
def test_all_pipeline_variants_keep_list_generator_support(source_path: Path):
    """The legacy and diffusers pipeline variants must share the same contract."""

    source = source_path.read_text(encoding="utf-8")
    expected_methods = 2 if source_path.name == "pipeline_gigaworld_diffusers.py" else 1
    assert source.count("elif isinstance(generator, list) and len(generator) != batch_size:") == expected_methods
    assert "generator = generator[0]" not in source
