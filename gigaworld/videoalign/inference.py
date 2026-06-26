import json
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections.abc import Mapping

import torch

from .data import DataConfig
from .prompt_template import build_prompt
from .train_reward import create_model_and_processor
from .utils import ModelConfig, PEFTLoraConfig, TrainingConfig, load_model_from_checkpoint
from .vision_process import process_video_tensor, process_vision_info


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return (
        config_dict["data_config"],
        None,
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict["inference_config"] if "inference_config" in config_dict else None,
    )


class VideoVLMRewardInference:
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device="cuda", dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )

        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward["VQ"] = (reward["VQ"] - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
            reward["MQ"] = (reward["MQ"] - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
            reward["TA"] = (reward["TA"] - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, "constant", 0)

        return sequences_padded, attention_mask_padded

    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, videos, prompts, fps=None, num_frames=None, max_pixels=None):
        """
        Modified to accept either file paths (str) or Tensors (torch.Tensor) in 'videos'.
        """
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if isinstance(videos, list) and all(isinstance(v, torch.Tensor) for v in videos):
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": "file://dummy_path"},
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt, self.data_config.eval_dim, self.data_config.prompt_template_type
                                ),
                            },
                        ],
                    }
                ]
                for prompt in prompts
            ]

            image_inputs = None
            video_inputs = [process_video_tensor(tensor) for tensor in videos]
        else:
            if num_frames is None:
                chat_data = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": f"file://{video_path}",
                                    "max_pixels": max_pixels,
                                    "fps": fps,
                                    "sample_type": self.data_config.sample_type,
                                },
                                {
                                    "type": "text",
                                    "text": build_prompt(
                                        prompt, self.data_config.eval_dim, self.data_config.prompt_template_type
                                    ),
                                },
                            ],
                        },
                    ]
                    for video_path, prompt in zip(videos, prompts)
                ]
            else:
                chat_data = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": f"file://{video_path}",
                                    "max_pixels": max_pixels,
                                    "nframes": num_frames,
                                    "sample_type": self.data_config.sample_type,
                                },
                                {
                                    "type": "text",
                                    "text": build_prompt(
                                        prompt, self.data_config.eval_dim, self.data_config.prompt_template_type
                                    ),
                                },
                            ],
                        },
                    ]
                    for video_path, prompt in zip(videos, prompts)
                ]
            image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(
        self,
        videos,
        prompts,
        fps=None,
        num_frames=None,
        max_pixels=None,
        use_norm=True,
        return_batch_score=False,
        device="cpu",
        dtype=torch.float32,
    ):
        """
        videos: List[str] (paths) OR List[torch.Tensor]
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."

        batch = self.prepare_batch(videos, prompts, fps, num_frames, max_pixels)
        rewards = self.model(return_dict=True, **batch)["logits"]

        rewards = [{"VQ": reward[0].item(), "MQ": reward[1].item(), "TA": reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]["Overall"] = rewards[i]["VQ"] + rewards[i]["MQ"] + rewards[i]["TA"]
        if return_batch_score:
            batch_score = {
                "VQ": torch.tensor(sum(r["VQ"] for r in rewards) / len(rewards), device=device, dtype=dtype),
                "MQ": torch.tensor(sum(r["MQ"] for r in rewards) / len(rewards), device=device, dtype=dtype),
                "TA": torch.tensor(sum(r["TA"] for r in rewards) / len(rewards), device=device, dtype=dtype),
                "Overall": torch.tensor(sum(r["Overall"] for r in rewards) / len(rewards), device=device, dtype=dtype),
            }
            return batch_score

        return rewards


def main():
    load_from_pretrained = "/mnt/bn/yufan-dev-my/ysh_new/Ckpts/Videoreward"
    device = "cuda:0"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

    video_paths = [
        "/mnt/bn/yufan-dev-my/ysh_new/Codes/0_exps/0_results/t2v/short/sana-video/2_240_ori81.mp4",
        "/mnt/bn/yufan-dev-my/ysh_new/Codes/0_exps/0_results/t2v/short/sana-video/4_240_ori81.mp4",
        "/mnt/bn/yufan-dev-my/ysh_new/Codes/0_exps/0_results/t2v/short/sana-video/5_240_ori81.mp4",
    ]

    prompts = [
        "A stunning mid-afternoon landscape photograph with a low camera angle, showcasing several giant wooly mammoths treading through a snowy meadow. Their long, wooly fur gently billows in the brisk wind as they move, creating a sense of natural movement. Snow-covered trees and dramatic snow-capped mountains loom in the distance, adding to the majestic setting. Wispy clouds and a high sun cast a warm glow over the scene, enhancing the serene and awe-inspiring atmosphere. The depth of field brings out the detailed textures of the mammoths and the snowy environment, capturing every nuance of these prehistoric giants in breathtaking clarity.",
        "A drone view of waves crashing against the rugged cliffs along Big Sur’s Garay Point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore, casting long shadows. In the distance, a small island with a lighthouse stands tall, its beam piercing the twilight. Green shrubbery covers the cliff’s edge, and the steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. The camera angle provides a bird's-eye view, capturing the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway. The scene is bathed in a warm, golden hue, highlighting the textures and details of the rocky terrain.",
        "A close-up 3D animated scene of a short, fluffy monster kneeling beside a melting red candle. The monster has large, wide eyes and an open mouth, gazing at the flame with a look of wonder and curiosity. Its soft, fluffy fur contrasts with the warm, dramatic lighting that highlights every detail of its gentle, innocent expression. The pose conveys a sense of playfulness and exploration, as if the creature is discovering the world for the first time. The background features a cozy, warmly lit room with subtle hints of a fireplace and soft furnishings, enhancing the overall atmosphere. The use of warm colors and dramatic lighting creates a captivating and inviting scene.",
    ]

    # # Way 1
    print(f"\n{'=' * 20} Way 1: File Path Input {'=' * 20}")
    with torch.no_grad():
        rewards_path = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards_path)

    # Way 2
    print(f"\n{'=' * 20} Way 2: Tensor Input {'=' * 20}")
    from video_reader import PyVideoReader

    video_tensors = []
    print("Loading videos into Tensors manually...")
    for i, path in enumerate(video_paths):
        vr = PyVideoReader(path, threads=0)
        frames = vr.get_batch(range(len(vr)))
        tensor_input = torch.tensor(frames).permute(0, 3, 1, 2)
        video_tensors.append(tensor_input)
    del video_paths

    print(f"Loaded {len(video_tensors)} tensors.")

    with torch.no_grad():
        rewards_tensor = inferencer.reward(
            video_tensors,  # [torch.Size([249, 3, 480, 832]), torch.Size([249, 3, 480, 832]), torch.Size([249, 3, 480, 832])]
            prompts,
            use_norm=True,
            return_batch_score=False,
        )
        print(rewards_tensor)

    # --- 验证环节 ---
    print(f"\n{'=' * 20} Verification {'=' * 20}")
    for i, (r_path, r_tensor) in enumerate(zip(rewards_path, rewards_tensor)):
        score_path = r_path["Overall"]
        score_tensor = r_tensor["Overall"]
        diff = abs(score_path - score_tensor)

        status = "✅ CONSISTENT" if diff < 1e-3 else "❌ MISMATCH"
        print(f"Video {i + 1}:")
        print(f"  Path Input Score:   {score_path:.4f}")
        print(f"  Tensor Input Score: {score_tensor:.4f}")
        print(f"  Difference:         {diff:.6f} -> {status}")


if __name__ == "__main__":
    main()
