import copy
import json
import os
import time

import deepspeed
import torch
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from diffusers.training_utils import _collate_lora_metadata, free_memory
from diffusers.utils import convert_unet_state_dict_to_peft

from ..pipelines.pipeline_gigaworld import GigaworldPipeline
from ..utils.create_ema_zero3 import EMAModel_Zero3, _z3_params_to_fetch
from ..utils.utils_base import NORM_LAYER_PREFIXES, load_extra_components, save_extra_components


GB = 1024 * 1024 * 1024


# Adapted from diffusers-style ema https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L263
class EMAModel_Zero3_LoRA(EMAModel_Zero3):
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls, args, path, model_cls, lora_config, transformer_additional_kwargs={}
    ) -> "EMAModel_Zero3_LoRA":
        model = model_cls.from_pretrained(
            args.model_config.transformer_model_name_or_path,
            subfolder=args.model_config.subfolder or "transformer",
            transformer_additional_kwargs=transformer_additional_kwargs,
        )
        model.add_adapter(lora_config)

        # ------------- load lora -------------
        lora_state_dict = GigaworldPipeline.lora_state_dict(path)
        model_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        model_state_dict = convert_unet_state_dict_to_peft(model_state_dict)
        # incompatible_keys = set_peft_model_state_dict(model, model_state_dict, adapter_name="default")
        # FIXME: set_peft_model_state_dict will try to shard the state dict for tensor parallelism if the model is wrapped in a TP wrapper, which can cause issues if the state dict was not saved with TP sharding in mind. For now, we can monkey patch the function to disable sharding during loading, but ideally we should save the state dict in a way that is compatible with TP sharding to avoid this kind of hack.
        orig_tp = set_peft_model_state_dict.__globals__.get(
            "_maybe_shard_state_dict_for_tp",
            None,
        )

        set_peft_model_state_dict.__globals__["_maybe_shard_state_dict_for_tp"] = (
            lambda *args, **kwargs: None
        )

        try:
            incompatible_keys = set_peft_model_state_dict(
                model,
                model_state_dict,
                adapter_name="default",
            )
        finally:
            if orig_tp is not None:
                set_peft_model_state_dict.__globals__["_maybe_shard_state_dict_for_tp"] = orig_tp

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                accelerator.print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.model_config.train_norm_layers:
            model_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            model._transformer_norm_layers = GigaworldPipeline._load_norm_into_transformer(
                model_norm_state_dict,
                transformer=model,
                discard_original_layers=False,
            )
        # ------------- load lora -------------

        # ------------- load extra components -------------
        load_extra_components(args, model, os.path.join(path, "transformer_partial.pth"))
        # ------------- load extra components -------------

        ema_model = cls(model, model_cls=model_cls, model_config=model.config)

        with open(os.path.join(path, "ema_kwargs.json"), "r") as f:
            ema_kwargs = json.load(f)
        ema_model.load_state_dict(ema_kwargs)

        return ema_model

    def save_pretrained(
        self, args, path, pretrained_name_or_path, lora_config, transformer_additional_kwargs={}, transformer_cpu=None
    ):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        rank = int(os.getenv("RANK", "0"))

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                if rank == 0:
                    model_state_dict[k] = v.data.cpu().clone()

        if rank == 0:
            state_dict = self.state_dict()
            state_dict.pop("model")

            os.makedirs(path, exist_ok=True)
            print(f"state_dict, {state_dict.keys()}")
            t_start = time.perf_counter()
            print(f"[{t_start:.4f}] self.model_cls.from_pretrained")

            print("self.model_cls", self.model_cls)
            if transformer_cpu is None:
                model = self.model_cls.from_pretrained(
                    pretrained_name_or_path,
                    subfolder=args.model_config.subfolder or "transformer",
                    transformer_additional_kwargs=transformer_additional_kwargs,
                )
                model.add_adapter(lora_config)
            else:
                model = transformer_cpu
            t1 = time.perf_counter()
            print(f"[{t1:.4f}] after self.model_cls.from_pretrained (耗时 {t1 - t_start:.4f} 秒)")

            miss, unexp = model.load_state_dict(model_state_dict, strict=False)
            assert len(unexp) == 0, f"miss: {miss}; unexp: {unexp}"

            # ------------- only save lora -------------
            config_dict = model.config if hasattr(model, "config") else self.model_config
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)

            modules_to_save = {}
            transformer_lora_layers_to_save = get_peft_model_state_dict(model)
            if args.model_config.train_norm_layers:
                transformer_norm_layers_to_save = {
                    f"transformer.{name}": param
                    for name, param in model.named_parameters()
                    if any(k in name for k in NORM_LAYER_PREFIXES)
                }
                transformer_lora_layers_to_save = {
                    **transformer_lora_layers_to_save,
                    **transformer_norm_layers_to_save,
                }
            modules_to_save["transformer"] = model
            GigaworldPipeline.save_lora_weights(
                path,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
            # ------------- only save lora -------------

            # ------------- only save extra components -------------
            save_extra_components(args, model_state_dict=model_state_dict, output_dir=path)
            # ------------- only save extra components -------------

            t2 = time.perf_counter()
            print(f"[{t2:.4f}] after save_pretrained (耗时 {t2 - t1:.4f} 秒)")

            print(f"[{t2:.4f}] 总耗时 {t2 - t_start:.4f} 秒")

            with open(os.path.join(path, "ema_kwargs.json"), "w") as f:
                json.dump(state_dict, f, indent=2)

        model = None
        transformer_cpu = None
        params_to_fetch = None
        state_dict = None
        model_state_dict = None
        transformer_lora_layers_to_save = None
        transformer_norm_layers_to_save = None
        modules_to_save = None
        del model
        del transformer_cpu
        del params_to_fetch
        del state_dict
        del model_state_dict
        del transformer_lora_layers_to_save
        del transformer_norm_layers_to_save
        del modules_to_save
        free_memory()

        print(f"rank {rank} done saved ema!")


def gather_zero3ema(accelerator, ema_model):
    model_to_save = ema_model.model.module if hasattr(ema_model.model, "module") else ema_model.model
    model_state_dict = {}
    for k, v in model_to_save.named_parameters():
        # only gather z3 params
        params_to_fetch = _z3_params_to_fetch([v])
        with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
            # if accelerator.process_index == 0:
            model_state_dict[k] = v.data.cpu().clone()
    return model_state_dict


def create_ema_model(
    accelerator,
    args,
    transformer,
    resume_checkpoint_path,
    model_cls,
    model_config,
    ds_config=None,
    lora_config=None,
    update_after_step=0,
    transformer_additional_kwargs={},
):
    ds_config["train_micro_batch_size_per_gpu"] = args.training_config.train_batch_size
    ds_config["fp16"]["enabled"] = False
    ds_config["bf16"]["enabled"] = False
    ds_config["gradient_accumulation_steps"] = args.training_config.gradient_accumulation_steps
    ds_config["train_batch_size"] = (
        args.training_config.train_batch_size
        * args.training_config.gradient_accumulation_steps
        * accelerator.num_processes
    )
    accelerator.print(f"EMA deepspeed config {ds_config}")

    if resume_checkpoint_path:
        ema_model = EMAModel_Zero3_LoRA.from_pretrained(
            args=args,
            path=resume_checkpoint_path,
            model_cls=model_cls,
            lora_config=lora_config,
            transformer_additional_kwargs=transformer_additional_kwargs,
        )
        accelerator.print(f"Successully resume EMAModel_Zero3 from {resume_checkpoint_path}")
    else:
        ema_model = EMAModel_Zero3_LoRA(
            copy.deepcopy(transformer),
            decay=args.training_config.ema_decay,
            model_cls=model_cls,
            model_config=model_config,
            update_after_step=update_after_step,
        )
        accelerator.print(f"EMAModel_Zero3 finish, memory_allocated: {torch.cuda.memory_allocated() / GB:.2f} GB")
        accelerator.print("Successully deepcopy EMAModel_Zero3 from model")
    ema_model.model, _, _, _ = deepspeed.initialize(
        model=ema_model.model, config_params=ds_config, distributed_port=args.training_config.ema_zero3_port
    )
    return ema_model


def create_ema_final(
    accelerator,
    args,
    transformer_cpu,
    model_cls,
    ds_config,
    transformer_lora_config,
    update_after_step=0,
    resume_checkpoint_path=None,
    transformer_additional_kwargs=None,
):
    ema_transformer = create_ema_model(
        accelerator,
        args=args,
        transformer=transformer_cpu,
        resume_checkpoint_path=resume_checkpoint_path,
        model_cls=model_cls,
        model_config=transformer_cpu.config,
        ds_config=ds_config,
        lora_config=transformer_lora_config,
        update_after_step=update_after_step,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    free_memory()
    return ema_transformer


if __name__ == "__main__":
    import json
    import sys
    from argparse import Namespace

    import deepspeed
    from accelerate import Accelerator

    sys.path.append("../../")
    from gigaworld.modules.transformer_gigaworld import GigaworldTransformer3DModel

    args = Namespace()
    args.data_config = Namespace()
    args.training_config = Namespace()
    args.model_config = Namespace()
    args.training_config.train_batch_size = 1
    args.training_config.gradient_accumulation_steps = 1
    args.training_config.ema_decay = 0.999
    args.training_config.ema_zero3_port = 10543
    args.model_config.train_norm_layers = False
    args.model_config.transformer_model_name_or_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    args.training_config.ema_deepspeed_config_file = "../../scripts/accelerate_configs/zero3.json"
    resume_checkpoint_path = None

    output_dir = "temp"
    accelerator = Accelerator()

    model_cls = GigaworldTransformer3DModel
    transformer = model_cls.from_pretrained(
        args.model_config.transformer_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    target_modules = set()
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.add(name)
    target_modules = list(target_modules)
    lora_config = LoraConfig(
        r=256,
        lora_alpha=256,
        # target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    transformer.add_adapter(lora_config)

    transformer_cpu = copy.deepcopy(transformer)
    transformer.to(device=accelerator.device, dtype=torch.bfloat16)
    accelerator.print(f"Load model finish, memory_allocated: {torch.cuda.memory_allocated() / GB:.2f} GB")

    with open(args.training_config.ema_deepspeed_config_file, "r") as f:
        ds_config = json.load(f)

    ema_transformer = create_ema_final(
        accelerator=accelerator,
        args=args,
        transformer_cpu=transformer_cpu,
        model_cls=model_cls,
        ds_config=ds_config,
        transformer_lora_config=lora_config,
    )
