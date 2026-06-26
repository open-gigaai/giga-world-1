import torch


# base_lrs: [5e-05] last_epoch: 384000 _step_count: 384001 _get_lr_called_within_step: False _last_lr: [5e-05] lr_lambdas: [None]

scheduler_path = "ablation_stage2-all_r256_1120_bigbatch_final_3/checkpoint-11500/scheduler.bin"
scheduler_state = torch.load(scheduler_path, map_location="cpu")

print("Original values:")
print(f"  last_epoch: {scheduler_state['last_epoch']}")
print(f"  _step_count: {scheduler_state['_step_count']}")
print(f"  base_lrs: {scheduler_state['base_lrs']}")
print(f"  _last_lr: {scheduler_state['_last_lr']}")

scheduler_state["last_epoch"] = 736000
scheduler_state["_step_count"] = 736001
scheduler_state["base_lrs"] = [4e-05]
scheduler_state["_last_lr"] = [4e-05]

torch.save(scheduler_state, scheduler_path)

# 验证修改
print("\nModified values:")
print(f"  last_epoch: {scheduler_state['last_epoch']}")
print(f"  _step_count: {scheduler_state['_step_count']}")
print(f"  base_lrs: {scheduler_state['base_lrs']}")
print(f"  _last_lr: {scheduler_state['_last_lr']}")
print(f"\n✓ Successfully saved to: {scheduler_path}")
