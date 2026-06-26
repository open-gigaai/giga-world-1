import random

import torch

from .utils_base import apply_schedule_shift


def apply_error_injection(
    args,
    recycle_vars,
    model_input,
    noise,
    timesteps,
    latents_history_long,
    latents_history_mid,
    latents_history_short,
    model_input_w_error,
    noise_w_error,
    is_keep_x0,
    latent_window_size,
):
    # Check if buffer has data for the current timestep grid
    current_grid_idx = get_timestep_grid(args, recycle_vars, timesteps, noise)
    has_latent_buffer_data = len(recycle_vars.latent_error_buffer[current_grid_idx]) > 0
    has_y_buffer_data = any(len(buffer) > 0 for buffer in recycle_vars.y_error_buffer.values())

    add_error_latent = False
    add_error_noise = False
    add_error_y = False
    use_clean_input = False

    latent_random = random.random()
    noise_random = random.random()
    y_random = random.random()
    clean_random = random.random()

    if latent_random < args.training_config.latent_prob:
        add_error_latent = True
    if noise_random < args.training_config.noise_prob:
        add_error_noise = True
    if y_random < args.training_config.y_prob:
        add_error_y = True
    if clean_random < args.training_config.clean_prob:
        add_error_noise = False
        add_error_y = False
        add_error_latent = False
        use_clean_input = True

    if add_error_noise and has_latent_buffer_data:
        noise_error_sampled = sample_noise_error_from_noise_buffer(
            args, recycle_vars, model_input, timesteps, model_input.dtype, model_input.device
        )
        noise_w_error = noise + noise_error_sampled.to(model_input.dtype)

    if add_error_y and has_y_buffer_data:
        len_4x = latents_history_long.shape[2]
        len_2x = latents_history_mid.shape[2]
        len_1x = latents_history_short.shape[2]

        hist_seq_len = len_4x + len_2x + len_1x
        hist_seq_len_copy = hist_seq_len

        ori_len_1x = len_1x
        if is_keep_x0:
            len_1x -= 1
            hist_seq_len -= 1
            begin_num = 1
        else:
            begin_num = 0

        max_windows = hist_seq_len // latent_window_size
        tail_num = hist_seq_len % latent_window_size

        assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

        tail_latents_history = None
        begin_latents_history = None
        if tail_num != 0:
            tail_latents_history, latents_history_long = (
                latents_history_long[:, :, :tail_num, :, :],
                latents_history_long[:, :, tail_num:, :, :],
            )
            # for tail
            if random.random() < args.training_config.y_prob:
                y_error_sampled = sample_y_error_from_latent_buffer(
                    args, recycle_vars, model_input, model_input.dtype, model_input.device
                )
                random_error_num = torch.randint(1, tail_num + 1, (1,)).item()
                tail_latents_history[:, :, -random_error_num:, ...] = (
                    tail_latents_history[:, :, -random_error_num:, ...]
                    + y_error_sampled[:, :, -random_error_num:, ...]
                )
        if begin_num != 0:
            begin_latents_history, latents_history_short = (
                latents_history_short[:, :, :begin_num, :, :],
                latents_history_short[:, :, begin_num:, :, :],
            )
            # for begin
            if random.random() < args.training_config.y_prob:
                y_error_sampled = sample_y_error_from_latent_buffer(
                    args, recycle_vars, model_input, model_input.dtype, model_input.device
                )
                begin_latents_history = begin_latents_history + y_error_sampled[:, :, :1, ...]

        # for mid
        mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
        window_num = mid_latents_history.shape[2] // latent_window_size
        assert mid_latents_history.shape[2] % latent_window_size == 0, (
            f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
        )
        seq_begin = 0
        for _ in range(window_num):
            seq_end = seq_begin + latent_window_size
            if random.random() < args.training_config.y_prob:
                y_error_sampled = sample_y_error_from_latent_buffer(
                    args, recycle_vars, model_input, model_input.dtype, model_input.device
                )
                max_start_idx = max(0, y_error_sampled.shape[2] - args.training_config.y_error_num)
                random_frame_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
                error_to_add = y_error_sampled[
                    :, :, random_frame_idx : random_frame_idx + args.training_config.y_error_num, ...
                ]
                # Modify
                mid_latents_history[:, :, seq_begin:seq_end, :, :][
                    :, :, random_frame_idx : random_frame_idx + args.training_config.y_error_num, :, :
                ] = (
                    mid_latents_history[:, :, seq_begin:seq_end, :, :][
                        :, :, random_frame_idx : random_frame_idx + args.training_config.y_error_num, :, :
                    ]
                    + error_to_add
                )
            seq_begin = seq_end

        # recover
        recovers = []
        if tail_latents_history is not None:
            recovers.append(tail_latents_history)
        recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
        if begin_latents_history is not None:
            recovers.append(begin_latents_history)
        recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
        mid_latents_history = torch.cat(recovers, dim=2)
        latents_history_long, latents_history_mid, latents_history_short = mid_latents_history.split(
            [len_4x, len_2x, ori_len_1x], dim=2
        )

    if add_error_latent and has_latent_buffer_data:
        latent_error_sampled = sample_latent_error_from_latent_buffer(
            args, recycle_vars, model_input, timesteps, model_input.dtype, model_input.device
        )
        model_input_w_error = model_input + latent_error_sampled.to(model_input.dtype)

    return (
        model_input_w_error,
        noise_w_error,
        latents_history_long,
        latents_history_mid,
        latents_history_short,
        use_clean_input,
    )


def step_recycle(scheduler, model_output, timestep, sample, to_final=False, self_corr=False):
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.cpu()
    timestep_id = torch.argmin((scheduler.temp_timesteps - timestep).abs())
    sigma = scheduler.temp_sigmas[timestep_id]
    if to_final or timestep_id + 1 >= len(scheduler.temp_timesteps):
        sigma_ = 1 if self_corr else 0
    else:
        sigma_ = scheduler.temp_sigmas[timestep_id + 1]
    prev_sample = sample + model_output * (sigma_ - sigma)
    return prev_sample


def get_timesteps(
    num_inference_steps=50,
    denoising_strength=1,
    shift=1.0,
    num_train_timesteps=1000,
    sigma_max=1.0,
    sigma_min=0.0,
    inverse_timesteps=False,
    extra_one_step=True,
    reverse_sigmas=False,
):
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    if extra_one_step:
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
    else:
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
    if inverse_timesteps:
        sigmas = torch.flip(sigmas, dims=[0])
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    if reverse_sigmas:
        sigmas = 1 - sigmas
    timesteps = sigmas * num_train_timesteps
    return timesteps, sigmas


def get_timestep_grid(args, recycle_vars, timesteps, noise):
    """Get the grid index for a given timesteps."""
    # Handle different timesteps formats (scalar tensor, tensor with batch dim, etc.)
    if isinstance(timesteps, torch.Tensor):
        if timesteps.numel() == 1:
            # Single timesteps value
            timestep_val = timesteps.item()
        else:
            # Tensor with batch dimension, take the first element
            timestep_val = timesteps.flatten()[0].item()
    else:
        # Already a scalar value
        timestep_val = timesteps

    if args.training_config.use_dynamic_shifting:
        temp_sigmas = apply_schedule_shift(
            recycle_vars.recycle_sigmas,
            noise,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )  # torch.Size([2, 1, 1, 1, 1])

        temp_inferece_timesteps = temp_sigmas * 1000.0  # rescale to [0, 1000.0)
        while temp_inferece_timesteps.ndim > 1:
            temp_inferece_timesteps = temp_inferece_timesteps.squeeze(-1)
    else:
        temp_inferece_timesteps = recycle_vars.recycle_inferece_timesteps

    # Ensure timesteps is within valid range and calculate grid index
    timestep_val = max(0, min(timestep_val, 999))  # Clamp to [0, 999]
    grid_idx = torch.argmin((temp_inferece_timesteps - timestep_val).abs()).item()

    # Ensure grid index is within valid range
    max_grid_idx = len(recycle_vars.latent_error_buffer) - 1
    grid_idx = min(grid_idx, max_grid_idx)

    return grid_idx


def sample_noise_error_from_noise_buffer(args, recycle_vars, latents, timestep, dtype=torch.bfloat16, device="cpu"):
    """Randomly sample an error from the buffer based on timestep grid."""
    grid_idx = get_timestep_grid(args, recycle_vars, timestep, latents)

    if not recycle_vars.latent_error_buffer[grid_idx]:
        return torch.zeros_like(latents)

    # Randomly select one sample from the corresponding grid
    selected_sample = random.choice(recycle_vars.latent_error_buffer[grid_idx])
    error_sample = selected_sample

    min_mod = 1.0 - args.training_config.error_modulate_factor
    max_mod = 1.0 + args.training_config.error_modulate_factor
    intensity_mod = random.uniform(min_mod, max_mod)
    error_sample = error_sample * intensity_mod

    error_sample = error_sample.to(device, dtype=dtype)

    return error_sample


def sample_latent_error_from_latent_buffer(args, recycle_vars, latents, timestep, dtype=torch.bfloat16, device="cpu"):
    """Randomly sample an error from the buffer based on timestep grid."""
    grid_idx = get_timestep_grid(args, recycle_vars, timestep, latents)

    if not recycle_vars.y_error_buffer[grid_idx]:
        return torch.zeros_like(latents)

    # Randomly select one sample from the corresponding grid
    selected_sample = random.choice(recycle_vars.y_error_buffer[grid_idx])
    error_sample = selected_sample

    min_mod = 1.0 - args.training_config.error_modulate_factor
    max_mod = 1.0 + args.training_config.error_modulate_factor
    intensity_mod = random.uniform(min_mod, max_mod)
    error_sample = error_sample * intensity_mod

    error_sample = error_sample.to(device, dtype=dtype)

    return error_sample


def sample_y_error_from_latent_buffer(args, recycle_vars, latents, dtype=torch.bfloat16, device="cpu"):
    """Specially sample y_error from buffer - can be configured to sample from all grids or custom range."""
    # Sample from all grids that have data
    all_samples = []
    for grid_idx, buffer in recycle_vars.y_error_buffer.items():
        if buffer:  # Only add non-empty buffers
            all_samples.extend(buffer)

    if not all_samples:
        return torch.zeros_like(latents)

    # Randomly select one sample from all available samples
    selected_sample = random.choice(all_samples)
    error_sample = selected_sample

    min_mod = 1.0 - args.training_config.error_modulate_factor
    max_mod = 1.0 + args.training_config.error_modulate_factor
    intensity_mod = random.uniform(min_mod, max_mod)
    error_sample = error_sample * intensity_mod

    error_sample = error_sample.to(device, dtype=dtype)

    return error_sample


def compute_l2_distance_batch(new_tensor, stored_tensors):
    """Compute L2 distances between new tensor and all stored tensors efficiently."""
    if not stored_tensors:
        return torch.tensor([])

    # Stack all stored tensors for batch computation
    stored_stack = torch.stack(stored_tensors)  # [num_stored, ...]
    new_flat = new_tensor.flatten()
    stored_flat = stored_stack.flatten(start_dim=1)  # [num_stored, flattened_size]

    # Compute L2 distances in batch
    distances = torch.norm(stored_flat - new_flat.unsqueeze(0), p=2, dim=1)
    return distances


def compute_l2_distance(tensor1, tensor2):
    """Compute L2 distance between two tensors"""
    # Flatten tensors
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    # Compute L2 distance (Euclidean distance)
    l2_distance = torch.norm(flat1 - flat2, p=2)
    return l2_distance.item()


def add_error_to_latent_buffer(args, recycle_vars, error_sample, timestep, noisy_model_input):
    """Add error sample to buffer using specified replacement strategy based on timestep grid."""
    grid_idx = get_timestep_grid(args, recycle_vars, timestep, noisy_model_input)
    error_cpu = error_sample.detach().cpu()

    if len(recycle_vars.latent_error_buffer[grid_idx]) < args.training_config.error_buffer_size:
        # Buffer not full, simply add
        recycle_vars.latent_error_buffer[grid_idx].append(error_cpu)
    else:
        # Buffer full, use specified replacement strategy
        if args.training_config.buffer_replacement_strategy == "random":
            # Random replacement - O(1), fastest
            replace_idx = random.randint(0, len(recycle_vars.latent_error_buffer[grid_idx]) - 1)
            recycle_vars.latent_error_buffer[grid_idx][replace_idx] = error_cpu

        elif args.training_config.buffer_replacement_strategy == "fifo":
            # First-in-first-out - O(1), simple queue behavior
            recycle_vars.latent_error_buffer[grid_idx].pop(0)
            recycle_vars.latent_error_buffer[grid_idx].append(error_cpu)

        elif args.training_config.buffer_replacement_strategy == "l2_batch":
            # Batch L2 computation - O(n) but vectorized, much faster than original
            distances = compute_l2_distance_batch(error_cpu, recycle_vars.latent_error_buffer[grid_idx])
            most_similar_idx = torch.argmin(distances).item()
            recycle_vars.latent_error_buffer[grid_idx][most_similar_idx] = error_cpu

        elif args.training_config.buffer_replacement_strategy == "l2_similarity":
            # Original L2 similarity method - O(n), slowest but most precise
            min_distance = float("inf")
            most_similar_idx = -1

            for i, stored_error in enumerate(recycle_vars.latent_error_buffer[grid_idx]):
                distance = compute_l2_distance(error_cpu, stored_error)
                if distance < min_distance:
                    min_distance = distance
                    most_similar_idx = i

            if most_similar_idx != -1:
                recycle_vars.latent_error_buffer[grid_idx][most_similar_idx] = error_cpu


def add_error_to_y_buffer(args, recycle_vars, error_sample, timestep, noisy_model_input):
    """Add error sample to buffer using specified replacement strategy based on timestep grid."""
    grid_idx = get_timestep_grid(args, recycle_vars, timestep, noisy_model_input)
    error_cpu = error_sample.detach().cpu()

    if len(recycle_vars.y_error_buffer[grid_idx]) < args.training_config.error_buffer_size:
        # Buffer not full, simply add
        recycle_vars.y_error_buffer[grid_idx].append(error_cpu)
    else:
        # Buffer full, use specified replacement strategy
        if args.training_config.buffer_replacement_strategy == "random":
            # Random replacement - O(1), fastest
            replace_idx = random.randint(0, len(recycle_vars.y_error_buffer[grid_idx]) - 1)
            recycle_vars.y_error_buffer[grid_idx][replace_idx] = error_cpu

        elif args.training_config.buffer_replacement_strategy == "fifo":
            # First-in-first-out - O(1), simple queue behavior
            recycle_vars.y_error_buffer[grid_idx].pop(0)
            recycle_vars.y_error_buffer[grid_idx].append(error_cpu)

        elif args.training_config.buffer_replacement_strategy == "l2_batch":
            # Batch L2 computation - O(n) but vectorized, much faster than original
            distances = compute_l2_distance_batch(error_cpu, recycle_vars.y_error_buffer[grid_idx])
            most_similar_idx = torch.argmin(distances).item()
            recycle_vars.y_error_buffer[grid_idx][most_similar_idx] = error_cpu

        elif args.training_config.buffer_replacement_strategy == "l2_similarity":
            # Original L2 similarity method - O(n), slowest but most precise
            min_distance = float("inf")
            most_similar_idx = -1

            for i, stored_error in enumerate(recycle_vars.y_error_buffer[grid_idx]):
                distance = compute_l2_distance(error_cpu, stored_error)
                if distance < min_distance:
                    min_distance = distance
                    most_similar_idx = i

            if most_similar_idx != -1:
                recycle_vars.y_error_buffer[grid_idx][most_similar_idx] = error_cpu


def update_error_buffers_distributed(
    args, recycle_vars, gathered_noise_errors, gathered_y_errors, gathered_timesteps, noisy_model_input
):
    """Update error buffers with samples gathered from all processes."""
    # gathered_tensors have shape [num_gpus, batch_size, ...] for errors
    # gathered_timesteps have shape [num_gpus, batch_size] for timesteps
    # In this case, batch_size is 1, so shapes are [num_gpus, 1, ...] and [num_gpus, 1]
    num_gpus = gathered_noise_errors.shape[0]
    for i in range(num_gpus):
        noise_error_sample = gathered_noise_errors[i]
        y_error_sample = gathered_y_errors[i]
        timestep_sample = gathered_timesteps[i]  # Get the corresponding timestep for this GPU

        add_error_to_latent_buffer(args, recycle_vars, noise_error_sample, timestep_sample, noisy_model_input)
        add_error_to_y_buffer(args, recycle_vars, y_error_sample, timestep_sample, noisy_model_input)


def update_error_buffers_local(args, recycle_vars, noise_error, y_error, timestep, noisy_model_input):
    """Update error buffers with samples from local GPU only (post-warmup)."""
    add_error_to_latent_buffer(args, recycle_vars, noise_error, timestep, noisy_model_input)
    add_error_to_y_buffer(args, recycle_vars, y_error, timestep, noisy_model_input)
