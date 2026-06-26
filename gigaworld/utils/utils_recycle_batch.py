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
    batch_size, _, _, h, w = noise.shape

    # Get grid indices for all batch items
    current_grid_indices = get_timestep_grid(args, recycle_vars, timesteps, noise)

    # Handle single item (backward compatibility)
    if isinstance(current_grid_indices, int):
        current_grid_indices = torch.tensor([current_grid_indices], device=noise.device)

    # Check buffer availability for each batch item
    has_latent_buffer_data = torch.tensor(
        [len(recycle_vars.latent_error_buffer[(h, w)][grid_idx.item()]) > 0 for grid_idx in current_grid_indices],
        device=noise.device,
    )

    has_y_buffer_data = any(len(buffer) > 0 for buffer in recycle_vars.y_error_buffer[(h, w)].values())

    # Generate random decisions for each batch item
    latent_random = torch.rand(batch_size, device=noise.device)
    noise_random = torch.rand(batch_size, device=noise.device)
    y_random = torch.rand(batch_size, device=noise.device)
    clean_random = torch.rand(batch_size, device=noise.device)

    # Determine which operations to apply for each batch item
    add_error_latent = latent_random < args.training_config.latent_prob
    add_error_noise = noise_random < args.training_config.noise_prob
    add_error_y = y_random < args.training_config.y_prob
    use_clean_input = clean_random < args.training_config.clean_prob

    # Clean input overrides all errors
    add_error_noise = add_error_noise & ~use_clean_input
    add_error_y = add_error_y & ~use_clean_input
    add_error_latent = add_error_latent & ~use_clean_input

    # Apply noise error
    if add_error_noise.any() and has_latent_buffer_data.any():
        noise_error_sampled = sample_noise_error_from_noise_buffer(
            args, recycle_vars, model_input, timesteps, model_input.dtype, model_input.device
        )
        mask = add_error_noise & has_latent_buffer_data
        if mask.any():
            noise_w_error[mask] = noise[mask] + noise_error_sampled[mask].to(model_input.dtype)

    # Apply y error for selected batch items
    if add_error_y.any() and has_y_buffer_data:
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

        # Process each batch item independently
        for batch_idx in range(batch_size):
            if not add_error_y[batch_idx]:
                continue

            # Split history for this batch item
            tail_latents_history = None
            begin_latents_history = None

            latents_4x_item = latents_history_long[batch_idx : batch_idx + 1]
            latents_2x_item = latents_history_mid[batch_idx : batch_idx + 1]
            latents_clean_item = latents_history_short[batch_idx : batch_idx + 1]

            if tail_num != 0:
                tail_latents_history = latents_4x_item[:, :, :tail_num, :, :]
                latents_4x_item = latents_4x_item[:, :, tail_num:, :, :]
                # Apply tail error
                if tail_latents_history.sum() != 0 and random.random() < args.training_config.y_prob:
                    y_error_sampled = sample_y_error_from_latent_buffer(
                        args,
                        recycle_vars,
                        model_input[batch_idx : batch_idx + 1],
                        model_input.dtype,
                        model_input.device,
                    )
                    random_error_num = torch.randint(1, tail_num + 1, (1,)).item()
                    tail_latents_history[:, :, -random_error_num:, ...] = (
                        tail_latents_history[:, :, -random_error_num:, ...]
                        + y_error_sampled[:, :, -random_error_num:, ...]
                    )

            if begin_num != 0:
                begin_latents_history = latents_clean_item[:, :, :begin_num, :, :]
                latents_clean_item = latents_clean_item[:, :, begin_num:, :, :]
                # Apply begin error
                if begin_latents_history.sum() != 0 and random.random() < args.training_config.y_prob:
                    y_error_sampled = sample_y_error_from_latent_buffer(
                        args,
                        recycle_vars,
                        model_input[batch_idx : batch_idx + 1],
                        model_input.dtype,
                        model_input.device,
                    )
                    begin_latents_history = begin_latents_history + y_error_sampled[:, :, :1, ...]

            # Process mid windows
            mid_latents_history = torch.cat([latents_4x_item, latents_2x_item, latents_clean_item], dim=2)
            window_num = mid_latents_history.shape[2] // latent_window_size
            assert mid_latents_history.shape[2] % latent_window_size == 0, (
                f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
            )

            seq_begin = 0
            for _ in range(window_num):
                seq_end = seq_begin + latent_window_size
                if (
                    mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0
                    and random.random() < args.training_config.y_prob
                ):
                    y_error_sampled = sample_y_error_from_latent_buffer(
                        args,
                        recycle_vars,
                        model_input[batch_idx : batch_idx + 1],
                        model_input.dtype,
                        model_input.device,
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

            # Recover structure
            recovers = []
            if tail_latents_history is not None:
                recovers.append(tail_latents_history)
            recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
            if begin_latents_history is not None:
                recovers.append(begin_latents_history)
            recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
            mid_latents_history = torch.cat(recovers, dim=2)

            # Split and update back to original tensors
            latents_4x_recovered, latents_2x_recovered, latents_clean_recovered = mid_latents_history.split(
                [len_4x, len_2x, ori_len_1x], dim=2
            )
            latents_history_long[batch_idx : batch_idx + 1] = latents_4x_recovered
            latents_history_mid[batch_idx : batch_idx + 1] = latents_2x_recovered
            latents_history_short[batch_idx : batch_idx + 1] = latents_clean_recovered

    # Apply latent error
    if add_error_latent.any() and has_latent_buffer_data.any():
        latent_error_sampled = sample_latent_error_from_latent_buffer(
            args, recycle_vars, model_input, timesteps, model_input.dtype, model_input.device
        )
        mask = add_error_latent & has_latent_buffer_data
        if mask.any():
            model_input_w_error[mask] = model_input[mask] + latent_error_sampled[mask].to(model_input.dtype)

    return (
        model_input_w_error,
        noise_w_error,
        latents_history_long,
        latents_history_mid,
        latents_history_short,
        use_clean_input,
    )


def step_recycle(scheduler, model_output, timestep, sample, to_final=False, self_corr=False):
    """
    Args:
        timestep: scalar, 1D tensor with shape [batch_size], or tensor that can be flattened
    """
    # Normalize timestep to 1D tensor
    if isinstance(timestep, torch.Tensor):
        timestep_vals = timestep.flatten().cpu()
    else:
        # Scalar value, convert to tensor
        timestep_vals = torch.tensor([timestep])

    batch_size = timestep_vals.shape[0]

    # Find timestep indices for all batch items
    # timestep_vals: [batch_size], scheduler.temp_timesteps: [num_timesteps]
    diffs = torch.abs(
        scheduler.temp_timesteps.unsqueeze(0) - timestep_vals.unsqueeze(-1)
    )  # [batch_size, num_timesteps]
    timestep_ids = torch.argmin(diffs, dim=-1)  # [batch_size]

    # Get sigmas for all batch items
    sigmas = scheduler.temp_sigmas[timestep_ids]  # [batch_size]

    # Calculate next sigmas
    if to_final:
        # All items go to final
        sigmas_next = torch.ones(batch_size) if self_corr else torch.zeros(batch_size)
    else:
        # Check which items are at the end
        at_end = timestep_ids + 1 >= len(scheduler.temp_timesteps)

        # Get next sigmas (clamped to valid range)
        next_ids = torch.clamp(timestep_ids + 1, 0, len(scheduler.temp_timesteps) - 1)
        sigmas_next = scheduler.temp_sigmas[next_ids]  # [batch_size]

        # Override with 1 or 0 for items at the end
        if self_corr:
            sigmas_next[at_end] = 1.0
        else:
            sigmas_next[at_end] = 0.0

    # Move sigmas to same device as sample
    sigmas = sigmas.to(sample.device, dtype=sample.dtype)
    sigmas_next = sigmas_next.to(sample.device, dtype=sample.dtype)

    # Compute prev_sample for all batch items
    # Reshape sigmas to broadcast correctly: [batch_size, 1, 1, 1, 1] for 5D tensors
    shape = [batch_size] + [1] * (sample.ndim - 1)
    sigma_diff = (sigmas_next - sigmas).view(*shape)

    prev_sample = sample + model_output * sigma_diff

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
    _, _, _, h, w = noise.shape

    # Handle different timesteps formats (scalar tensor, tensor with batch dim, etc.)
    if isinstance(timesteps, torch.Tensor):
        timestep_vals = timesteps.flatten()
    else:
        # Already a scalar value
        timestep_vals = torch.tensor([timesteps], device=noise.device if hasattr(noise, "device") else "cpu")

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
    timestep_vals = torch.clamp(timestep_vals, 0, 999)
    grid_timesteps = temp_inferece_timesteps.to(timestep_vals.device)

    diffs = torch.abs(grid_timesteps.unsqueeze(0) - timestep_vals.unsqueeze(-1))
    grid_indices = torch.argmin(diffs, dim=-1)

    # Ensure grid index is within valid range
    max_grid_idx = len(recycle_vars.latent_error_buffer[(h, w)]) - 1
    grid_indices = torch.clamp(grid_indices, 0, max_grid_idx)

    return grid_indices


def sample_noise_error_from_noise_buffer(args, recycle_vars, latents, timestep, dtype=torch.bfloat16, device="cpu"):
    """Randomly sample an error from the buffer based on timestep grid."""
    batch_size, _, _, h, w = latents.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, latents)

    # Handle single item (backward compatibility)
    if isinstance(grid_indices, int):
        grid_indices = torch.tensor([grid_indices], device=device)

    # Initialize output tensor
    error_samples = torch.zeros_like(latents)

    # Sample error for each item in batch
    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()

        if not recycle_vars.latent_error_buffer[(h, w)][grid_idx]:
            continue  # Keep zeros for this batch item

        # Randomly select one sample from the corresponding grid
        selected_sample = random.choice(recycle_vars.latent_error_buffer[(h, w)][grid_idx])

        # Apply random intensity modulation
        min_mod = 1.0 - args.training_config.error_modulate_factor
        max_mod = 1.0 + args.training_config.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)

        error_sample = selected_sample * intensity_mod
        error_sample = error_sample

        # Assign to the i-th batch item
        error_samples[i] = error_sample

    error_samples = error_samples.to(device, dtype=dtype)

    return error_samples


def sample_latent_error_from_latent_buffer(args, recycle_vars, latents, timestep, dtype=torch.bfloat16, device="cpu"):
    """Randomly sample an error from the buffer based on timestep grid."""
    batch_size, _, _, h, w = latents.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, latents)

    # Handle single item (backward compatibility)
    if isinstance(grid_indices, int):
        grid_indices = torch.tensor([grid_indices], device=device)

    # Initialize output tensor
    error_samples = torch.zeros_like(latents)

    # Sample error for each item in batch
    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()

        if not recycle_vars.y_error_buffer[(h, w)][grid_idx]:
            continue  # Keep zeros for this batch item

        # Randomly select one sample from the corresponding grid
        selected_sample = random.choice(recycle_vars.y_error_buffer[(h, w)][grid_idx])

        # Apply random intensity modulation
        min_mod = 1.0 - args.training_config.error_modulate_factor
        max_mod = 1.0 + args.training_config.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)

        error_sample = selected_sample * intensity_mod
        error_sample = error_sample

        # Assign to the i-th batch item
        error_samples[i] = error_sample

    error_samples = error_samples.to(device, dtype=dtype)

    return error_samples


def sample_y_error_from_latent_buffer(args, recycle_vars, latents, dtype=torch.bfloat16, device="cpu"):
    """Specially sample y_error from buffer - can be configured to sample from all grids or custom range."""
    batch_size, _, _, h, w = latents.shape

    # Sample from all grids that have data
    all_samples = []
    for grid_idx, buffer in recycle_vars.y_error_buffer[(h, w)].items():
        if buffer:  # Only add non-empty buffers
            all_samples.extend(buffer)

    if not all_samples:
        return torch.zeros_like(latents)

    # Initialize output tensor
    error_samples = torch.zeros_like(latents)

    # Sample independently for each batch item
    for i in range(batch_size):
        # Randomly select one sample from all available samples
        selected_sample = random.choice(all_samples)

        # Apply random intensity modulation
        min_mod = 1.0 - args.training_config.error_modulate_factor
        max_mod = 1.0 + args.training_config.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)

        error_sample = selected_sample * intensity_mod
        error_sample = error_sample

        # Assign to the i-th batch item
        error_samples[i] = error_sample

    error_samples = error_samples.to(device, dtype=dtype)

    return error_samples


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
    batch_size, _, _, h, w = noisy_model_input.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, noisy_model_input)
    error_cpu = error_sample.detach().cpu()

    # Process each batch item
    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()
        error_cpu = error_sample[i].detach().cpu()

        if len(recycle_vars.latent_error_buffer[(h, w)][grid_idx]) < args.training_config.error_buffer_size:
            # Buffer not full, simply add
            recycle_vars.latent_error_buffer[(h, w)][grid_idx].append(error_cpu)
        else:
            # Buffer full, use specified replacement strategy
            if args.training_config.buffer_replacement_strategy == "random":
                # Random replacement - O(1), fastest
                replace_idx = random.randint(0, len(recycle_vars.latent_error_buffer[(h, w)][grid_idx]) - 1)
                recycle_vars.latent_error_buffer[(h, w)][grid_idx][replace_idx] = error_cpu

            elif args.training_config.buffer_replacement_strategy == "fifo":
                # First-in-first-out - O(1), simple queue behavior
                recycle_vars.latent_error_buffer[(h, w)][grid_idx].pop(0)
                recycle_vars.latent_error_buffer[(h, w)][grid_idx].append(error_cpu)

            elif args.training_config.buffer_replacement_strategy == "l2_batch":
                # Batch L2 computation - O(n) but vectorized, much faster than original
                distances = compute_l2_distance_batch(error_cpu, recycle_vars.latent_error_buffer[(h, w)][grid_idx])
                most_similar_idx = torch.argmin(distances).item()
                recycle_vars.latent_error_buffer[(h, w)][grid_idx][most_similar_idx] = error_cpu

            elif args.training_config.buffer_replacement_strategy == "l2_similarity":
                # Original L2 similarity method - O(n), slowest but most precise
                min_distance = float("inf")
                most_similar_idx = -1

                for j, stored_error in enumerate(recycle_vars.latent_error_buffer[(h, w)][grid_idx]):
                    distance = compute_l2_distance(error_cpu, stored_error)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_idx = j

                if most_similar_idx != -1:
                    recycle_vars.latent_error_buffer[(h, w)][grid_idx][most_similar_idx] = error_cpu


def add_error_to_y_buffer(args, recycle_vars, error_sample, timestep, noisy_model_input):
    """Add error sample to buffer using specified replacement strategy based on timestep grid."""
    batch_size, _, _, h, w = noisy_model_input.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, noisy_model_input)
    error_cpu = error_sample.detach().cpu()

    # Process each batch item
    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()
        error_cpu = error_sample[i].detach().cpu()

        if len(recycle_vars.y_error_buffer[(h, w)][grid_idx]) < args.training_config.error_buffer_size:
            # Buffer not full, simply add
            recycle_vars.y_error_buffer[(h, w)][grid_idx].append(error_cpu)
        else:
            # Buffer full, use specified replacement strategy
            if args.training_config.buffer_replacement_strategy == "random":
                # Random replacement - O(1), fastest
                replace_idx = random.randint(0, len(recycle_vars.y_error_buffer[(h, w)][grid_idx]) - 1)
                recycle_vars.y_error_buffer[(h, w)][grid_idx][replace_idx] = error_cpu

            elif args.training_config.buffer_replacement_strategy == "fifo":
                # First-in-first-out - O(1), simple queue behavior
                recycle_vars.y_error_buffer[(h, w)][grid_idx].pop(0)
                recycle_vars.y_error_buffer[(h, w)][grid_idx].append(error_cpu)

            elif args.training_config.buffer_replacement_strategy == "l2_batch":
                # Batch L2 computation - O(n) but vectorized, much faster than original
                distances = compute_l2_distance_batch(error_cpu, recycle_vars.y_error_buffer[(h, w)][grid_idx])
                most_similar_idx = torch.argmin(distances).item()
                recycle_vars.y_error_buffer[(h, w)][grid_idx][most_similar_idx] = error_cpu

            elif args.training_config.buffer_replacement_strategy == "l2_similarity":
                # Original L2 similarity method - O(n), slowest but most precise
                min_distance = float("inf")
                most_similar_idx = -1

                for j, stored_error in enumerate(recycle_vars.y_error_buffer[(h, w)][grid_idx]):
                    distance = compute_l2_distance(error_cpu, stored_error)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_idx = j

                if most_similar_idx != -1:
                    recycle_vars.y_error_buffer[(h, w)][grid_idx][most_similar_idx] = error_cpu


def update_error_buffers_distributed(
    args, recycle_vars, gathered_noise_errors, gathered_y_errors, gathered_timesteps, noisy_model_input
):
    """Update error buffers with samples gathered from all processes.
    Args:
        gathered_noise_errors: shape [num_gpus, batch_size, ...]
        gathered_y_errors: shape [num_gpus, batch_size, ...]
        gathered_timesteps: shape [num_gpus, batch_size]
    """
    num_gpus = gathered_noise_errors.shape[0]

    # Process each GPU's batch
    for gpu_idx in range(num_gpus):
        noise_error_batch = gathered_noise_errors[gpu_idx]  # [batch_size, ...]
        y_error_batch = gathered_y_errors[gpu_idx]  # [batch_size, ...]
        timestep_batch = gathered_timesteps[gpu_idx]  # [batch_size]

        # Add the entire batch to buffers
        add_error_to_latent_buffer(args, recycle_vars, noise_error_batch, timestep_batch, noisy_model_input)
        add_error_to_y_buffer(args, recycle_vars, y_error_batch, timestep_batch, noisy_model_input)


def update_error_buffers_local(args, recycle_vars, noise_error, y_error, timestep, noisy_model_input):
    """Update error buffers with samples from local GPU only (post-warmup).
    Args:
        noise_error: shape [batch_size, ...]
        y_error: shape [batch_size, ...]
        timestep: shape [batch_size] or scalar
    """
    add_error_to_latent_buffer(args, recycle_vars, noise_error, timestep, noisy_model_input)
    add_error_to_y_buffer(args, recycle_vars, y_error, timestep, noisy_model_input)


def process_and_update_error_buffers(
    args,
    recycle_vars,
    accelerator,
    global_step,
    noise_scheduler_copy,
    model_pred,
    target,
    timesteps,
    noisy_model_input,
    use_clean_input,
):
    x_0_pred = step_recycle(
        noise_scheduler_copy,
        model_pred,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=True,
    )
    noise_corr_gt = step_recycle(
        noise_scheduler_copy,
        target,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=True,
    )
    noise_error = x_0_pred - noise_corr_gt

    x_1_pred = step_recycle(
        noise_scheduler_copy,
        model_pred,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=False,
    )
    latent_corr_gt = step_recycle(
        noise_scheduler_copy,
        target,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=False,
    )
    y_error = x_1_pred - latent_corr_gt

    # Check if we're in warmup phase
    if global_step <= args.training_config.buffer_warmup_iter:

        def gather_with_optional_gpu_dim(tensor, keep_gpu_dim=False):
            gathered = accelerator.gather(tensor)

            if keep_gpu_dim:
                num_processes = accelerator.num_processes
                batch_size = tensor.shape[0]
                gathered = gathered.view(num_processes, batch_size, *gathered.shape[1:])

            return gathered

        # During warmup: gather errors and timesteps from all GPUs and update buffers
        gathered_noise_errors = gather_with_optional_gpu_dim(noise_error, keep_gpu_dim=True)
        gathered_y_errors = gather_with_optional_gpu_dim(y_error, keep_gpu_dim=True)
        gathered_timesteps = gather_with_optional_gpu_dim(timesteps, keep_gpu_dim=True)
        gathered_use_clean = gather_with_optional_gpu_dim(use_clean_input, keep_gpu_dim=True)
        # Shape: [num_gpus, batch_size]

        clean_mask = gathered_use_clean  # [num_gpus, batch_size]
        non_clean_mask = ~clean_mask  # [num_gpus, batch_size]
        num_gpus = gathered_noise_errors.shape[0]

        # Process clean samples: update with probability for each one
        if clean_mask.any():
            for gpu_idx in range(num_gpus):
                gpu_clean_mask = clean_mask[gpu_idx]
                if gpu_clean_mask.any():
                    p = random.random()
                    if p < args.training_config.clean_buffer_update_prob:
                        update_error_buffers_distributed(
                            args,
                            recycle_vars,
                            gathered_noise_errors[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            gathered_y_errors[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            gathered_timesteps[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            noisy_model_input,
                        )

        # Process non-clean samples: always update
        if non_clean_mask.any():
            for gpu_idx in range(num_gpus):
                gpu_non_clean_mask = non_clean_mask[gpu_idx]
                if gpu_non_clean_mask.any():
                    update_error_buffers_distributed(
                        args,
                        recycle_vars,
                        gathered_noise_errors[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        gathered_y_errors[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        gathered_timesteps[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        noisy_model_input,
                    )

    else:
        # After warmup: only use local GPU errors
        # Separate clean and non-clean samples
        clean_mask = use_clean_input  # Boolean tensor
        non_clean_mask = ~use_clean_input

        # Process clean samples: update with probability
        if clean_mask.any():
            p = random.random()
            if p < args.training_config.clean_buffer_update_prob:
                update_error_buffers_local(
                    args,
                    recycle_vars,
                    noise_error[clean_mask],
                    y_error[clean_mask],
                    timesteps[clean_mask],
                    noisy_model_input,
                )

        # Process non-clean samples: always update
        if non_clean_mask.any():
            update_error_buffers_local(
                args,
                recycle_vars,
                noise_error[non_clean_mask],
                y_error[non_clean_mask],
                timesteps[non_clean_mask],
                noisy_model_input,
            )
