import os
import re

import gradio as gr


def parse_video_name(filename):
    """Parse video filename to extract step and index"""
    # Match checkpoint-{step}_{idx}.mp4 format
    match = re.match(r"checkpoint-(\d+)_(\d+)\.mp4$", filename)
    if match:
        step = int(match.group(1))
        idx = int(match.group(2))
        return step, idx
    return None, None


def get_video_list(folder_path):
    """Get all mp4 videos from folder"""
    if not os.path.exists(folder_path):
        return []

    videos = []
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            step, idx = parse_video_name(file)
            if step is not None:
                videos.append({"filename": file, "step": step, "idx": idx, "path": os.path.join(folder_path, file)})

    # Sort by step and idx
    videos.sort(key=lambda x: (x["step"], x["idx"]))
    return videos


def create_video_mapping(videos):
    """Create (step, idx) -> filename mapping"""
    mapping = {}
    for video in videos:
        key = (video["step"], video["idx"])
        mapping[key] = video["filename"]
    return mapping


def get_step_idx_mapping(common_keys):
    """Extract step and idx mapping from common (step, idx) keys"""
    step_idx_map = {}  # {step: [idx1, idx2, ...]}
    all_steps = set()
    all_indices = set()

    for step, idx in common_keys:
        all_steps.add(step)
        all_indices.add(idx)
        if step not in step_idx_map:
            step_idx_map[step] = []
        step_idx_map[step].append(idx)

    # Sort
    for step in step_idx_map:
        step_idx_map[step].sort()

    return sorted(all_steps), sorted(all_indices), step_idx_map


def load_videos(folder1, folder2):
    """Load videos from both folders and match them"""
    if not folder1 or not folder2:
        return (
            None,
            None,
            "Please enter both folder paths",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
            {},
        )

    videos1 = get_video_list(folder1)
    videos2 = get_video_list(folder2)

    if not videos1:
        return (
            None,
            None,
            f"No video files found in folder 1 (total {len(os.listdir(folder1)) if os.path.exists(folder1) else 0} files)",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
            {},
        )
    if not videos2:
        return (
            None,
            None,
            f"No video files found in folder 2 (total {len(os.listdir(folder2)) if os.path.exists(folder2) else 0} files)",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
            {},
        )

    # Create (step, idx) to filename mapping
    video_map1 = create_video_mapping(videos1)
    video_map2 = create_video_mapping(videos2)

    # Find common (step, idx) combinations
    common_keys = sorted(set(video_map1.keys()) & set(video_map2.keys()))

    if not common_keys:
        # Show detailed information for debugging
        steps1 = {v["step"] for v in videos1}
        steps2 = {v["step"] for v in videos2}
        info = "No matching videos found in both folders\n"
        info += f"Folder 1: {len(videos1)} videos found\n"
        info += f"Folder 2: {len(videos2)} videos found\n"
        info += f"Folder 1 steps: {sorted(steps1)}\n"
        info += f"Folder 2 steps: {sorted(steps2)}\n"
        info += f"Common steps: {sorted(steps1 & steps2)}"
        return (
            None,
            None,
            info,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
            {},
        )

    # Get all steps and indices
    all_steps, all_indices, step_idx_map = get_step_idx_mapping(common_keys)

    # Load first video
    first_key = common_keys[0]
    first_step, first_idx = first_key

    filename1 = video_map1[first_key]
    filename2 = video_map2[first_key]

    video1_path = os.path.join(folder1, filename1)
    video2_path = os.path.join(folder2, filename2)

    info = f"Found {len(common_keys)} matching video pairs\n"
    info += f"Folder 1: {len(videos1)} videos\n"
    info += f"Folder 2: {len(videos2)} videos\n"
    info += f"Current: Step {first_step}, Index {first_idx}\n"
    info += f"File 1: {filename1}\n"
    info += f"File 2: {filename2}"

    # Get available indices for current step
    available_indices = step_idx_map.get(first_step, [])

    progress = f"1 / {len(common_keys)}"

    return (
        video1_path,
        video2_path,
        info,
        gr.update(choices=all_steps, value=first_step),
        gr.update(choices=available_indices, value=first_idx),
        gr.update(interactive=first_step > all_steps[0]),
        gr.update(interactive=first_step < all_steps[-1]),
        gr.update(interactive=first_idx > available_indices[0] if available_indices else False),
        gr.update(interactive=first_idx < available_indices[-1] if available_indices else False),
        progress,
        video_map1,
        video_map2,
        step_idx_map,
    )


def update_available_indices(selected_step, step_idx_map):
    """Update available index list"""
    if not step_idx_map or selected_step is None:
        return gr.update(choices=[], value=None)

    available_indices = step_idx_map.get(selected_step, [])
    first_idx = available_indices[0] if available_indices else None

    return gr.update(choices=available_indices, value=first_idx)


def update_videos_from_selectors(folder1, folder2, selected_step, selected_idx, video_map1, video_map2, step_idx_map):
    """Update videos based on selected step and idx"""
    if selected_step is None or selected_idx is None:
        return None, None, "Please select step and index", gr.update(), gr.update(), gr.update(), gr.update(), ""

    key = (selected_step, selected_idx)

    if key not in video_map1 or key not in video_map2:
        return (
            None,
            None,
            f"Video not found for Step {selected_step}, Index {selected_idx}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    filename1 = video_map1[key]
    filename2 = video_map2[key]

    video1_path = os.path.join(folder1, filename1)
    video2_path = os.path.join(folder2, filename2)

    info = f"Current: Step {selected_step}, Index {selected_idx}\n"
    info += f"File 1: {filename1}\n"
    info += f"File 2: {filename2}"

    # Get all steps and indices for current step
    all_steps = sorted(step_idx_map.keys())
    available_indices = step_idx_map.get(selected_step, [])

    # Update button states
    prev_step_interactive = selected_step > all_steps[0]
    next_step_interactive = selected_step < all_steps[-1]
    prev_idx_interactive = selected_idx > available_indices[0] if available_indices else False
    next_idx_interactive = selected_idx < available_indices[-1] if available_indices else False

    # Calculate current video position
    all_keys = sorted(set(video_map1.keys()) & set(video_map2.keys()))
    current_idx = all_keys.index(key) + 1
    progress = f"{current_idx} / {len(all_keys)}"

    return (
        video1_path,
        video2_path,
        info,
        gr.update(interactive=prev_step_interactive),
        gr.update(interactive=next_step_interactive),
        gr.update(interactive=prev_idx_interactive),
        gr.update(interactive=next_idx_interactive),
        progress,
    )


def navigate_step(folder1, folder2, current_step, current_idx, video_map1, video_map2, step_idx_map, direction):
    """Navigate to previous or next step"""
    if not step_idx_map or current_step is None:
        return (
            None,
            None,
            "Please load videos first",
            current_step,
            current_idx,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    all_steps = sorted(step_idx_map.keys())
    current_step_idx = all_steps.index(current_step)

    if direction == "prev":
        new_step_idx = max(0, current_step_idx - 1)
    else:  # next
        new_step_idx = min(len(all_steps) - 1, current_step_idx + 1)

    new_step = all_steps[new_step_idx]

    # Get first available index for new step
    available_indices = step_idx_map.get(new_step, [])
    new_idx = available_indices[0] if available_indices else current_idx

    return update_videos_from_selectors(folder1, folder2, new_step, new_idx, video_map1, video_map2, step_idx_map) + (
        new_step,
        new_idx,
    )


def navigate_idx(folder1, folder2, current_step, current_idx, video_map1, video_map2, step_idx_map, direction):
    """Navigate to previous or next index"""
    if not step_idx_map or current_step is None or current_idx is None:
        return (
            None,
            None,
            "Please load videos first",
            current_step,
            current_idx,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    available_indices = step_idx_map.get(current_step, [])
    if not available_indices or current_idx not in available_indices:
        return (
            None,
            None,
            "Index not in list",
            current_step,
            current_idx,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    current_idx_pos = available_indices.index(current_idx)

    if direction == "prev":
        new_idx_pos = max(0, current_idx_pos - 1)
    else:  # next
        new_idx_pos = min(len(available_indices) - 1, current_idx_pos + 1)

    new_idx = available_indices[new_idx_pos]

    return update_videos_from_selectors(
        folder1, folder2, current_step, new_idx, video_map1, video_map2, step_idx_map
    ) + (current_step, new_idx)


# Create Gradio interface
with gr.Blocks(title="Video Comparison Tool") as demo:
    gr.Markdown("# Video Comparison Tool")
    gr.Markdown(
        "Enter two folder paths to automatically match and compare checkpoint-{step}_{idx}.mp4 format video files"
    )

    # Store state
    video_map1_state = gr.State({})
    video_map2_state = gr.State({})
    step_idx_map_state = gr.State({})

    with gr.Row():
        folder1_input = gr.Textbox(label="Folder 1 Path", placeholder="/path/to/folder1", scale=2)
        folder2_input = gr.Textbox(label="Folder 2 Path", placeholder="/path/to/folder2", scale=2)

    load_btn = gr.Button("Load Videos", variant="primary")

    info_text = gr.Textbox(label="Information", interactive=False, lines=6)

    # Step navigation controls
    with gr.Row():
        prev_step_btn = gr.Button("⬅️ Previous Step", interactive=False, scale=1)
        step_selector = gr.Dropdown(label="Select Step", choices=[], interactive=True, scale=2)
        next_step_btn = gr.Button("Next Step ➡️", interactive=False, scale=1)

    # Index navigation controls
    with gr.Row():
        prev_idx_btn = gr.Button("⬅️ Previous Index", interactive=False, scale=1)
        idx_selector = gr.Dropdown(label="Select Index", choices=[], interactive=True, scale=2)
        next_idx_btn = gr.Button("Next Index ➡️", interactive=False, scale=1)

    progress_text = gr.Textbox(label="Progress", value="0 / 0", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Folder 1")
            video1 = gr.Video(label="Video 1", autoplay=True, loop=True)

        with gr.Column():
            gr.Markdown("### Folder 2")
            video2 = gr.Video(label="Video 2", autoplay=True, loop=True)

    # Event bindings
    load_btn.click(
        fn=load_videos,
        inputs=[folder1_input, folder2_input],
        outputs=[
            video1,
            video2,
            info_text,
            step_selector,
            idx_selector,
            prev_step_btn,
            next_step_btn,
            prev_idx_btn,
            next_idx_btn,
            progress_text,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
    )

    # When step changes, update available indices
    step_selector.change(
        fn=update_available_indices, inputs=[step_selector, step_idx_map_state], outputs=[idx_selector]
    ).then(
        fn=update_videos_from_selectors,
        inputs=[
            folder1_input,
            folder2_input,
            step_selector,
            idx_selector,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
        outputs=[video1, video2, info_text, prev_step_btn, next_step_btn, prev_idx_btn, next_idx_btn, progress_text],
    )

    # When index changes, update videos
    idx_selector.change(
        fn=update_videos_from_selectors,
        inputs=[
            folder1_input,
            folder2_input,
            step_selector,
            idx_selector,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
        outputs=[video1, video2, info_text, prev_step_btn, next_step_btn, prev_idx_btn, next_idx_btn, progress_text],
    )

    # Step navigation buttons
    prev_step_btn.click(
        fn=lambda f1, f2, s, i, vm1, vm2, sim: navigate_step(f1, f2, s, i, vm1, vm2, sim, "prev"),
        inputs=[
            folder1_input,
            folder2_input,
            step_selector,
            idx_selector,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
        outputs=[
            video1,
            video2,
            info_text,
            prev_step_btn,
            next_step_btn,
            prev_idx_btn,
            next_idx_btn,
            progress_text,
            step_selector,
            idx_selector,
        ],
    )

    next_step_btn.click(
        fn=lambda f1, f2, s, i, vm1, vm2, sim: navigate_step(f1, f2, s, i, vm1, vm2, sim, "next"),
        inputs=[
            folder1_input,
            folder2_input,
            step_selector,
            idx_selector,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
        outputs=[
            video1,
            video2,
            info_text,
            prev_step_btn,
            next_step_btn,
            prev_idx_btn,
            next_idx_btn,
            progress_text,
            step_selector,
            idx_selector,
        ],
    )

    # Index navigation buttons
    prev_idx_btn.click(
        fn=lambda f1, f2, s, i, vm1, vm2, sim: navigate_idx(f1, f2, s, i, vm1, vm2, sim, "prev"),
        inputs=[
            folder1_input,
            folder2_input,
            step_selector,
            idx_selector,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
        outputs=[
            video1,
            video2,
            info_text,
            prev_step_btn,
            next_step_btn,
            prev_idx_btn,
            next_idx_btn,
            progress_text,
            step_selector,
            idx_selector,
        ],
    )

    next_idx_btn.click(
        fn=lambda f1, f2, s, i, vm1, vm2, sim: navigate_idx(f1, f2, s, i, vm1, vm2, sim, "next"),
        inputs=[
            folder1_input,
            folder2_input,
            step_selector,
            idx_selector,
            video_map1_state,
            video_map2_state,
            step_idx_map_state,
        ],
        outputs=[
            video1,
            video2,
            info_text,
            prev_step_btn,
            next_step_btn,
            prev_idx_btn,
            next_idx_btn,
            progress_text,
            step_selector,
            idx_selector,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        allowed_paths=[
            "0_ablation_videos",
            "ablation_stage3_1_warmup",
        ],
    )
