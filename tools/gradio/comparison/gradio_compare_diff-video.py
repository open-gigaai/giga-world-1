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


def get_step_idx_info(video_map):
    """Extract step and idx information from video mapping"""
    all_steps = set()
    all_indices = set()
    idx_step_map = {}  # {idx: [step1, step2, ...]}

    for step, idx in video_map.keys():
        all_steps.add(step)
        all_indices.add(idx)
        if idx not in idx_step_map:
            idx_step_map[idx] = []
        idx_step_map[idx].append(step)

    # Sort
    for idx in idx_step_map:
        idx_step_map[idx].sort()

    return sorted(all_steps), sorted(all_indices), idx_step_map


def load_videos(folder_path):
    """Load videos from folder"""
    if not folder_path:
        return (
            None,
            None,
            "Please enter folder path",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
        )

    videos = get_video_list(folder_path)

    if not videos:
        return (
            None,
            None,
            f"No video files found in folder ({len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0} files total)",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
        )

    # Create (step, idx) to filename mapping
    video_map = create_video_mapping(videos)

    # Get all step and index information
    all_steps, all_indices, idx_step_map = get_step_idx_info(video_map)

    # Filter indices with at least 2 steps
    valid_indices = [idx for idx in all_indices if len(idx_step_map[idx]) >= 2]

    if not valid_indices:
        info = (
            f"Found {len(videos)} videos, but no comparable videos (need at least 2 different steps for same index)\n"
        )
        info += f"Steps: {all_steps}\n"
        info += f"Indices: {all_indices}"
        return (
            None,
            None,
            info,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "0 / 0",
            {},
            {},
        )

    # Select first valid index and its first two steps
    first_idx = valid_indices[0]
    available_steps = idx_step_map[first_idx]
    step1 = available_steps[0]
    step2 = available_steps[1] if len(available_steps) > 1 else available_steps[0]

    # Load videos
    filename1 = video_map.get((step1, first_idx))
    filename2 = video_map.get((step2, first_idx))

    video1_path = os.path.join(folder_path, filename1) if filename1 else None
    video2_path = os.path.join(folder_path, filename2) if filename2 else None

    info = f"Found {len(videos)} videos, {len(valid_indices)} comparable indices\n"
    info += f"Current Index: {first_idx}\n"
    info += f"Step1: {step1} - {filename1}\n"
    info += f"Step2: {step2} - {filename2}"

    progress = f"1 / {len(valid_indices)}"

    return (
        video1_path,
        video2_path,
        info,
        gr.update(choices=valid_indices, value=first_idx),
        gr.update(choices=available_steps, value=step1),
        gr.update(choices=available_steps, value=step2),
        gr.update(interactive=first_idx > valid_indices[0]),
        gr.update(interactive=first_idx < valid_indices[-1]),
        gr.update(interactive=True),
        gr.update(interactive=True),
        progress,
        video_map,
        idx_step_map,
    )


def update_videos(folder_path, selected_idx, selected_step1, selected_step2, video_map, idx_step_map):
    """Update videos based on selected idx and two steps"""
    if selected_idx is None or selected_step1 is None or selected_step2 is None:
        return None, None, "Please select index and steps", gr.update(), gr.update(), gr.update(), gr.update(), ""

    key1 = (selected_step1, selected_idx)
    key2 = (selected_step2, selected_idx)

    filename1 = video_map.get(key1)
    filename2 = video_map.get(key2)

    if not filename1 or not filename2:
        return (
            None,
            None,
            f"Complete video pair not found: Index {selected_idx}, Step1 {selected_step1}, Step2 {selected_step2}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    video1_path = os.path.join(folder_path, filename1)
    video2_path = os.path.join(folder_path, filename2)

    info = f"Current Index: {selected_idx}\n"
    info += f"Step1: {selected_step1} - {filename1}\n"
    info += f"Step2: {selected_step2} - {filename2}"

    # Get all valid indices
    all_indices = [idx for idx in idx_step_map.keys() if len(idx_step_map[idx]) >= 2]
    all_indices.sort()

    # Update button states
    prev_idx_interactive = selected_idx > all_indices[0] if all_indices else False
    next_idx_interactive = selected_idx < all_indices[-1] if all_indices else False

    # Calculate progress
    current_pos = all_indices.index(selected_idx) + 1 if selected_idx in all_indices else 0
    progress = f"{current_pos} / {len(all_indices)}"

    return (
        video1_path,
        video2_path,
        info,
        gr.update(interactive=prev_idx_interactive),
        gr.update(interactive=next_idx_interactive),
        gr.update(),
        gr.update(),
        progress,
    )


def update_available_steps(selected_idx, idx_step_map):
    """Update available steps list for current index"""
    if not idx_step_map or selected_idx is None:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    available_steps = idx_step_map.get(selected_idx, [])
    first_step = available_steps[0] if available_steps else None
    second_step = available_steps[1] if len(available_steps) > 1 else first_step

    return (
        gr.update(choices=available_steps, value=first_step),
        gr.update(choices=available_steps, value=second_step),
    )


def navigate_idx(folder_path, current_idx, step1, step2, video_map, idx_step_map, direction):
    """Navigate to previous or next index"""
    if not idx_step_map or current_idx is None:
        return (
            None,
            None,
            "Please load videos first",
            current_idx,
            step1,
            step2,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    # Get all valid indices
    all_indices = [idx for idx in idx_step_map.keys() if len(idx_step_map[idx]) >= 2]
    all_indices.sort()

    if current_idx not in all_indices:
        return (
            None,
            None,
            "Current Index invalid",
            current_idx,
            step1,
            step2,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    current_idx_pos = all_indices.index(current_idx)

    if direction == "prev":
        new_idx_pos = max(0, current_idx_pos - 1)
    else:  # next
        new_idx_pos = min(len(all_indices) - 1, current_idx_pos + 1)

    new_idx = all_indices[new_idx_pos]

    # Get available steps for new index
    available_steps = idx_step_map.get(new_idx, [])
    new_step1 = available_steps[0] if available_steps else step1
    new_step2 = available_steps[1] if len(available_steps) > 1 else available_steps[0]

    result = update_videos(folder_path, new_idx, new_step1, new_step2, video_map, idx_step_map)
    return result + (new_idx, new_step1, new_step2)


# Create Gradio interface
with gr.Blocks(title="Video Comparison Tool - Different Step Comparison") as demo:
    gr.Markdown("# Video Comparison Tool - Different Step Comparison")
    gr.Markdown(
        "Enter folder path to compare videos of same index at different steps (checkpoint-{step}_{idx}.mp4 format)"
    )

    # Store state
    video_map_state = gr.State({})
    idx_step_map_state = gr.State({})

    folder_input = gr.Textbox(label="Folder Path", placeholder="/path/to/folder", scale=2)

    load_btn = gr.Button("Load Videos", variant="primary")

    info_text = gr.Textbox(label="Information", interactive=False, lines=5)

    # Index navigation controls
    with gr.Row():
        prev_idx_btn = gr.Button("⬅️ Previous Index", interactive=False, scale=1)
        idx_selector = gr.Dropdown(label="Select Index", choices=[], interactive=True, scale=2)
        next_idx_btn = gr.Button("Next Index ➡️", interactive=False, scale=1)

    # Step selectors
    with gr.Row():
        step1_selector = gr.Dropdown(label="Select Step1 (Left)", choices=[], interactive=True, scale=1)
        step2_selector = gr.Dropdown(label="Select Step2 (Right)", choices=[], interactive=True, scale=1)

    progress_text = gr.Textbox(label="Progress", value="0 / 0", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 1")
            video1 = gr.Video(label="Video 1", autoplay=True, loop=True)

        with gr.Column():
            gr.Markdown("### Step 2")
            video2 = gr.Video(label="Video 2", autoplay=True, loop=True)

    # Event binding
    load_btn.click(
        fn=load_videos,
        inputs=[folder_input],
        outputs=[
            video1,
            video2,
            info_text,
            idx_selector,
            step1_selector,
            step2_selector,
            prev_idx_btn,
            next_idx_btn,
            gr.State(),
            gr.State(),
            progress_text,
            video_map_state,
            idx_step_map_state,
        ],
    )

    # When index changes, update available steps and videos
    def handle_idx_change(folder_path, selected_idx, video_map, idx_step_map):
        """Handle index change - update steps and videos together"""
        if not idx_step_map or selected_idx is None:
            return (
                None,
                None,
                "Please select index",
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(),
                gr.update(),
                "",
            )

        # Get available steps for new index
        available_steps = idx_step_map.get(selected_idx, [])
        new_step1 = available_steps[0] if available_steps else None
        new_step2 = available_steps[1] if len(available_steps) > 1 else available_steps[0]

        # Update videos with new steps
        result = update_videos(folder_path, selected_idx, new_step1, new_step2, video_map, idx_step_map)

        return (
            result[0],  # video1
            result[1],  # video2
            result[2],  # info
            gr.update(choices=available_steps, value=new_step1),  # step1_selector
            gr.update(choices=available_steps, value=new_step2),  # step2_selector
            result[3],  # prev_idx_btn
            result[4],  # next_idx_btn
            result[7],  # progress
        )

    idx_selector.change(
        fn=handle_idx_change,
        inputs=[folder_input, idx_selector, video_map_state, idx_step_map_state],
        outputs=[video1, video2, info_text, step1_selector, step2_selector, prev_idx_btn, next_idx_btn, progress_text],
    )

    step1_selector.select(
        fn=update_videos,
        inputs=[folder_input, idx_selector, step1_selector, step2_selector, video_map_state, idx_step_map_state],
        outputs=[video1, video2, info_text, prev_idx_btn, next_idx_btn, gr.State(), gr.State(), progress_text],
    )

    step2_selector.select(
        fn=update_videos,
        inputs=[folder_input, idx_selector, step1_selector, step2_selector, video_map_state, idx_step_map_state],
        outputs=[video1, video2, info_text, prev_idx_btn, next_idx_btn, gr.State(), gr.State(), progress_text],
    )

    # Index navigation buttons
    prev_idx_btn.click(
        fn=lambda f, i, s1, s2, vm, ism: navigate_idx(f, i, s1, s2, vm, ism, "prev"),
        inputs=[folder_input, idx_selector, step1_selector, step2_selector, video_map_state, idx_step_map_state],
        outputs=[
            video1,
            video2,
            info_text,
            prev_idx_btn,
            next_idx_btn,
            gr.State(),
            gr.State(),
            progress_text,
            idx_selector,
            step1_selector,
            step2_selector,
        ],
    )

    next_idx_btn.click(
        fn=lambda f, i, s1, s2, vm, ism: navigate_idx(f, i, s1, s2, vm, ism, "next"),
        inputs=[folder_input, idx_selector, step1_selector, step2_selector, video_map_state, idx_step_map_state],
        outputs=[
            video1,
            video2,
            info_text,
            prev_idx_btn,
            next_idx_btn,
            gr.State(),
            gr.State(),
            progress_text,
            idx_selector,
            step1_selector,
            step2_selector,
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
