import random

import numpy as np
import torch
from PIL import Image, ImageSequence
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from video_reader import PyVideoReader


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR


def clip_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC, antialias=False),
            CenterCrop(n_px),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def clip_transform_Image(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC, antialias=False),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def align_dimension(value, alignment=2):
    return int(round(value / alignment) * alignment)


def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    if video_path.endswith(".gif"):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGB")
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith(".png"):
        frame = Image.open(video_path)
        frame = frame.convert("RGB")
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith(".mp4"):
        vr = PyVideoReader(video_path, threads=0)
        if width is not None and height is not None:
            (_, original_height, original_width) = vr.get_shape()
            original_aspect_ratio = original_width / original_height
            if width > height:
                target_width = width
                target_height = int(width / original_aspect_ratio)
            else:
                target_height = height
                target_width = int(height * original_aspect_ratio)
            target_height = align_dimension(target_height, 2)
            target_width = align_dimension(target_width, 2)
        vr = PyVideoReader(video_path, target_height=target_height, target_width=target_width, threads=0)
        buffer = vr.decode()
        vr = None
        del vr
    else:
        raise NotImplementedError

    frames = buffer
    if num_frames and not video_path.endswith(".mp4"):
        frame_indices = get_frame_indices(num_frames, len(frames), sample="middle")
        frames = frames[frame_indices]

    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames


def read_frames_decord_by_fps(
    video_path,
    sample_fps=2,
    sample="rand",
    fix_start=None,
    max_num_frames=-1,
    trimmed30=False,
    num_frames=8,
    width=None,
    height=None,
):
    vr_info = PyVideoReader(video_path, threads=0)
    (vlen, original_height, original_width) = vr_info.get_shape()
    fps = vr_info.get_fps()
    duration = vlen / float(fps)
    vr_info = None
    del vr_info

    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    target_width = None
    target_height = None
    if width is not None and height is not None:
        original_aspect_ratio = original_width / original_height
        if width > height:
            target_width = width
            target_height = int(width / original_aspect_ratio)
        else:
            target_height = height
            target_width = int(height * original_aspect_ratio)
        target_height = align_dimension(target_height, 2)
        target_width = align_dimension(target_width, 2)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps, max_num_frames=max_num_frames
    )

    vr = PyVideoReader(video_path, target_height=target_height, target_width=target_width, threads=0)
    buffer = vr.decode()
    vr = None
    del vr

    frames = buffer[frame_indices]
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)

    frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    return frames


def load_video_frames(video_path, start_ratio=0.0, end_ratio=1.0, num_frames=8, height=384, width=640):
    # First pass: get video shape
    vr = PyVideoReader(video_path, threads=0)
    (total_frames, original_height, original_width) = vr.get_shape()

    # Calculate target dimensions maintaining aspect ratio
    original_aspect_ratio = original_width / original_height
    if width > height:
        target_width = width
        target_height = int(width / original_aspect_ratio)
    else:
        target_height = height
        target_width = int(height * original_aspect_ratio)

    target_height = align_dimension(target_height, 2)
    target_width = align_dimension(target_width, 2)

    # Calculate frame range
    start_frame = int(total_frames * start_ratio)
    end_frame = int(total_frames * end_ratio)
    portion_length = end_frame - start_frame

    if portion_length < num_frames:
        # Expand the range to accommodate num_frames
        needed_frames = num_frames - portion_length
        expansion = needed_frames / 2

        # Try to expand symmetrically
        new_start = max(0, start_frame - int(np.ceil(expansion)))
        new_end = min(total_frames, end_frame + int(np.floor(expansion)))

        # If still not enough, expand further in available direction
        if new_end - new_start < num_frames:
            if new_start == 0:
                new_end = min(total_frames, new_start + num_frames)
            elif new_end == total_frames:
                new_start = max(0, new_end - num_frames)

        start_frame = new_start
        end_frame = new_end
        portion_length = end_frame - start_frame

        # Now sample frames
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    else:
        # Sample uniformly from the portion
        step = portion_length / num_frames
        frame_indices = [int(start_frame + i * step) for i in range(num_frames)]

    # Ensure indices are within bounds
    frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

    # Second pass: decode only needed frames with target dimensions
    vr = PyVideoReader(video_path, target_height=target_height, target_width=target_width, threads=0)
    frames = vr.get_batch(frame_indices)  # Only decode needed frames (num_frames, H, W, C)

    # Convert to tensor if needed and permute to (T, C, H, W)
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

    # Clean up
    vr = None
    del vr

    return frames


def extract_video_segment(input_path, output_path, start_ratio, end_ratio):
    """
    尽可能保持原视频编码参数
    """
    import ffmpeg

    # 获取原视频信息
    probe = ffmpeg.probe(input_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

    duration = float(probe["format"]["duration"])
    start_time = duration * start_ratio
    segment_duration = duration * (end_ratio - start_ratio)

    # 检测原视频编码参数
    orig_codec = video_stream.get("codec_name", "h264")
    orig_pix_fmt = video_stream.get("pix_fmt", "yuv420p")

    # 如果原视频是 h264/h265,使用相同编码器
    if orig_codec in ["h264", "hevc"]:
        codec_name = "libx264" if orig_codec == "h264" else "libx265"
    else:
        codec_name = "libx264"  # fallback

    (
        ffmpeg.input(input_path, ss=start_time)
        .output(
            output_path,
            t=segment_duration,
            vcodec=codec_name,
            crf=0,
            preset="medium",
            pix_fmt=orig_pix_fmt,
            acodec="copy",
            vsync="cfr",
            map_metadata=0,
        )
        .overwrite_output()
        .run(quiet=True)
    )
