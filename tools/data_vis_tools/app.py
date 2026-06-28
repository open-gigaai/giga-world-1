#!/usr/bin/env python3
"""
Bimanual URDF/STL 3D action animation viewer.

Run example:
  python tools/data_vis_tools/app.py --host 0.0.0.0 --port 8090

Open browser:
  http://127.0.0.1:8090
"""
import argparse
import base64
import importlib.util
import json
import math
import mimetypes
import os
import pickle
import sys
import traceback
import xml.etree.ElementTree as ET
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
PIPER_DIR = os.path.join(APP_DIR, "piper")
TOOLS_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
DEFAULT_URDF_PATH = os.path.join(PIPER_DIR, "piper.urdf")
DEFAULT_DATA_PKL = os.path.join(PROJECT_ROOT, "example", "toy_datapipeline_dataset", "labels", "data.pkl")
DEFAULT_LEFT_BASE = [0.0, 0.325, 0.0]
DEFAULT_RIGHT_BASE = [0.0, -0.30, 0.0]


def resolve_path(path, base=PROJECT_ROOT):
    """Resolve a path that may be absolute or relative-to-project.

    Absolute paths are returned unchanged; relative paths are resolved
    against ``base`` (defaults to PROJECT_ROOT).
    """
    if not path:
        return path
    p = os.path.expanduser(str(path))
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(base, p))


def resolve_against_pkl(path, pkl_abs):
    """Resolve a video path stored in the pkl against the pkl's directory."""
    if not path:
        return path
    p = os.path.expanduser(str(path))
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(os.path.dirname(pkl_abs), p))

DATA_CACHE = {}
ROBOT_CACHE = {}
CAM_CALIB_MODULE = None


def get_cam_calib_module():
    global CAM_CALIB_MODULE
    if CAM_CALIB_MODULE is None:
        if TOOLS_DIR not in sys.path:
            sys.path.insert(0, TOOLS_DIR)
        cam_app_path = os.path.join(TOOLS_DIR, "cam_calib_tools", "app.py")
        spec = importlib.util.spec_from_file_location("giga_cam_calib_tools_app", cam_app_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        CAM_CALIB_MODULE = module
    return CAM_CALIB_MODULE


# ===================== SE3 / URDF utility functions (standalone, no modification to original code) =====================
def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_to_R(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(float(theta)), math.sin(float(theta))
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def make_T(xyz=None, rpy=None):
    T = np.eye(4, dtype=np.float64)
    if xyz is not None:
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    if rpy is not None:
        T[:3, :3] = rpy_to_R(rpy)
    return T


def joint_motion_T(joint_type, axis, q):
    T = np.eye(4, dtype=np.float64)
    if joint_type in ("revolute", "continuous"):
        T[:3, :3] = axis_angle_to_R(axis, q)
    elif joint_type == "prismatic":
        T[:3, 3] = np.asarray(axis, dtype=np.float64) * float(q)
    return T


def mat4_to_list(T):
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()


def parse_vec(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.strip().split()], dtype=np.float64)


def resolve_mesh_path(urdf_path, filename):
    if not filename:
        return None
    if filename.startswith("package://"):
        filename = filename[len("package://"):]
        parts = filename.split("/", 1)
        rel = parts[1] if len(parts) == 2 else parts[0]
        candidates = [
            os.path.join(os.path.dirname(urdf_path), rel),
            os.path.join(os.path.dirname(os.path.dirname(urdf_path)), rel),
            os.path.join(PIPER_DIR, rel),
            os.path.join(TOOLS_DIR, rel),
        ]
    elif os.path.isabs(filename):
        candidates = [filename]
    else:
        candidates = [os.path.normpath(os.path.join(os.path.dirname(urdf_path), filename))]

    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.abspath(candidates[0])


def prefer_stl(mesh_path):
    if mesh_path is None:
        return None
    root, _ = os.path.splitext(mesh_path)
    for ext in (".STL", ".stl"):
        p = root + ext
        if os.path.exists(p):
            return p
    return mesh_path if os.path.splitext(mesh_path)[1].lower() == ".stl" else None


class URDFRobotModel:
    def __init__(self, urdf_path):
        self.urdf_path = os.path.abspath(urdf_path)
        self.links = {}
        self.joints = []
        self.child_to_joint = {}
        self.parent_to_joints = {}
        self._parse()

    def _parse_mesh_item(self, node):
        origin = node.find("origin")
        xyz = parse_vec(origin.attrib.get("xyz"), [0, 0, 0]) if origin is not None else np.zeros(3)
        rpy = parse_vec(origin.attrib.get("rpy"), [0, 0, 0]) if origin is not None else np.zeros(3)
        mesh = node.find("geometry/mesh")
        if mesh is None:
            return None
        mesh_path = resolve_mesh_path(self.urdf_path, mesh.attrib.get("filename"))
        mesh_path = prefer_stl(mesh_path)
        if mesh_path is None or not os.path.exists(mesh_path):
            return None
        scale = parse_vec(mesh.attrib.get("scale"), [1, 1, 1]).tolist()
        return {"path": mesh_path, "origin_T": make_T(xyz, rpy), "scale": scale}

    def _parse_color(self, visual_node):
        color_node = visual_node.find("material/color")
        if color_node is None:
            return [0.78, 0.82, 0.93, 1.0]
        rgba = parse_vec(color_node.attrib.get("rgba"), [0.78, 0.82, 0.93, 1.0])
        if len(rgba) == 3:
            rgba = np.concatenate([rgba, [1.0]])
        return rgba.tolist()

    def _parse(self):
        root = ET.parse(self.urdf_path).getroot()
        for link in root.findall("link"):
            name = link.attrib["name"]
            visuals = []
            for visual in link.findall("visual"):
                item = self._parse_mesh_item(visual)
                if item is not None:
                    item["color"] = self._parse_color(visual)
                    visuals.append(item)
            if not visuals:
                for collision in link.findall("collision"):
                    item = self._parse_mesh_item(collision)
                    if item is not None:
                        item["color"] = [0.78, 0.82, 0.93, 1.0]
                        visuals.append(item)
            self.links[name] = {"name": name, "visuals": visuals}

        for j in root.findall("joint"):
            parent = j.find("parent")
            child = j.find("child")
            if parent is None or child is None:
                continue
            origin = j.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz"), [0, 0, 0]) if origin is not None else np.zeros(3)
            rpy = parse_vec(origin.attrib.get("rpy"), [0, 0, 0]) if origin is not None else np.zeros(3)
            axis_node = j.find("axis")
            axis = parse_vec(axis_node.attrib.get("xyz"), [1, 0, 0]) if axis_node is not None else np.array([1, 0, 0], dtype=np.float64)
            item = {
                "name": j.attrib["name"],
                "type": j.attrib.get("type", "fixed"),
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
                "origin_T": make_T(xyz, rpy),
                "axis": axis,
            }
            self.joints.append(item)
            self.child_to_joint[item["child"]] = item
            self.parent_to_joints.setdefault(item["parent"], []).append(item)

    def guess_base_link(self):
        child_links = set(self.child_to_joint.keys())
        roots = [name for name in self.links if name not in child_links]
        if "base_link" in self.links:
            return "base_link"
        return roots[0] if roots else next(iter(self.links))

    def guess_ee_link(self, base_link):
        best = (0, base_link)
        def dfs(link, depth):
            nonlocal best
            if depth > best[0]:
                best = (depth, link)
            for joint in self.parent_to_joints.get(link, []):
                dfs(joint["child"], depth + 1)
        dfs(base_link, 0)
        return best[1]

    def find_chain(self, base_link="base_link", ee_link="link6"):
        if base_link not in self.links:
            base_link = self.guess_base_link()
        if ee_link not in self.links:
            ee_link = self.guess_ee_link(base_link)
        chain = []
        cur = ee_link
        while cur != base_link:
            if cur not in self.child_to_joint:
                raise ValueError(f"Cannot trace from {ee_link} back to {base_link}, break point at link={cur}")
            joint = self.child_to_joint[cur]
            chain.append(joint)
            cur = joint["parent"]
        chain.reverse()
        return chain, base_link, ee_link

    def fk(self, qpos, base_T, base_link="base_link", ee_link="link6"):
        chain, base_link, ee_link = self.find_chain(base_link, ee_link)
        movable = [j for j in chain if j["type"] in ("revolute", "continuous", "prismatic")]
        qmap = {j["name"]: float(qpos[i]) for i, j in enumerate(movable[:len(qpos)])}
        T = np.asarray(base_T, dtype=np.float64).copy()
        poses = {base_link: T.copy()}
        for joint in chain:
            T = T @ joint["origin_T"]
            if joint["type"] in ("revolute", "continuous", "prismatic"):
                T = T @ joint_motion_T(joint["type"], joint["axis"], qmap.get(joint["name"], 0.0))
            poses[joint["child"]] = T.copy()
        return poses, chain, base_link, ee_link

    def to_client_model(self, base_link="base_link", ee_link="link6"):
        chain, base_link, ee_link = self.find_chain(base_link, ee_link)
        used_links = [base_link] + [j["child"] for j in chain]
        links = []
        for name in used_links:
            link = self.links.get(name, {"visuals": []})
            visuals = []
            for item in link["visuals"]:
                visuals.append({
                    "url": "/mesh?path=" + quote(item["path"]),
                    "origin_T": mat4_to_list(item["origin_T"]),
                    "scale": item["scale"],
                    "color": item["color"],
                })
            links.append({"name": name, "visuals": visuals})
        joints = []
        for joint in chain:
            joints.append({
                "name": joint["name"],
                "type": joint["type"],
                "parent": joint["parent"],
                "child": joint["child"],
                "origin_T": mat4_to_list(joint["origin_T"]),
                "axis": np.asarray(joint["axis"], dtype=np.float64).tolist(),
            })
        return {"base_link": base_link, "ee_link": ee_link, "links": links, "joints": joints}


def get_robot(urdf_path):
    urdf_path = os.path.abspath(urdf_path)
    if urdf_path not in ROBOT_CACHE:
        ROBOT_CACHE[urdf_path] = URDFRobotModel(urdf_path)
    return ROBOT_CACHE[urdf_path]


def safe_pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(path):
    path = resolve_path(path)
    if path not in DATA_CACHE:
        data = safe_pickle_load(path)
        if not isinstance(data, list):
            raise ValueError("Top-level of data pkl must be a list, each element is one episode")
        # Resolve any relative video paths against the pkl directory so
        # downstream code (cv2.VideoCapture etc.) can open them directly.
        pkl_dir = os.path.dirname(path)
        for ep in data:
            if not isinstance(ep, dict):
                continue
            for k, v in list(ep.items()):
                if isinstance(v, str) and v and not os.path.isabs(v) and \
                        any(v.lower().endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png")):
                    ep[k] = os.path.abspath(os.path.join(pkl_dir, v))
        DATA_CACHE[path] = data
    return DATA_CACHE[path]


def split_dual_arm_action(frame):
    arr = np.asarray(frame, dtype=np.float64).reshape(-1)
    if arr.shape[0] >= 16:
        return arr[:8], arr[8:16]
    if arr.shape[0] >= 14:
        return arr[:7], arr[7:14]
    if arr.shape[0] >= 12:
        return arr[:6], arr[6:12]
    raise ValueError(f"action/qpos dimension is insufficient, need at least 12 dims, got {arr.shape}")


def action_to_joint_frames(action):
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action[None]
    left, right = [], []
    for frame in action:
        l, r = split_dual_arm_action(frame)
        left.append(l[:6].tolist())
        right.append(r[:6].tolist())
    return left, right


def episode_summary(ep, idx):
    out = {"episode_index": idx, "keys": sorted([str(k) for k in ep.keys()])}
    for key in ("episode_name", "data_index", "task", "caption"):
        if key in ep:
            out[key] = ep[key]
    for key in ("action", "qpos"):
        if key in ep:
            try:
                arr = np.asarray(ep[key])
                out[key + "_shape"] = list(arr.shape)
            except Exception:
                out[key + "_shape"] = "unknown"
    return out


def get_camera_video_path(ep):
    for key in ("cam_high_video_path", "main_camera_video_path", "main_video_path", "video_path", "image_path"):
        if key in ep and ep[key]:
            return ep[key], key
    return None, None


def read_video_frame_data_url(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open high/main camera video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video frame count is invalid: {video_path}")
    frame_idx = int(np.clip(frame_idx, 0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read high/main camera frame: {video_path}, frame={frame_idx}")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("Failed to encode camera jpg")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + b64, frame_idx, total


def build_camera_frame_payload(params):
    data_pkl = resolve_path(params.get("data_pkl", ""))
    meta = load_dataset(data_pkl)
    if not meta:
        raise ValueError("Data is empty")
    episode_index = int(np.clip(int(params.get("episode_index", 0)), 0, len(meta) - 1))
    ep = meta[episode_index]
    image_key = params.get("image_key")
    if image_key:
        video_path = ep.get(image_key)
        video_key = image_key
        if not video_path:
            return {"image": None, "video_path": None, "video_key": video_key, "error": f"Episode does not have video field: {image_key}"}
    else:
        video_path, video_key = get_camera_video_path(ep)
    if video_path is None:
        return {"image": None, "video_path": None, "video_key": None, "error": "Episode does not have high/main camera video field"}
    image, frame_idx, total = read_video_frame_data_url(video_path, int(params.get("frame_idx", 0)))
    return {"image": image, "frame_idx": frame_idx, "total_frames": total, "video_path": video_path, "video_key": video_key}


def build_calib_api_payload(params):
    """Proxy/wrapper for APIs implemented in cam_calib_tools/app.py.

    The wrapper intentionally keeps this URDF viewer decoupled from the calibration
    tool implementation.  Clients can POST to /api/calib with one of:
      - {"api": "function_name", ...}
      - {"action": "alias", ...}
      - {"endpoint": "alias", ...}

    If no api/action/endpoint is provided, the response lists callable candidates
    exported by the calibration module.
    """
    module = get_cam_calib_module()
    api = params.get("api") or params.get("action") or params.get("endpoint") or params.get("path")
    if isinstance(api, str):
        api = api.strip().strip("/")
        if api.startswith("api/"):
            api = api[len("api/"):]
        if api.startswith("calib/"):
            api = api[len("calib/"):]

    aliases = {
        "defaults": ["load_saved_params"],
        "list_episodes": ["list_episodes"],
        "episodes": ["list_episodes"],
        "render": ["render_overlay"],
        "overlay": ["render_overlay"],
        "render3d": ["render_3d"],
        "render_3d": ["render_3d"],
        "save": ["save_params"],
        "save_params": ["save_params"],
        "camera_frame": ["__local_camera_frame__"],
        "cam_frame": ["__local_camera_frame__"],
        "scene": ["compute_scene", "render_overlay"],
        "project": ["project_points_to_image"],
    }

    call_names = []
    if api == "defaults":
        return {
            "params": module.load_saved_params(getattr(module, "DEFAULT_SAVE_PATH")),
            "save_path": getattr(module, "DEFAULT_SAVE_PATH"),
        }
    if api:
        call_names.extend(aliases.get(api, []))
        call_names.append(api)
    else:
        available = sorted(
            name for name in dir(module)
            if not name.startswith("_") and callable(getattr(module, name, None))
        )
        return {"available": available, "error": "missing api/action/endpoint"}

    last_error = None
    for name in call_names:
        if name == "__local_camera_frame__":
            return build_camera_frame_payload(params)
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        try:
            return fn(params)
        except TypeError as exc:
            last_error = exc
            try:
                return fn(**params)
            except TypeError as exc2:
                last_error = exc2
                continue

    raise KeyError(f"Cannot find callable API in cam_calib_tools: {api}; last_error={last_error}")


def build_scene_payload(params):
    data_pkl = resolve_path(params.get("data_pkl", ""))
    if not os.path.exists(data_pkl):
        raise FileNotFoundError(f"data_pkl does not exist: {data_pkl}")
    urdf_path = resolve_path(params.get("urdf_path") or DEFAULT_URDF_PATH, PIPER_DIR)
    source_key = params.get("source_key", "action") or "action"
    episode_index = int(params.get("episode_index", 0))
    ee_link = params.get("ee_link", "link6") or "link6"

    meta = load_dataset(data_pkl)
    if not meta:
        raise ValueError("Data is empty")
    episode_index = int(np.clip(episode_index, 0, len(meta) - 1))
    ep = meta[episode_index]
    if source_key not in ep:
        fallback = "qpos" if source_key == "action" and "qpos" in ep else "action"
        if fallback not in ep:
            raise KeyError(f"Cannot find {source_key} in episode, and no fallback action/qpos available")
        source_key = fallback

    left_frames, right_frames = action_to_joint_frames(ep[source_key])
    robot = get_robot(urdf_path)
    model = robot.to_client_model(base_link="base_link", ee_link=ee_link)

    left_base = params.get("left_base", DEFAULT_LEFT_BASE)
    right_base = params.get("right_base", DEFAULT_RIGHT_BASE)
    payload = {
        "data_pkl": data_pkl,
        "urdf_path": urdf_path,
        "source_key": source_key,
        "episode": episode_summary(ep, episode_index),
        "total_episodes": len(meta),
        "frame_count": len(left_frames),
        "left_base_T": mat4_to_list(make_T(left_base, [0, 0, 0])),
        "right_base_T": mat4_to_list(make_T(right_base, [0, 0, 0])),
        "left": left_frames,
        "right": right_frames,
        "robot": model,
    }
    return payload


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>🤖 GIGA DATA VIEWER</title>
<link rel="icon" type="image/png" href="/favicon-32.png" />
<link rel="icon" type="image/png" sizes="256x256" href="/favicon.png" />
<style>
html, body { margin:0; height:100%; overflow:hidden; font-family: Arial, sans-serif; background:#111; color:#eee; }
#layout { display:grid; grid-template-columns: 330px 1fr 360px; height:100%; }
#panel, #joint_panel { padding:14px; background:#1f1f1f; overflow:auto; }
#panel { border-right:1px solid #333; }
#joint_panel { border-left:1px solid #333; }
#viewer { position:relative; }
#topbar { position:absolute; left:12px; right:12px; top:12px; display:flex; gap:8px; align-items:center; padding:8px; background:rgba(0,0,0,0.58); border-radius:6px; z-index:3; }
#topbar button, #topbar label { width:auto; margin:0; white-space:nowrap; }
canvas { display:block; }
label { display:block; margin-top:10px; font-size:13px; color:#ccc; }
input, select, button { box-sizing:border-box; width:100%; margin-top:5px; padding:7px; border-radius:4px; border:1px solid #444; background:#2a2a2a; color:#eee; }
button { cursor:pointer; background:#363f52; }
button:hover { background:#42506b; }
.row { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
.row3 { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px; }
.ctrl { display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; margin-top:8px; }
#frame_slider { width:100%; }
#info { white-space:pre-wrap; word-break:break-all; padding:8px; border-radius:4px; background:#141414; color:#b7e3ff; min-height:100px; font-size:12px; }
#status { position:absolute; left:12px; bottom:12px; padding:8px 10px; background:rgba(0,0,0,0.55); border-radius:5px; font-size:13px; z-index:3; }
.small { font-size:12px; color:#aaa; line-height:1.4; }
#camera_preview { position:absolute; right:12px; bottom:12px; width:320px; background:rgba(0,0,0,0.62); border:1px solid #444; border-radius:6px; padding:6px; z-index:4; }
#camera_preview img { width:100%; display:block; border-radius:4px; }
#camera_preview_title { font-size:12px; color:#ddd; margin-bottom:4px; }
#axis_legend { position:absolute; right:12px; top:62px; padding:8px 10px; background:rgba(0,0,0,0.58); border-radius:6px; z-index:3; font-size:13px; line-height:1.6; }
.axis_x { color:#ff5555; font-weight:bold; }
.axis_y { color:#55ff55; font-weight:bold; }
.axis_z { color:#5590ff; font-weight:bold; }
.hidden { display:none; }
.joint_row { display:grid; grid-template-columns: 54px 1fr 78px; gap:8px; align-items:center; margin:10px 0; }
.joint_row input[type=range] { margin:0; padding:0; height:24px; accent-color:#6db8ff; cursor:pointer; }
.joint_val { font-family:monospace; color:#b7e3ff; text-align:right; padding:4px; margin:0; }
.side_title { margin-top:14px; padding-top:10px; border-top:1px solid #333; color:#fff; }
.checkline { display:flex; gap:6px; align-items:center; margin:0; }
.checkline input { width:auto; margin:0; }
.curve_box { margin-top:14px; padding-top:10px; border-top:1px solid #333; }
.curve_canvas { width:100%; height:150px; background:#111; border:1px solid #333; border-radius:5px; display:block; margin-top:8px; }
.curve_legend { font-size:12px; color:#aaa; line-height:1.5; }
</style>
<script async src="https://unpkg.com/es-module-shims@1.10.0/dist/es-module-shims.js"></script>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="layout">
  <div id="panel">
    <h2>🤖 GIGA DATA VIEWER</h2>
    <div class="small">Read action (or qpos) from pkl, render bimanual 3D animation via URDF STL meshes. Left-click to rotate, right-click to pan, scroll to zoom.</div>

    <label>data_pkl</label>
    <input id="data_pkl" placeholder="relative path is resolved against the project root" />

    <!-- urdf_path is auto-filled with the bundled piper URDF; no need to edit manually -->
    <input id="urdf_path" type="hidden" />

    <div class="row">
      <div><label>Data Field</label><select id="source_key"><option value="action">action</option><option value="qpos">qpos</option></select></div>
      <div><label>episode</label><input id="episode_index" type="number" value="0" min="0" /></div>
    </div>

    <div class="hidden">
      <input id="ee_link" value="link6" />
      <input id="lbx" type="number" value="0.0"><input id="lby" type="number" value="0.325"><input id="lbz" type="number" value="0.0">
      <input id="rbx" type="number" value="0.0"><input id="rby" type="number" value="-0.30"><input id="rbz" type="number" value="0.0">
    </div>

    <button id="load_btn">Load Data/URDF</button>

    <h3>Playback</h3>
    <input id="frame_slider" type="range" min="0" max="0" value="0" />
    <div class="row"><input id="frame_num" type="number" value="0" min="0" /><input id="fps" type="number" value="15" min="1" max="60" /></div>
    <div class="ctrl">
      <button id="prev_btn">Prev Frame</button>
      <button id="play_btn">Play</button>
      <button id="pause_btn">Pause</button>
      <button id="next_btn">Next Frame</button>
    </div>

    <h3>View</h3>
    <div class="ctrl">
      <button id="view_front">Front</button>
      <button id="view_side">Side</button>
      <button id="view_top">Top</button>
      <button id="view_iso">Iso</button>
    </div>
    <button id="reset_view">Reset View</button>

    <h3>Status</h3>
    <pre id="info"></pre>
  </div>
  <div id="viewer">
    <div id="topbar">
      <button id="screenshot_btn">Screenshot</button>
      <button id="record_btn">Save Video</button>
      <button id="stop_record_btn">Stop Recording</button>
      <label class="checkline"><input id="show_skeleton" type="checkbox" checked>Skeleton</label>
      <label class="checkline"><input id="show_joint_axes" type="checkbox">Joint Axes</label>
      <label class="checkline"><input id="transparent_model" type="checkbox" checked>Transparent Model</label>
      <label class="checkline"><input id="show_camera_preview" type="checkbox" checked>cam</label>
    </div>
    <div id="axis_legend" class="hidden">
      <div><span class="axis_x">X</span> = Red</div>
      <div><span class="axis_y">Y</span> = Green</div>
      <div><span class="axis_z">Z</span> = Blue</div>
    </div>
    <div id="camera_preview" class="hidden">
      <div id="camera_preview_title">high/main camera</div>
      <img id="camera_preview_img" />
    </div>
    <div id="status">Not Loaded</div>
  </div>
  <div id="joint_panel">
    <h2>Joint Angles</h2>
    <div class="small">Updates during playback; dragging slider pauses and manually changes the current pose.</div>
    <div class="side_title">Left Arm</div>
    <div id="left_joint_controls"></div>
    <div class="side_title">Right Arm</div>
    <div id="right_joint_controls"></div>
    <div class="curve_box">
      <h3>Joint Normalized Curves</h3>
      <div class="curve_legend">X-axis = frame; Y-axis = per-joint min-max normalized value within the current episode; vertical line = current frame.</div>
      <canvas id="left_joint_curve" class="curve_canvas"></canvas>
      <canvas id="right_joint_curve" class="curve_canvas"></canvas>
    </div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const $ = id => document.getElementById(id);
$('urdf_path').value = ''' + json.dumps(DEFAULT_URDF_PATH) + r''';
// Default data_pkl is fetched from the server (resolved against PROJECT_ROOT)
fetch('/api/defaults').then(r=>r.json()).then(d=>{ if(d && d.data_pkl) $('data_pkl').value = d.data_pkl; });

let scene, camera, renderer, controls;
let leftRoot, rightRoot, robotModel, dataPayload;
let leftLinkGroups = {}, rightLinkGroups = {};
let modelMaterials = [];
let leftSkeleton, rightSkeleton, leftJointAxes, rightJointAxes;
let playing = false, timer = null, currentFrame = 0;
let manualLeft = null, manualRight = null;
let mediaRecorder = null, recordedChunks = [];
let cameraPreviewBusy = false, cameraPreviewPending = false, lastCameraPreviewTime = 0;
let activeJointSlider = null;
let stlLoader = new STLLoader();

initThree();
animate();

function initThree(){
  const viewer = $('viewer');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);
  scene.up.set(0, 0, 1);
  camera = new THREE.PerspectiveCamera(55, viewer.clientWidth / viewer.clientHeight, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.05, -1.15, 0.58);
  renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  viewer.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.25, 0.0, 0.22);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.2));
  const light = new THREE.DirectionalLight(0xffffff, 1.0);
  light.position.set(1.5, -2.0, 3.0);
  scene.add(light);
  const grid = new THREE.GridHelper(2.0, 20, 0x555555, 0x333333);
  grid.name = 'ground_grid';
  grid.rotation.x = Math.PI / 2;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.35));

  window.addEventListener('resize', () => {
    camera.aspect = viewer.clientWidth / viewer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    drawJointCurves();
  });
}

function animate(){
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function setStatus(text){ $('status').textContent = text; }
function setInfo(obj){ $('info').textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2); }
function num(id){ return parseFloat($(id).value); }

function params(){
  return {
    data_pkl: $('data_pkl').value,
    urdf_path: $('urdf_path').value,
    source_key: $('source_key').value,
    episode_index: parseInt($('episode_index').value || '0'),
    ee_link: $('ee_link').value || 'link6',
    left_base: [num('lbx'), num('lby'), num('lbz')],
    right_base: [num('rbx'), num('rby'), num('rbz')],
  };
}

async function loadScene(){
  pause();
  setStatus('Loading...');
  const p = params();
  if(!p.data_pkl){
    throw new Error('Please provide a data_pkl file path (e.g. /path/to/data.pkl).');
  }
  if(!/\.pkl$/i.test(p.data_pkl)){
    throw new Error('data_pkl must point to a .pkl file, not a directory: ' + p.data_pkl);
  }
  const res = await fetch('/api/scene', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(p)});
  const body = await res.json();
  if(!res.ok || body.error){ throw new Error(body.error || res.statusText); }
  dataPayload = body;
  robotModel = body.robot;
  await buildRobotMeshes();
  buildJointControls();
  $('frame_slider').max = Math.max(0, body.frame_count - 1);
  $('frame_num').max = Math.max(0, body.frame_count - 1);
  setFrame(0);
  updateCameraPreview(true);
  setDefaultView();
  setStatus(`Loaded ${body.frame_count} frames`);
  setInfo({episode: body.episode, source_key: body.source_key, frame_count: body.frame_count, total_episodes: body.total_episodes});
}

function clearRobot(){
  if(leftRoot) scene.remove(leftRoot);
  if(rightRoot) scene.remove(rightRoot);
  if(leftSkeleton) scene.remove(leftSkeleton);
  if(rightSkeleton) scene.remove(rightSkeleton);
  if(leftJointAxes) scene.remove(leftJointAxes);
  if(rightJointAxes) scene.remove(rightJointAxes);
  leftRoot = new THREE.Group();
  rightRoot = new THREE.Group();
  leftSkeleton = new THREE.Group();
  rightSkeleton = new THREE.Group();
  leftJointAxes = new THREE.Group();
  rightJointAxes = new THREE.Group();
  leftRoot.name = 'left_arm';
  rightRoot.name = 'right_arm';
  scene.add(leftRoot);
  scene.add(rightRoot);
  scene.add(leftSkeleton);
  scene.add(rightSkeleton);
  scene.add(leftJointAxes);
  scene.add(rightJointAxes);
  leftLinkGroups = {};
  rightLinkGroups = {};
  modelMaterials = [];
  manualLeft = null;
  manualRight = null;
}

async function buildRobotMeshes(){
  clearRobot();
  leftRoot.applyMatrix4(matFromList(dataPayload.left_base_T));
  rightRoot.applyMatrix4(matFromList(dataPayload.right_base_T));
  for(const side of ['left', 'right']){
    const root = side === 'left' ? leftRoot : rightRoot;
    const map = side === 'left' ? leftLinkGroups : rightLinkGroups;
    for(const link of robotModel.links){
      const g = new THREE.Group();
      g.name = `${side}:${link.name}`;
      root.add(g);
      map[link.name] = g;
      for(const vis of link.visuals){
        await addVisual(g, vis, side);
      }
    }
  }
}

function addVisual(parent, vis, side){
  return new Promise((resolve) => {
    stlLoader.load(vis.url, geom => {
      geom.computeVertexNormals();
      const rgba = vis.color || [0.78, 0.82, 0.93, 1.0];
      const color = side === 'left' ? new THREE.Color(rgba[0] * 0.75, Math.min(1, rgba[1] * 1.08), rgba[2] * 0.75) : new THREE.Color(Math.min(1, rgba[0] * 1.08), rgba[1] * 0.72, rgba[2] * 0.45);
      const transparent = $('transparent_model').checked;
      const mat = new THREE.MeshStandardMaterial({color, roughness:0.65, metalness:0.05, transparent:true, opacity:transparent ? 0.46 : 1.0, depthWrite:!transparent});
      modelMaterials.push(mat);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(matFromList(vis.origin_T));
      mesh.scale.set(vis.scale[0], vis.scale[1], vis.scale[2]);
      parent.add(mesh);
      resolve();
    }, undefined, err => {
      console.warn('mesh load failed', vis.url, err);
      resolve();
    });
  });
}

function matFromList(list){
  const m = new THREE.Matrix4();
  const e = list.flat();
  m.set(e[0],e[1],e[2],e[3], e[4],e[5],e[6],e[7], e[8],e[9],e[10],e[11], e[12],e[13],e[14],e[15]);
  return m;
}

function identity(){ return new THREE.Matrix4(); }
function axisAngle(axis, q){
  const v = new THREE.Vector3(axis[0], axis[1], axis[2]).normalize();
  return new THREE.Matrix4().makeRotationAxis(v, q);
}
function prismatic(axis, q){
  return new THREE.Matrix4().makeTranslation(axis[0]*q, axis[1]*q, axis[2]*q);
}
function jointMotion(joint, q){
  if(joint.type === 'revolute' || joint.type === 'continuous') return axisAngle(joint.axis, q || 0);
  if(joint.type === 'prismatic') return prismatic(joint.axis, q || 0);
  return identity();
}

function applyFK(side, qpos){
  const map = side === 'left' ? leftLinkGroups : rightLinkGroups;
  if(!robotModel || !map[robotModel.base_link]) return [];
  let T = identity();
  map[robotModel.base_link].matrix.copy(T);
  map[robotModel.base_link].matrixAutoUpdate = false;
  const jointWorld = [{name: robotModel.base_link, T: T.clone(), movable:false}];
  let qi = 0;
  for(const joint of robotModel.joints){
    T = T.clone().multiply(matFromList(joint.origin_T));
    const isMovable = joint.type === 'revolute' || joint.type === 'continuous' || joint.type === 'prismatic';
    if(isMovable){
      T = T.clone().multiply(jointMotion(joint, qpos[qi] || 0));
      qi += 1;
    }
    jointWorld.push({name: joint.child, T: T.clone(), movable:isMovable});
    if(map[joint.child]){
      map[joint.child].matrixAutoUpdate = false;
      map[joint.child].matrix.copy(T);
    }
  }
  return jointWorld;
}

function transformJointWorld(side, jointWorld){
  const base = side === 'left' ? matFromList(dataPayload.left_base_T) : matFromList(dataPayload.right_base_T);
  return jointWorld.map(item => ({...item, T: base.clone().multiply(item.T)}));
}

function positionFromMatrix(T){
  const p = new THREE.Vector3();
  p.setFromMatrixPosition(T);
  return p;
}

function drawSkeleton(group, jointWorld, color){
  group.clear();
  group.visible = $('show_skeleton').checked;
  const pts = jointWorld.map(item => positionFromMatrix(item.T));
  if(pts.length >= 2){
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({color, linewidth:2});
    group.add(new THREE.Line(geom, mat));
  }
  for(const p of pts){
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 12, 8),
      new THREE.MeshBasicMaterial({color})
    );
    marker.position.copy(p);
    group.add(marker);
  }
}

function makeAxisLabel(text, color, pos){
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 42px Arial';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map:texture, transparent:true, depthTest:false}));
  sprite.position.copy(pos);
  sprite.scale.set(0.06, 0.06, 0.06);
  return sprite;
}

function makeThickAxis(T){
  const g = new THREE.Group();
  const axes = [
    {name:'X', color:0xff3333, css:'#ff5555', dir:new THREE.Vector3(1,0,0)},
    {name:'Y', color:0x33ff33, css:'#55ff55', dir:new THREE.Vector3(0,1,0)},
    {name:'Z', color:0x3377ff, css:'#5590ff', dir:new THREE.Vector3(0,0,1)},
  ];
  const len = 0.09;
  const radius = 0.0045;
  for(const a of axes){
    const start = new THREE.Vector3(0,0,0);
    const end = a.dir.clone().multiplyScalar(len);
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([start, end]),
      new THREE.LineBasicMaterial({color:a.color, linewidth:4, depthTest:false})
    );
    line.renderOrder = 999;
    g.add(line);
    const mid = end.clone().multiplyScalar(0.5);
    const cyl = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, len, 10),
      new THREE.MeshBasicMaterial({color:a.color, depthTest:false})
    );
    cyl.position.copy(mid);
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), a.dir.clone().normalize());
    g.add(cyl);
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(radius * 2.2, 0.018, 12),
      new THREE.MeshBasicMaterial({color:a.color, depthTest:false})
    );
    cone.position.copy(end);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), a.dir.clone().normalize());
    g.add(cone);
    g.add(makeAxisLabel(a.name, a.css, a.dir.clone().multiplyScalar(len + 0.035)));
  }
  g.matrixAutoUpdate = false;
  g.matrix.copy(T);
  return g;
}

function drawJointAxes(group, jointWorld){
  group.clear();
  const visible = $('show_joint_axes').checked;
  group.visible = visible;
  $('axis_legend').classList.toggle('hidden', !visible);
  for(const item of jointWorld){
    if(!item.movable) continue;
    group.add(makeThickAxis(item.T));
  }
}

function updateJointControls(leftQ, rightQ){
  for(const [side, q] of [['left', leftQ], ['right', rightQ]]){
    for(let i = 0; i < q.length; i++){
      const slider = $(`${side}_joint_${i}`);
      const val = $(`${side}_joint_${i}_val`);
      const id = `${side}_${i}`;
      if(slider && activeJointSlider !== id) slider.value = q[i];
      if(val && activeJointSlider !== id) val.value = Number(q[i]).toFixed(3);
    }
  }
}

function currentQ(){
  const leftQ = manualLeft || (dataPayload ? dataPayload.left[currentFrame].slice() : []);
  const rightQ = manualRight || (dataPayload ? dataPayload.right[currentFrame].slice() : []);
  return {leftQ, rightQ};
}

function refreshPose(){
  if(!dataPayload) return;
  const {leftQ, rightQ} = currentQ();
  const leftWorld = transformJointWorld('left', applyFK('left', leftQ));
  const rightWorld = transformJointWorld('right', applyFK('right', rightQ));
  drawSkeleton(leftSkeleton, leftWorld, 0x62d08f);
  drawSkeleton(rightSkeleton, rightWorld, 0xffa15a);
  drawJointAxes(leftJointAxes, leftWorld);
  drawJointAxes(rightJointAxes, rightWorld);
  updateJointControls(leftQ, rightQ);
}

function updateModelTransparency(){
  const transparent = $('transparent_model').checked;
  for(const mat of modelMaterials){
    mat.transparent = true;
    mat.opacity = transparent ? 0.46 : 1.0;
    mat.depthWrite = !transparent;
    mat.needsUpdate = true;
  }
}

async function updateCameraPreview(force=false){
  if(!$('show_camera_preview').checked){
    $('camera_preview').classList.add('hidden');
    return;
  }
  const now = performance.now();
  if(!force && playing && now - lastCameraPreviewTime < 250) return;
  lastCameraPreviewTime = now;
  if(!dataPayload || cameraPreviewBusy){ cameraPreviewPending = true; return; }
  cameraPreviewBusy = true;
  cameraPreviewPending = false;
  try{
    const res = await fetch('/api/camera_frame', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({...params(), frame_idx:currentFrame})});
    const body = await res.json();
    if(body.image){
      $('camera_preview_img').src = body.image;
      $('camera_preview_title').textContent = `${body.video_key} | ${body.frame_idx + 1}/${body.total_frames}`;
      $('camera_preview').classList.remove('hidden');
    } else {
      $('camera_preview').classList.add('hidden');
    }
  } catch(err){
    $('camera_preview').classList.add('hidden');
  } finally {
    cameraPreviewBusy = false;
    if(cameraPreviewPending) updateCameraPreview();
  }
}

function setFrame(i){
  if(!dataPayload) return;
  manualLeft = null;
  manualRight = null;
  currentFrame = Math.max(0, Math.min(dataPayload.frame_count - 1, parseInt(i || 0)));
  $('frame_slider').value = currentFrame;
  $('frame_num').value = currentFrame;
  refreshPose();
  updateCameraPreview();
  drawJointCurves();
  setStatus(`${playing ? 'Playing' : 'Paused'} | frame ${currentFrame + 1}/${dataPayload.frame_count}`);
}

function resizeCanvasToDisplaySize(canvas){
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if(canvas.width !== width || canvas.height !== height){
    canvas.width = width;
    canvas.height = height;
  }
  return {width, height, dpr};
}

function drawJointCurve(canvasId, frames, title){
  const canvas = $(canvasId);
  if(!canvas) return;
  const {width, height, dpr} = resizeCanvasToDisplaySize(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.scale(dpr, dpr);
  const w = width / dpr;
  const h = height / dpr;
  const padL = 28, padR = 8, padT = 18, padB = 18;
  const plotW = Math.max(1, w - padL - padR);
  const plotH = Math.max(1, h - padT - padB);

  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
  for(let k = 1; k < 4; k++){
    const y = padT + plotH * k / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
  }

  ctx.fillStyle = '#aaa';
  ctx.font = '12px Arial';
  ctx.fillText(title, padL, 12);
  if(!frames || frames.length === 0){
    ctx.fillText('No Data', padL + 8, padT + 24);
    ctx.restore();
    return;
  }

  const jointCount = Math.max(...frames.map(f => (f || []).length));
  const colors = ['#ff6666', '#66d9ef', '#a6e22e', '#fd971f', '#ae81ff', '#ffd866', '#f92672', '#66ffcc'];
  const n = frames.length;

  for(let j = 0; j < jointCount; j++){
    const vals = frames.map(f => Number((f || [])[j] || 0));
    let mn = Math.min(...vals), mx = Math.max(...vals);
    if(!Number.isFinite(mn) || !Number.isFinite(mx)) continue;
    const range = Math.max(1e-9, mx - mn);
    ctx.strokeStyle = colors[j % colors.length];
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for(let i = 0; i < n; i++){
      const x = padL + (n <= 1 ? 0 : i / (n - 1)) * plotW;
      const norm = (vals[i] - mn) / range;
      const y = padT + (1 - norm) * plotH;
      if(i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const cx = padL + (n <= 1 ? 0 : currentFrame / (n - 1)) * plotW;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(cx, padT);
  ctx.lineTo(cx, padT + plotH);
  ctx.stroke();

  ctx.fillStyle = '#aaa';
  ctx.font = '11px Arial';
  ctx.fillText('0', 4, padT + plotH + 3);
  ctx.fillText('1', 4, padT + 4);
  ctx.fillText(`${currentFrame + 1}/${n}`, padL + plotW - 48, h - 4);
  ctx.restore();
}

function drawJointCurves(){
  if(!dataPayload) return;
  const leftFrames = dataPayload.left.map(f => f.slice());
  const rightFrames = dataPayload.right.map(f => f.slice());
  if(manualLeft) leftFrames[currentFrame] = manualLeft.slice();
  if(manualRight) rightFrames[currentFrame] = manualRight.slice();
  drawJointCurve('left_joint_curve', leftFrames, 'left');
  drawJointCurve('right_joint_curve', rightFrames, 'right');
}

function setManualJoint(side, i, value){
  if(!dataPayload) return;
  value = Number(value);
  if(!Number.isFinite(value)) return;
  if(!manualLeft || !manualRight){
    manualLeft = dataPayload.left[currentFrame].slice();
    manualRight = dataPayload.right[currentFrame].slice();
  }
  const q = side === 'left' ? manualLeft : manualRight;
  q[i] = value;
  const slider = $(`${side}_joint_${i}`);
  const val = $(`${side}_joint_${i}_val`);
  if(slider) slider.value = value;
  if(val) val.value = Number(value).toFixed(3);
  refreshPose();
  drawJointCurves();
  setStatus(`Manual Adjust | frame ${currentFrame + 1}/${dataPayload.frame_count}`);
}

function buildJointControls(){
  const movableNames = robotModel.joints.filter(j => j.type === 'revolute' || j.type === 'continuous' || j.type === 'prismatic').map(j => j.name);
  for(const side of ['left', 'right']){
    const box = $(`${side}_joint_controls`);
    box.innerHTML = '';
    movableNames.forEach((name, i) => {
      const row = document.createElement('div');
      row.className = 'joint_row';
      row.innerHTML = `<span>${name}</span><input id="${side}_joint_${i}" type="range" min="-3.2" max="3.2" step="0.001"><input id="${side}_joint_${i}_val" class="joint_val" type="number" min="-3.2" max="3.2" step="0.001" value="0.000">`;
      box.appendChild(row);
      const slider = row.querySelector(`#${side}_joint_${i}`);
      const val = row.querySelector(`#${side}_joint_${i}_val`);
      slider.addEventListener('pointerdown', () => { activeJointSlider = `${side}_${i}`; pause(false); });
      slider.addEventListener('pointerup', () => { activeJointSlider = null; });
      slider.addEventListener('touchend', () => { activeJointSlider = null; });
      slider.addEventListener('input', e => {
        activeJointSlider = `${side}_${i}`;
        pause(false);
        setManualJoint(side, i, parseFloat(e.target.value));
      });
      slider.addEventListener('change', () => { activeJointSlider = null; });
      val.addEventListener('focus', () => { activeJointSlider = `${side}_${i}`; pause(false); });
      val.addEventListener('input', e => {
        activeJointSlider = `${side}_${i}`;
        pause(false);
        const value = Math.max(-3.2, Math.min(3.2, parseFloat(e.target.value || '0')));
        setManualJoint(side, i, value);
      });
      val.addEventListener('blur', () => { activeJointSlider = null; refreshPose(); });
    });
  }
}

function step(delta){ setFrame(currentFrame + delta); }
function play(){
  if(!dataPayload || playing) return;
  playing = true;
  const tick = () => {
    const fps = Math.max(1, Math.min(60, parseFloat($('fps').value || '15')));
    step(1);
    if(currentFrame >= dataPayload.frame_count - 1) setFrame(0);
    timer = setTimeout(tick, 1000 / fps);
  };
  tick();
}
function pause(resetManual=true){
  playing = false;
  if(timer) clearTimeout(timer);
  timer = null;
  if(resetManual) setFrame(currentFrame);
}

function setView(pos){
  camera.position.set(pos[0], pos[1], pos[2]);
  controls.target.set(0.25, 0.0, 0.22);
  controls.update();
}
function setDefaultView(){ setView([1.05, -1.15, 0.58]); }

function screenshot(){
  renderer.render(scene, camera);
  const a = document.createElement('a');
  a.href = renderer.domElement.toDataURL('image/png');
  a.download = `urdf_frame_${String(currentFrame).padStart(5, '0')}.png`;
  a.click();
}

function startRecord(){
  if(mediaRecorder && mediaRecorder.state === 'recording') return;
  recordedChunks = [];
  const stream = renderer.domElement.captureStream(Math.max(1, parseFloat($('fps').value || '15')));
  mediaRecorder = new MediaRecorder(stream, {mimeType: 'video/webm'});
  mediaRecorder.ondataavailable = e => { if(e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, {type:'video/webm'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'urdf_animation.webm';
    a.click();
    URL.revokeObjectURL(a.href);
  };
  mediaRecorder.start();
  play();
  setStatus('Recording...');
}
function stopRecord(){
  if(mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
  pause();
}

$('load_btn').onclick = () => loadScene().catch(err => { setStatus('Load failed'); setInfo(err.stack || err.message); });
$('prev_btn').onclick = () => { pause(); step(-1); };
$('next_btn').onclick = () => { pause(); step(1); };
$('play_btn').onclick = play;
$('pause_btn').onclick = pause;
$('frame_slider').oninput = e => { pause(false); setFrame(e.target.value); };
$('frame_num').onchange = e => { pause(false); setFrame(e.target.value); };
$('view_front').onclick = () => setView([1.15, 0.0, 0.35]);
$('view_side').onclick = () => setView([0.25, -1.15, 0.35]);
$('view_top').onclick = () => setView([0.25, 0.0, 1.55]);
$('view_iso').onclick = setDefaultView;
$('reset_view').onclick = setDefaultView;
$('screenshot_btn').onclick = screenshot;
$('record_btn').onclick = startRecord;
$('stop_record_btn').onclick = stopRecord;
$('show_skeleton').onchange = refreshPose;
$('show_joint_axes').onchange = refreshPose;
$('transparent_model').onchange = updateModelTransparency;
$('show_camera_preview').onchange = () => updateCameraPreview(true);
window.addEventListener('keydown', e => {
  if(e.code === 'Space'){ playing ? pause() : play(); e.preventDefault(); }
  if(e.code === 'ArrowLeft'){ pause(); step(-1); }
  if(e.code === 'ArrowRight'){ pause(); step(1); }
});
</script>
</body>
</html>
'''


class AppHandler(BaseHTTPRequestHandler):
    server_version = "URDFVisTools/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n).decode("utf-8") if n > 0 else "{}"
        return json.loads(raw or "{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path in ("/favicon.png", "/favicon-32.png", "/favicon.ico"):
                name = parsed.path.lstrip("/")
                path = os.path.join(STATIC_DIR, name)
                if not os.path.exists(path):
                    self.send_json({"error": "favicon not found"}, HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                with open(path, "rb") as f:
                    self.send_bytes(f.read(), content_type)
                return
            if parsed.path in ("/calib", "/calib.html"):
                self.send_json({"error": "calibration page not implemented in this build"}, HTTPStatus.NOT_FOUND)
                return
            if parsed.path.startswith("/static/"):
                rel = parsed.path[len("/static/"):].lstrip("/")
                path = os.path.abspath(os.path.join(STATIC_DIR, rel))
                if not path.startswith(os.path.abspath(STATIC_DIR) + os.sep) or not os.path.exists(path):
                    self.send_json({"error": "static file not found"}, HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                with open(path, "rb") as f:
                    self.send_bytes(f.read(), content_type)
                return
            if parsed.path == "/mesh":
                qs = parse_qs(parsed.query)
                path = os.path.abspath(unquote(qs.get("path", [""])[0]))
                if not os.path.exists(path):
                    self.send_json({"error": f"mesh does not exist: {path}"}, HTTPStatus.NOT_FOUND)
                    return
                with open(path, "rb") as f:
                    self.send_bytes(f.read(), "model/stl")
                return
            if parsed.path == "/api/defaults":
                self.send_json({
                    "data_pkl": DEFAULT_DATA_PKL,
                    "urdf_path": DEFAULT_URDF_PATH,
                    "left_base": DEFAULT_LEFT_BASE,
                    "right_base": DEFAULT_RIGHT_BASE,
                })
                return
            if parsed.path == "/api/calib/defaults":
                self.send_json(build_calib_api_payload({"path": "defaults"}))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc), "traceback": traceback.format_exc()}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/scene":
                self.send_json(build_scene_payload(self.read_json()))
                return
            if parsed.path == "/api/camera_frame":
                self.send_json(build_camera_frame_payload(self.read_json()))
                return
            if parsed.path == "/api/calib" or parsed.path.startswith("/api/calib/"):
                params = self.read_json()
                if parsed.path.startswith("/api/calib/") and not any(k in params for k in ("api", "action", "endpoint", "path")):
                    params["path"] = parsed.path[len("/api/calib/"):]
                self.send_json(build_calib_api_payload(params))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc), "traceback": traceback.format_exc()}, HTTPStatus.BAD_REQUEST)


def main():
    parser = argparse.ArgumentParser(description="🤖 GIGA DATA VIEWER")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"🤖 GIGA DATA VIEWER: http://{args.host}:{args.port}")
    print(f"Default URDF: {DEFAULT_URDF_PATH}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
