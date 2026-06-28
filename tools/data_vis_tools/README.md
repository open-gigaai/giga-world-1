# 🤖 GIGA URDF Vis Tools

> Bimanual robot arm URDF/STL 3D animation visualization + camera calibration tool

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Three.js](https://img.shields.io/badge/Three.js-0.160-green)
![License](https://img.shields.io/badge/License-Proprietary-red)

---

## 📖 Introduction

This tool is used for **Giga-DataCraft** dataset visualization and analysis, integrating two core features:

| Feature | Page | Description |
|---------|------|-------------|
| 🦾 URDF 3D Viewer | `/` | Load pkl → parse action/qpos → bimanual URDF/STL 3D animation |
| 📷 Camera Calibration | `/calib` | Camera intrinsic/extrinsic calibration, multi-frame overlay, 3D FK + Camera visualization |

---

## 🏗️ Project Structure

```
tools/data_vis_tools/
├── piper/                    # Piper robot URDF + mesh assets
│   ├── urdf/                 # URDF/xacro description files
│   └── meshes/               # STL/DAE/GLB mesh files
├── app.py                    # Backend Python server (HTTP + API)
├── __init__.py
├── README.md                 # This document
└── static/                   # Frontend static resources
    ├── index.html            # Page 1: URDF 3D Viewer
    ├── calib.html            # Page 2: Camera Calibration
    ├── css/
    │   └── app.css           # Shared styles (dark theme, card layout)
    └── js/
        ├── app.js            # Page 1: Three.js 3D rendering + interaction
        └── calib.js          # Page 2: Camera calibration params + overlay interaction
```

**Design Principles:**
- Standalone tool, **does not modify** original code (`robo_tools.py`, `cam_utils.py`, `utils.py`, `key_pts_utils.py`)
- Frontend-backend separation: Python backend provides API, frontend HTML/CSS/JS independently maintained
- Dynamically loads `cam_calib_tools` calibration logic via `importlib` to avoid code duplication

---

## 🚀 Quick Start

### 1. Start the Server

```bash
cd tools/data_vis_tools

# Default port 8090
python app.py

# Custom host / port
python app.py --host 0.0.0.0 --port 8090
```

After starting, terminal output:

```
🤖 GIGA DATA VIEWER: http://0.0.0.0:8090
Default URDF: <repo>/tools/data_vis_tools/piper/piper.urdf
```

### 2. Open Browser

| Page | URL |
|------|-----|
| 🦾 URDF 3D Viewer | `http://<host>:8090/` |
| 📷 Camera Calibration | `http://<host>:8090/calib` |

> Use the top navigation bar to switch between pages.

---

## 🦾 Page 1: URDF 3D Viewer

### UI Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  🤖 GIGA DATA VIEWER     [🤖 URDF Viewer]  [📷 Camera Calibration]           │
├────────────┬──────────────────────────────────────────┬──────────────────────┤
│            │  ┌──────────────────────────────────┐    │                      │
│  📦 Data   │  │  📸 Screenshot  🎥 Save Video  Skeleton  Axes  │    │  🦾 Joint Angles │
│  🎬 Play   │  └──────────────────────────────────┘    │  Left Arm            │
│  📐 View   │                                          │  joint1  ═══●══  -0.7│
│  🧾 Status │         ┌────────────────────┐           │  joint2  ══════●  1.6 │
│            │         │                    │           │  joint3  ═══●══  -1.0│
│  data_pkl  │         │   3D Viewport       │           │  joint4  ════●═  0.6│
│  urdf_path │         │   (Three.js)       │           │  joint5  ════●═  0.2│
│  episode   │         │                    │           │  joint6  ═══●══  -0.7│
│  FPS: 15   │         │   🔴X  🟢Y  🔵Z   │           │                      │
│  [▶ Play]  │         │                    │           │  Right Arm           │
│  [⏸ Pause] │         │   [Skeleton] [Axes]│           │  joint1  ══════●  -0.0│
│  [⏮][⏭]   │         │                    │           │  joint2  ════●═  0.0 │
│            │         └────────────────────┘           │  joint3  ═════●  -0.0│
│  ┌────────────────────────────────────┐               │  joint4  ══════●  0.0 │
│  │ high/main camera   [cam ☑]        │               │  joint5  ══════●  0.0 │
│  │ ┌────────────────────────────────┐ │               │  joint6  ════●═  0.0 │
│  │ │  Camera Live View               │ │               │                      │
│  │ └────────────────────────────────┘ │               │  📈 Joint Normalized Curves │
│  └────────────────────────────────────┘               │  ┌──────────────────┐│
│         Not Loaded                                     │  │ ╱╲  ╱╲           ││
├──────────────────────────────────────────────────────┤  │╱  ╲╱  ╲╱╲  ╱╲   ││
│  Status: loaded | frame 42/300 | left/right             │  └──────────────────┘│
└──────────────────────────────────────────────────────┴──────────────────────┘
```

### Features

| Feature | Shortcut/Button | Description |
|---------|----------------|-------------|
| 📦 Load Data | `🚀 Load Data/URDF` | Load pkl + URDF, initialize 3D scene |
| 🎬 Playback | `▶`/`⏸`/`⏮`/`⏭` | Play/Pause/Prev/Next frame |
| 📐 View | `Front`/`Side`/`Top`/`Iso` | 4 preset views |
| 📸 Screenshot | `📸 Screenshot` | Save current view as PNG |
| 🎥 Record | `🎥 Save Video` → `⏹ Stop` | Record canvas stream as WebM |
| 🦴 Skeleton | `Skeleton ☑` | Show/hide robot arm skeleton lines |
| 📏 Joint Axes | `Joint Axes ☑` | Show XYZ axes for each joint (R=X G=Y B=Z) |
| 🔍 Transparent | `Transparent ☑` | Toggle STL model transparency |
| 📷 Camera Preview | `cam ☑` | Show high/main camera view in bottom-right |
| 🦾 Manual Joint | Right panel | Drag slider or input value, pause and manually change pose |
| 📈 Curves | Bottom-right | Min-max normalized curves for left/right arm joints over frames |

### Joint Angle Panel

The right panel shows 6 joint angles for each arm. **Updates during playback; dragging slider or inputting value pauses and manually changes the current pose.**

```
🦾 Joint Angles
Updates during playback; drag slider or input value to pause and manually adjust pose.

── Left Arm ─────────────────────
joint1  ════════●══════  -0.716
joint2  ═════════════●═   1.647
joint3  ══════●════════  -1.011
joint4  ════════●══════   0.625
joint5  ════════●══════   0.231
joint6  ════════●══════  -0.715

── Right Arm ────────────────────
joint1  ═════════════●═  -0.016
joint2  ═══════════════   0.036
joint3  ═════════════●═  -0.020
joint4  ═══════════════   0.000
joint5  ═══════════════   0.018
joint6  ═══════════════   0.012
```

### Normalized Curves

Bottom-right shows normalized curves for left/right arm joints over frames. White vertical line marks current frame.

```
📈 Joint Normalized Curves
X-axis = frame; Y-axis = min-max normalized value; white line = current frame

Left arm curves:                      Right arm curves:
┌────────────────────────┐    ┌────────────────────────┐
│  ╱╲   ╱╲               │    │                        │
│ ╱  ╲ ╱  ╲   ╱╲        │    │   ╱╲  ╱╲               │
│╱    ╲    ╲ ╱  ╲   |   │    │  ╱  ╲╱  ╲   |         │
│          ╲╱    ╲╱     │    │ ╱        ╲╱  ╲         │
└────────────────────────┘    └────────────────────────┘
joint1  joint2  joint3        joint1  joint2  joint3
joint4  joint5  joint6        joint4  joint5  joint6
```

---

## 📷 Page 2: Camera Calibration

### UI Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  📷 Camera Calibration     [🤖 URDF Viewer]  [📷 Camera Calibration]         │
├────────────────────────────────────┬─────────────────────────────────────────┤
│                                    │                                         │
│  📦 Data & Playback                │  ⚙️ Parameter Panel                      │
│  ┌──────────────────────────────┐  │                                         │
│  │ data.pkl    image key  ep.   │  │  💾 Save Path                          │
│  │ /path/...   cam_high    3    │  │  ─────────────────────                  │
│  │ Frame Slider ════════════●══ │  │                                         │
│  │ Frame Num: 120   Status: ready│  │  📐 Camera Extrinsics                   │
│  │                              │  │  cam_pos.x  ═══●═══  -0.0              │
│  │ 🚀Load  🔄Refresh 💾Save ↩Reset│  │  cam_pos.y  ══════●  0.0              │
│  └──────────────────────────────┘  │  cam_pos.z  ══════●  0.65             │
│                                    │  cam_forward.x ═●═══  0.50            │
│  🖼 Multi-Frame Overlay Results    │  cam_forward.y ═════  0.00            │
│  ┌──────────────────────────────┐  │  cam_forward.z ══●══ -0.80            │
│  │ Count: 4  frame step: 10     │  │                                         │
│  └──────────────────────────────┘  │  🔍 Camera Intrinsics K                 │
│                                    │  K.fx  ══════●══  609                  │
│  ┌──────────┐  ┌──────────┐       │  K.fy  ══════●══  609                  │
│  │ Overlay  │  │ Overlay  │       │  K.cx  ══════●══  320                  │
│  │ frame 110│  │ frame 115│       │  K.cy  ══════●══  240                  │
│  └──────────┘  └──────────┘       │                                         │
│  ┌──────────┐  ┌──────────┐       │  🦾 Robot Parameters                   │
│  │ Overlay  │  │ Overlay  │       │  left_arm_base_pos.y ═══●═  0.325      │
│  │ frame 120│  │ frame 125│       │  right_arm_base_pos.y ═●══ -0.30       │
│  └──────────┘  └──────────┘       │  camera_offset_local.z ═══●  0.20      │
│                                    │                                         │
│  🌐 3D FK + Camera                 │  🧾 Status                               │
│  ┌──────────────────────────────┐  │  {"episode_index": 3, ...}             │
│  │  3D Robot Arm + Camera Frustum│  │                                         │
│  └──────────────────────────────┘  │                                         │
└────────────────────────────────────┴─────────────────────────────────────────┘
```

### Features

| Feature | Description |
|---------|-------------|
| 📦 Data Loading | Read pkl dataset, select episode |
| 🖼 Multi-Frame Overlay | Generate calibration overlay images centered on current frame (EE projection + skeleton overlay) |
| 🌐 3D FK + Camera | Matplotlib 3D robot arm FK + camera frustum rendering |
| 📐 Camera Extrinsics | cam_pos / cam_forward real-time slider adjustment |
| 🔍 Camera Intrinsics | fx / fy / cx / cy real-time slider adjustment |
| 🦾 Robot Parameters | Left/right arm base xyz, camera offset xyz real-time adjustment |
| 💾 Save Parameters | Export calibration results as JSON file |

---

## 🛠️ Tech Stack

### Backend (Python)

| Module | Purpose |
|--------|---------|
| `http.server` | HTTP static file serving + REST API |
| `xml.etree.ElementTree` | URDF parsing |
| `numpy` | SE(3) / FK matrix computation |
| `cv2` (OpenCV) | Video frame reading and JPEG encoding |
| `pickle` | pkl data loading |
| `importlib` | Dynamic loading of `cam_calib_tools` calibration logic |

### Frontend (Web)

| Technology | Purpose |
|------------|---------|
| Three.js 0.160 | 3D rendering engine |
| `STLLoader` | Load STL mesh files |
| `OrbitControls` | 3D viewport interaction (drag rotate/zoom/pan) |
| Canvas API | Screenshot/recording/joint curve drawing |
| CSS Grid | Responsive three-column layout |

---

## 📡 API Endpoints

### URDF Viewer

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main page |
| `GET` | `/static/...` | Static resources |
| `GET` | `/mesh?path=...` | Return STL mesh file |
| `POST` | `/api/scene` | Load data + URDF, return 3D scene |
| `POST` | `/api/camera_frame` | Read camera frame for specified frame |

### Camera Calibration

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/calib` | Calibration page |
| `GET` | `/api/calib/defaults` | Get default calibration parameters |
| `POST` | `/api/calib/list_episodes` | List episodes |
| `POST` | `/api/calib/render` | Render overlay calibration results |
| `POST` | `/api/calib/render3d` | Render 3D FK + Camera |
| `POST` | `/api/calib/save` | Save calibration parameters JSON |

---

## 📦 Data Format

### pkl File Format

```python
# Top level is a list, each element is an episode dict
[
    {
        "episode_name": "episode_001",
        "action": np.ndarray,    # shape: (T, 12/14/16)
        "qpos": np.ndarray,      # shape: (T, 12/14/16)
        "cam_high_video_path": "/path/to/video.mp4",
        # ... other fields
    },
    ...
]
```

### action / qpos Dimension Split

| Total Dim | Left Arm | Right Arm | Description |
|-----------|----------|-----------|-------------|
| ≥16 | `[0:8]` | `[8:16]` | 7+1 (includes gripper) |
| ≥14 | `[0:7]` | `[7:14]` | 7 dim |
| ≥12 | `[0:6]` | `[6:12]` | 6 dim joint angles |

Default first 6 dims per arm are used as joint angles for FK computation.

---

## 🔧 Dependencies

```bash
# Core dependencies
pip install numpy opencv-python matplotlib

# Frontend Three.js loaded via CDN, no installation needed
```

---

## 📌 Default Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8090` | Listen port |
| URDF | `tools/data_vis_tools/piper/piper.urdf` | Default URDF path |
| Left arm base | `[0.0, 0.325, 0.0]` | Left arm base xyz |
| Right arm base | `[0.0, -0.30, 0.0]` | Right arm base xyz |
| FPS | `15` | Default playback frame rate |

---

## 🎨 Quick Reference

```
┌─────────────────────────────────────────────────────┐
│  URDF 3D Viewer Shortcuts                            │
│                                                     │
│  🖱 Left Drag    → Rotate 3D view                   │
│  🖱 Right Drag   → Pan 3D view                     │
│  🖱 Scroll       → Zoom                            │
│  ⏮ ⏭           → Prev/Next frame                  │
│  ▶ ⏸           → Play/Pause                       │
│  📐 Front/Side/Top/Iso → Preset views               │
│                                                     │
│  Camera Calibration Shortcuts                        │
│                                                     │
│  📐 Drag slider  → Real-time update overlay / 3D / multi-frame │
│  💾 Save params  → Export calibration as JSON        │
│  🖼 Frame count  → Adjust number of images and step │
└─────────────────────────────────────────────────────┘
```

---

## ⚠️ Notes

1. **Do not modify original tool code** — This tool implements FK / URDF parsing independently, does not import original `robo_tools.py` etc.
2. **STL mesh paths** — URDF-referenced mesh supports `package://` and relative paths, prefers `.STL` / `.stl`
3. **Camera video** — Calibration page reads `cam_high_video_path` field, also falls back to `main_camera_video_path`
4. **Frontend CDN** — Three.js loaded via unpkg CDN, requires network access

---

> Built with ❤️ for Giga-DataCraft
