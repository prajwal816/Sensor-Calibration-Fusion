# 3D Sensor Calibration & Fusion System

A production-grade, modular Python pipeline for **multi-sensor calibration**, **rigid-body transformation**, **point cloud registration**, and **RGB-D fusion**. Designed for robotics and perception applications where RGB and depth sensors must be aligned into a unified 3-D representation.

---

## Table of Contents

- [Architecture](#architecture)
- [Calibration Theory](#calibration-theory)
- [Transformation Mathematics](#transformation-mathematics)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Before vs After Results](#before-vs-after-results)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [ROS Integration](#ros-integration)

---

## Architecture

```
Raw RGB Images ──► Intrinsic Calibration ──► Camera Matrix K
Raw Depth Maps ──► Depth Back-projection ──► 3-D Points
                      │
                      ▼
              ┌───────────────┐
              │   Extrinsic   │
              │  Calibration  │──► T_rgb_depth (SE(3))
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐     ┌────────────────────┐
              │    Global     │────►│   ICP Refinement    │
              │ Registration  │     │ (Point-to-Plane)    │
              └───────────────┘     └────────┬───────────┘
                                             │
                                             ▼
                                    ┌────────────────────┐
                                    │   Sensor Fusion    │
                                    │  (RGB-D Merge)     │
                                    └────────┬───────────┘
                                             │
                                             ▼
                                    Unified Coloured 3-D
                                       Point Cloud
```

---

## Calibration Theory

### Intrinsic Calibration

Every camera maps 3-D world points to 2-D pixels through the **pinhole model**:

```
┌ u ┐       ┌ fx  0  cx ┐ ┌ X ┐
│ v │ = 1/Z │  0  fy cy │ │ Y │
└ 1 ┘       └  0   0  1 ┘ └ Z ┘
```

Where **K** = `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` is the **intrinsic matrix**.

| Parameter | Meaning |
|-----------|---------|
| `fx, fy`  | Focal length in pixel units |
| `cx, cy`  | Principal point (optical centre) |

**Distortion coefficients** `(k₁, k₂, p₁, p₂, k₃)` model radial and tangential lens distortion. We use OpenCV's `calibrateCamera()` with ≥ 10 checkerboard images to estimate both K and distortion.

### Extrinsic Calibration

The **extrinsic matrix** describes the rigid relationship between two sensors:

```
T = ┌ R  t ┐  ∈ SE(3)
    └ 0  1 ┘
```

Where R ∈ SO(3) is a 3×3 rotation, and t ∈ ℝ³ is the translation. We obtain this via `cv2.stereoCalibrate()` using synchronised checkerboard views from both sensors.

---

## Transformation Mathematics

### Homogeneous Coordinates

A 3-D point **p** = (x, y, z) is lifted to homogeneous form **p̃** = (x, y, z, 1). This lets translations be expressed as matrix multiplications:

```
p'  =  T · p̃

    ┌ x' ┐     ┌ r₁₁ r₁₂ r₁₃ tₓ ┐ ┌ x ┐
    │ y' │  =  │ r₂₁ r₂₂ r₂₃ tᵧ │ │ y │
    │ z' │     │ r₃₁ r₃₂ r₃₃ t_z │ │ z │
    └  1 ┘     └  0   0   0   1  ┘ └ 1 ┘
```

### Composition & Inversion

| Operation | Formula | Effect |
|-----------|---------|--------|
| Compose   | T₁₂ = T₁ · T₂ | Chain two transforms |
| Inverse   | T⁻¹ = `[Rᵀ, −Rᵀt; 0, 1]` | Reverse the transform |
| Apply     | p' = T · p̃ | Transform a point |

### Supported Rotation Representations

- **Euler angles** (roll, pitch, yaw) — `rotation_from_euler()`
- **Axis-angle** — `rotation_from_axis_angle()`
- **Quaternion** (x, y, z, w) — `rotation_from_quaternion()`

All convert to 3×3 rotation matrices via SciPy's `Rotation` class.

---

## Pipeline Overview

The automation script `run_pipeline.py` executes these stages in order:

```
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Intrinsic Calibration                        │
│  ► Detect checkerboard corners                          │
│  ► cv2.calibrateCamera() → K, distortion coeffs         │
│  ► Report reprojection RMSE                             │
├──────────────────────────────────────────────────────────┤
│  Stage 2 — Point Cloud Registration                     │
│  ► Measure BEFORE alignment error + Chamfer distance    │
│  ► FPFH + RANSAC global registration (coarse)           │
│  ► Point-to-plane ICP refinement (fine)                 │
│  ► Measure AFTER alignment error + Chamfer distance     │
├──────────────────────────────────────────────────────────┤
│  Stage 3 — Sensor Fusion                                │
│  ► Back-project depth → 3-D points                      │
│  ► Colour each point from aligned RGB                   │
│  ► Merge multiple views                                 │
│  ► Statistical outlier removal                          │
│  ► Export unified point cloud (PLY)                     │
└──────────────────────────────────────────────────────────┘
```

### Step-by-Step Calibration Guide

1. **Capture images** — Take 10–20 photos of a checkerboard from varying angles and distances.
2. **Run intrinsic calibration**:
   ```bash
   python scripts/calibrate.py intrinsic --image-dir data/calibration_images
   ```
3. **Run extrinsic calibration** (if using two sensors):
   ```bash
   python scripts/calibrate.py extrinsic \
       --rgb-dir data/rgb_images \
       --depth-dir data/depth_images \
       --rgb-intrinsics data/output/intrinsics.json
   ```
4. **Register point clouds**:
   ```bash
   python scripts/register.py global --source source.ply --target target.ply
   python scripts/register.py icp --source source.ply --target target.ply
   ```
5. **Fuse sensors**:
   ```bash
   python scripts/fuse.py rgbd --rgb image.png --depth depth.png --intrinsics intrinsics.json
   ```
6. **Or run everything end-to-end**:
   ```bash
   python scripts/run_pipeline.py --config configs/default.yaml
   ```

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd Sensor-Calibration-Fusion
pip install -r requirements.txt

# 2. Generate synthetic test data
python scripts/generate_sample_data.py

# 3. Run the full pipeline
python scripts/run_pipeline.py --config configs/default.yaml

# 4. Inspect outputs
#    data/output/intrinsics.json       — calibrated camera parameters
#    data/output/registered_source.ply — aligned point cloud
#    data/output/fused_scene.ply       — coloured fused point cloud
#    data/output/pipeline_report.json  — metrics report
```

---

## CLI Reference

### `scripts/calibrate.py`

| Command | Description |
|---------|-------------|
| `intrinsic --image-dir DIR` | Intrinsic calibration from checkerboard images |
| `extrinsic --rgb-dir DIR --depth-dir DIR` | Stereo extrinsic calibration |

### `scripts/register.py`

| Command | Description |
|---------|-------------|
| `icp --source FILE --target FILE` | ICP point cloud alignment |
| `global --source FILE --target FILE` | FPFH + RANSAC global registration |

### `scripts/fuse.py`

| Command | Description |
|---------|-------------|
| `rgbd --rgb FILE --depth FILE --intrinsics FILE` | RGB-D fusion |
| `multi --clouds FILE [FILE …] --transforms FILE [FILE …]` | Multi-sensor merge |

### `scripts/run_pipeline.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/default.yaml` | Configuration file |
| `--skip-calibration` | off | Skip intrinsic calibration |

---

## Before vs After Results

The pipeline reports quantitative metrics at each stage:

| Metric | Before Registration | After Registration |
|--------|--------------------|--------------------|
| Fitness | Low (~0.05) | High (~0.95+) |
| Inlier RMSE | High | Low (< 0.001) |
| Chamfer Distance | High | Significantly reduced |

### Key Metrics

- **Alignment Error** — Evaluates registration quality using Open3D's `evaluate_registration()`.
- **RMSE** — Root Mean Square Error of point-to-point correspondences after alignment.
- **ICP Convergence Time** — Wall-clock time for ICP to converge (typically < 1 s).
- **Chamfer Distance** — Bidirectional mean nearest-neighbour distance between clouds.

Results are saved in `data/output/pipeline_report.json` after each run.

---

## Configuration

All parameters are in `configs/default.yaml`:

```yaml
calibration:
  board_size: [9, 6]          # Checkerboard inner corners
  square_size: 0.025          # Square side in metres

registration:
  icp:
    method: point_to_plane    # or point_to_point
    max_iterations: 200
    max_correspondence_distance: 0.05

fusion:
  depth_scale: 1000.0         # Raw depth → metres
  depth_trunc: 3.0            # Max depth (metres)
```

---

## Project Structure

```
Sensor-Calibration-Fusion/
├── configs/
│   └── default.yaml              # Central configuration
├── data/                         # Generated / raw data (git-ignored)
├── ros_nodes/
│   ├── calibration_node.py       # Simulated ROS calibration publisher
│   └── fusion_node.py            # Simulated ROS fusion subscriber
├── scripts/
│   ├── calibrate.py              # CLI — camera calibration
│   ├── register.py               # CLI — point cloud registration
│   ├── fuse.py                   # CLI — sensor fusion
│   ├── run_pipeline.py           # End-to-end automation
│   └── generate_sample_data.py   # Synthetic test data
├── src/
│   ├── calibration/
│   │   ├── calibration_data.py   # CalibrationResult dataclass
│   │   ├── extrinsic.py          # Stereo extrinsic calibration
│   │   └── intrinsic.py          # Checkerboard intrinsic calibration
│   ├── fusion/
│   │   ├── depth_to_pointcloud.py
│   │   ├── multi_sensor_fusion.py
│   │   └── rgb_depth_fusion.py
│   ├── registration/
│   │   ├── global_registration.py  # FPFH + RANSAC
│   │   ├── icp.py                  # ICP (point-to-point / point-to-plane)
│   │   └── metrics.py             # Alignment error, RMSE, Chamfer
│   ├── transforms/
│   │   ├── coordinate_frames.py   # Frame conversions
│   │   ├── homogeneous.py         # Homogeneous matrix builders
│   │   └── rigid_transform.py     # RigidTransform class (SE(3))
│   └── utils/
│       ├── io_utils.py            # Load/save images, clouds, configs
│       ├── logger.py              # Configurable logging
│       └── visualization.py       # Matplotlib / Open3D viz helpers
├── requirements.txt
└── README.md
```

---

## ROS Integration

The `ros_nodes/` directory provides **simulated ROS nodes** that work without a real ROS installation:

```bash
# Publish calibration data
python ros_nodes/calibration_node.py --calibration data/output/intrinsics.json

# Fuse point clouds
python ros_nodes/fusion_node.py --clouds data/point_clouds/source.ply data/point_clouds/target.ply
```

When deployed in a real ROS environment, replace the shim classes with actual `rospy` publishers/subscribers.

---

## License

MIT

---

## Acknowledgements

Built with [OpenCV](https://opencv.org/), [Open3D](http://www.open3d.org/), [SciPy](https://scipy.org/), and [NumPy](https://numpy.org/).
