# Extract Sharpest Frame

Python-only tool for extracting sharp frames from a video.

This repository provides two entry points:

- `extract_sharpest_frame.py`: command-line interface
- `extract_sharpest_frame_gui.py`: desktop GUI with Japanese/English switching

The implementation uses OpenCV to read video frames, calculates sharpness with the variance of the Laplacian, groups frames by chunk, and saves the sharpest frame from each chunk as a JPEG or PNG.

---
**Windows binary edition with GUI is sold on BOOTH. No need python command and easy to run!**
- [BOOTH URL] https://kotohibi-cg.booth.pm/ 
- [Gumroad URL] https://kotohibi.gumroad.com/
  - Only binary edition can support the following features
    - Generating masks by YOLO automatically (person, car, bus, etc...)
      - More accurate SfM for Metashape and others
      - <img src="./image/yolo_1.png" alt="GUI Screenshot" width="640" />
    - Faster frame extraction (Multi process optimization)
    - V0.4.0 
    Exclude similar frames. This is useful when movement speed during shooting is irregular.
      - <img src="./image/similar_frame.png" alt="GUI Screenshot" width="640" />
      - https://x.com/kotohibi_3d/status/2039716550194950248
    - V0.5.0
      - Custom mask, video range, save and load config
      - https://x.com/kotohibi_3d/status/2041150636797092322
    - V0.6.0
      - Added mask only mode for still images
      - Able to review and adjust the effect of eliminating similar frames before execution.
      - https://x.com/kotohibi_3d/status/2046282840883826952

---
Refer to the detail workflow  
- [[English]Easy&Fast 3D Gaussian Splatting workflow with 360 Camera](https://zenn.dev/kotohibi/articles/409bc16876b9e0)
- [[日本語]Easy&Fast 3D Gaussian Splatting workflow with 360 Camera](https://zenn.dev/kotohibi/articles/28b137f1873921)


## Features

- 100% Python workflow with no `ffmpeg` executable dependency
- Extract one sharp frame per chunk with `--chunk-size`
- Multiprocess metadata extraction with configurable worker count
- Reuse existing `_sharpness_metadata.csv` when available
- Regenerate metadata when needed
- `--analysis-only` mode for metadata creation only
- GUI with English/Japanese switching
- GUI confirmation dialog when metadata already exists
- GUI stop button for cancelling a running job
- `tqdm`-style progress output in the GUI log area
- Optional video range selection by frame number or seconds
- Optional similar-frame review/exclusion for irregular camera movement
- Optional custom-mask-aware sharpness scoring and mask export
- Optional YOLO-based object mask generation when `ultralytics` is installed separately
- Mask-only mode for still images
- JSON config load/save from CLI and GUI

## GUI
<img src="./image/gui.png" alt="GUI Screenshot" width="640" />


## Requirements

- Python 3.8+
- Packages from `requirements.txt`

Current Python dependencies:

- `opencv-python`
- `tqdm`

`extract_sharpest_frame_gui.py` uses Tkinter, which is included with standard desktop Python installations in most environments.

## Installation

Clone the repository:

```bash
git clone https://github.com/Kotohibi/Extract_sharpest_frame.git
cd Extract_sharpest_frame
pip install -r requirements.txt
```

## CLI Usage

Basic example:

```bash
python extract_sharpest_frame.py --video /path/to/video.mp4
```

Windows PowerShell example:

```powershell
python .\extract_sharpest_frame.py --video C:\path\to\video.mp4
```

Save output to a custom folder:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --output-dir ./sharp_frames
```

Extract one frame every 30 frames:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --chunk-size 30
```

Analyze metadata with 4 workers:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --workers 4
```

Create metadata only:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --output-dir ./sharp_frames \
  --analysis-only
```

Analyze only a video range:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --start-time 5 \
  --end-time 30
```

Review and exclude overly similar selected frames:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --similarity-threshold 0.985
```

The review is saved as `_similar_frame_review.csv`. Use `--review-similarity-only` to create only the review CSV before extracting frames.

Use a custom mask for sharpness scoring and export matching mask files:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --custom-mask /path/to/mask.png \
  --save-masks
```

Generate automatic object masks with YOLO:

```bash
pip install ultralytics
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --yolo-model yolov8n-seg.pt \
  --yolo-classes person,car,bus
```

Generate masks for still images only:

```bash
python extract_sharpest_frame.py \
  --mask-only-images image_001.jpg image_002.jpg \
  --custom-mask /path/to/mask.png \
  --mask-output-dir ./masks
```

Save and load JSON config:

```bash
python extract_sharpest_frame.py \
  --video /path/to/video.mp4 \
  --chunk-size 30 \
  --save-config config.json

python extract_sharpest_frame.py --config config.json
```

## GUI Usage

Start the GUI:

```bash
python extract_sharpest_frame_gui.py
```

Windows PowerShell example:

```powershell
python .\extract_sharpest_frame_gui.py
```

GUI behavior:

- Select a video file and output folder
- Change UI language between English and Japanese
- Run extraction or analysis-only mode
- Set worker count for metadata extraction
- Set a frame range, similarity threshold, custom mask, YOLO model, and extra CLI arguments
- Save and load GUI settings as JSON config
- If `_sharpness_metadata.csv` already exists, choose whether to reuse it
- Stop a running job with the `Stop` button
- View progress and logs in the log area

## CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--video` | string | required | Input video file path |
| `--chunk-size` | int | `30` | Select 1 frame per N frames |
| `--scale-width` | int | `1920` | Resize wider frames to this width for analysis |
| `--workers` | int | `4` | Worker count for metadata extraction multiprocessing |
| `--output-dir` | string | `sharp_frames` | Output directory |
|`--output-format`| string | `png` | Output file format, png or jpg |
| `--output-pattern` | string | `output_frame_%05d.jpg` | Output filename pattern |
| `--jpeg-quality` | int or percent | `95` | JPEG quality percentage from `1` to `100` |
| `--analysis-only` | flag | off | Create metadata only without writing JPEG or PNG files |
| `--start-frame` | int | `0` | First frame to analyze |
| `--end-frame` | int | unset | Stop before this frame |
| `--start-time` | float | unset | Start time in seconds |
| `--end-time` | float | unset | End time in seconds |
| `--custom-mask` | string | unset | Static mask image used for sharpness scoring and mask export |
| `--similarity-threshold` | float | `0` | Drop selected frames whose similarity to the previous kept frame is at or above this threshold |
| `--review-similarity-only` | flag | off | Write `_similar_frame_review.csv` and skip image extraction |
| `--save-masks` | flag | off | Save `*_mask.png` files for extracted frames using the custom mask |
| `--yolo-model` | string | unset | Optional Ultralytics YOLO model path/name for automatic object masks |
| `--yolo-classes` | string | `person,bicycle,car,motorcycle,bus,truck` | COCO class names or IDs to include in YOLO masks |
| `--mask-only-images` | strings | unset | Still-image mask-only mode |
| `--mask-output-dir` | string | `masks` | Output folder for still-image mask-only mode |
| `--config` | string | unset | Load options from JSON config |
| `--save-config` | string | unset | Save resolved options to JSON config |

## Output Files

The tool writes the following files into `--output-dir`:

- `_sharpness_metadata.csv`: frame number and sharpness score for the analyzed video
- `_sharpness_metadata.json`: analysis options used to determine whether metadata can be safely reused
- `_similar_frame_review.csv`: similar-frame keep/drop review when enabled
- `output_frame_00001.jpg`, `output_frame_00002.jpg`, ...: extracted sharp frames
- `output_frame_00001_mask.png`, `output_frame_00002_mask.png`, ...: optional mask images

## How It Works

1. Open the input video with OpenCV.
2. Compute a sharpness score for each frame using the variance of the Laplacian.
3. Save frame scores to `_sharpness_metadata.csv`.
4. Split frames into chunks based on `--chunk-size`.
5. Choose the highest-scoring frame in each chunk.
6. Save the selected frames as JPEG images.

If metadata already exists, it can be reused instead of analyzing the video again.

When `--workers` is greater than `1`, metadata extraction is split across multiple processes and merged back in original frame order.

YOLO mask generation is optional and requires installing `ultralytics` separately. If a YOLO segmentation model is used, instance masks are written; with detection-only models, bounding-box masks are written.

## Notes

- A smaller `--chunk-size` produces more extracted frames.
- Lowering `--scale-width` can improve performance on high-resolution videos.
- The sharpness score is Laplacian-based, so results will differ from ffmpeg `blurdetect` output.
- If a GUI job is cancelled during metadata generation, the partial metadata file is removed.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
