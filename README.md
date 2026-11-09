# Workplace Safety Sentinel

A starter prototype for Hack With Bangalore 4.0 using Python and YOLO to process workplace images or video and display object detections.

**Current limitation:** the default `yolo11n.pt` model is a general COCO object detector. It can detect people and common objects, but it does **not** establish whether a worker is wearing a helmet or safety vest. PPE-related alerts require a verified PPE-trained model with suitable class labels.

**PipeGen:** PipeGen was not used because an NVIDIA GPU was unavailable in our development environment. This prototype supports CPU inference; Apple Silicon users may also test `mps` if their installed PyTorch/Ultralytics versions support it.

## Requirements

- Python 3.10 or newer recommended
- pip
- A sample image or video for testing
- Internet access on the first run if YOLO model weights need to download

## Setup

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

Image:

```bash
python main.py --source path/to/workplace.jpg
```

Video:

```bash
python main.py --source path/to/workplace.mp4
```

Webcam:

```bash
python main.py --source 0
```

The first run uses the lightweight `yolo11n.pt` model and may download its weights automatically. Annotated results are saved under `outputs/`. Potential risk events are appended to `incidents.json`. Generic person detections are not recorded as safety violations.

To use a PPE-trained model, supply its local model path:

```bash
python main.py --source path/to/workplace.mp4 --model path/to/ppe_model.pt
```

Only use a model after checking its class labels and confirming that it was trained for the intended PPE conditions. The current risk-label mapping in `main.py` may need to be adjusted to match those labels.

## Incident log

`incidents.json` contains model-flagged potential safety-risk events. Each event is a prompt for human review, not a definitive safety determination. If no supported risk class is detected, the file remains an empty array or retains earlier logged events.

## Evidence

Save a 1–2 minute recording of the actual application running in `evidence/` only if the submission rules permit committing the video to GitHub. Otherwise submit it through the organizer's requested channel.

## Limitations

- This is a prototype, not a certified safety system.
- The default model is not trained to determine PPE compliance.
- False positives and false negatives are possible.
- Human review is required before treating any flag as an incident.
- Test only with media you have permission to use.

## Team

Add team members and their contributions here.
