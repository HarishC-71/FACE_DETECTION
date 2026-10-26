# Real-Time Human Emotion Detection

A desktop application that reads your webcam, finds a face, and estimates the
**visible facial expression** in real time (Happy, Sad, Angry, Fearful, Disgust,
Surprised, Neutral). Built with Python, OpenCV, the FER library and Tkinter.

> **Important:** the output is an estimate of a facial *expression* from a
> pretrained model. It is not a measurement of what a person really feels, and
> it can be wrong (lighting, camera angle, glasses, culture and individual
> differences all matter).

## Project structure

```
human_emotion_recognition/
├── main.py               Tkinter GUI, camera and worker threads, probability bars
├── emotion_detector.py   Face detection + emotion prediction (FER wrapper)
├── requirements.txt      Pinned dependencies
├── README.md
├── outputs/              Captured images are saved here
└── screenshots/          Put your own screenshots here for the report
```

## Installation on Windows 11 (VS Code)

1. **Install Python 3.10 or 3.11 (64-bit)** from <https://www.python.org/downloads/windows/>.
   Tick **"Add python.exe to PATH"**. Do **not** use Python 3.12 or newer:
   the pinned TensorFlow 2.15 does not support it. Check with `py --list`.
2. **Open the project folder** in VS Code: *File > Open Folder...* and choose
   `human_emotion_recognition`. Open a terminal (*Terminal > New Terminal*, PowerShell).
3. **Create and activate a virtual environment** (use `-3.10` if that is the version you installed):
   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   If activation is blocked, run once:
   `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
4. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   TensorFlow is large (several hundred MB); this can take a few minutes.
5. In VS Code press `Ctrl+Shift+P > Python: Select Interpreter` and pick the `venv` one.
6. **Run**:
   ```powershell
   python main.py
   ```
7. Windows may ask for camera permission. Allow it
   (*Settings > Privacy & security > Camera > Let desktop apps access your camera*).
8. Click **Start Camera** and try different expressions: smile, frown, raise
   your eyebrows with an open mouth, and so on. Use **Capture Image** to save a frame
   to `outputs/`.

## Using the interface

| Element | Meaning |
|---|---|
| Left panel | Live webcam view (mirrored) with a green box and a `Happy (0.92)` style label |
| Detected Emotion | The highest-scoring expression of the largest face |
| Emotion Probabilities | Seven coloured bars with scores that sum to about 1.00 |
| Live Detection | Up to five recent face thumbnails with their predicted expression |
| Status bar | Camera state, face found or not, camera FPS, inference time, model state |
| Buttons | Start Camera, Stop Camera, Capture Image (annotated frame), Exit |

## Sample output (illustrative)

When you smile at the camera:

```
Real-Time Human Emotion Detection
Detected Emotion: Happy

Happy       ██████████████████  0.92
Neutral     █                   0.05
Surprised   █                   0.02
Sad         ░                   0.01
Angry       ░                   0.00
Fearful     ░                   0.00
Disgust     ░                   0.00
```

A green rectangle surrounds the face with the label `Happy (0.92)`.
These numbers are an example only. Real scores are computed by the model on every run.

## How it works (algorithm and architecture)

1. **Capture** - a camera thread reads frames with OpenCV and keeps only the newest one.
2. **Face detection** - FER uses an OpenCV Haar cascade (`mtcnn=False`) to locate faces.
3. **Pre-processing** - each face is cropped, converted to grayscale and resized to 48x48 pixels.
4. **Classification** - a pretrained convolutional neural network (trained on the FER-2013
   dataset) outputs a probability for each of the seven expressions.
5. **Post-processing** - scores are renamed to the display labels, normalised to sum to 1,
   and smoothed over time (exponential moving average, alpha 0.5) so the bars do not flicker.
6. **Display** - the Tk thread draws the box, label, bars and thumbnails about 30 times a second.

Threads keep the interface responsive:

```
Camera thread --> latest frame --> Inference thread --> latest result --> Tk GUI (30 fps)
Model-loading thread (once at start-up)
```

The GUI never waits for the model. If inference is slower than the camera, frames
are simply skipped and the newest one is analysed. Face results older than 1.5 s are discarded.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open the webcam` | Close Teams/Zoom/Camera app; allow camera access in Windows privacy settings; try another USB port. The app tries camera indices 0 and 1. |
| Black or frozen video | Unplug and replug the camera; restart the app. |
| `The FER library could not be imported` | Python is probably too new. Use Python 3.10 or 3.11 and recreate the venv. |
| `No matching distribution found for tensorflow==2.15.0` | You are on Python 3.12+ or a 32-bit Python. Install 64-bit Python 3.10/3.11. |
| `No module named 'moviepy.editor'` | `pip install moviepy==1.0.3` (moviepy 2.x removed that module). |
| NumPy errors such as `_ARRAY_API not found` | `pip install numpy==1.26.4` |
| `No module named 'tkinter'` | Reinstall Python and keep the *tcl/tk and IDLE* option ticked. |
| Window says "Loading emotion model..." for a long time | The first start loads TensorFlow and can take 10-30 seconds. |
| Very low FPS | Close other heavy programs. Lower `max_width` in `emotion_detector.py` (default 480). |
| Face not detected | Face the camera, improve front lighting, move closer, remove strong backlight. |
| Wrong expression predicted | Expected sometimes; the model is small and trained on low-resolution 48x48 images. |

### If FER cannot be installed at all

Check `pip install fer` on its own. If a newer FER release uses a different layout,
`emotion_detector.py` already tries both `from fer import FER` and `from fer.fer import FER`.
As a last resort, swap the model inside `EmotionDetector` for another local library
(for example DeepFace with `detector_backend="opencv"`); only `analyze()` has to return the
same `FaceResult` list, so `main.py` needs no changes.

## What has and has not been tested

Tested in a Linux sandbox with Python 3.11, a virtual display and a **simulated camera**
(a still photo of a smiling person fed in place of the webcam): the pinned `requirements.txt`
installs in a fresh environment, the FER model loads, a face is detected and classified as Happy,
the GUI shows the box, label, bars and thumbnails, Capture Image saves a JPEG to `outputs/`,
Stop Camera and Exit release the camera cleanly. **Not tested:** a real webcam, Windows 11,
and accuracy across many real faces or expressions. Please run it on your PC and report any problem.

## Conclusion (for a project report)

This project shows that a real-time expression-recognition tool can be built with free,
offline components: OpenCV for video and face detection, a pretrained CNN for classification and
Tkinter for the interface. Separating capture, inference and display into threads keeps the
interface smooth, and temporal smoothing makes the output readable. The system estimates
only visible expressions in the seven categories and should not be used to judge a person's
real emotions, character or abilities.

### Limitations

- Trained on FER-2013 (small, low-resolution, imbalanced); accuracy on "Disgust" and "Fearful" is weak.
- Haar-cascade detection prefers frontal, well-lit faces.
- A single frame-based model with no understanding of context, speech or culture.

### Future enhancements

- Use a stronger face detector (MTCNN, MediaPipe) and a more accurate emotion model.
- Support several faces at once with per-person tracking.
- Add video-file input, session logging to CSV and an emotion-over-time chart.
- Use sequence models over several frames; add audio and text cues.
- Evaluate on a labelled dataset and report accuracy and a confusion matrix.
- Package as a Windows `.exe` with PyInstaller.
- Add privacy controls (no storage by default, explicit consent notice).
#   F A C E _ D E T E C T I O N  
 