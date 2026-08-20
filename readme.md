# BRAINTEK AI-Assisted Camera Experience — Prototype

Non-Clinical Behavioral Telemetry & Closed-Loop Intervention System, built for the
24-Hour AI Solution Design Challenge.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place three short demo clips in an `assets/` folder next to `app.py`:

- `assets/agitation.mp4` — Case Study 1 (agitation / sensory-overload profile)
- `assets/fatigue.mp4` — Case Study 2 (low-engagement / flat-affect profile)
- `assets/recovery.mp4` — used for the Phase 2 post-action reassessment in Video Mode

If any asset or dependency (opencv, mediapipe, deepface) or the webcam is
unavailable, the app **does not crash** — it transparently swaps in a
labeled mathematical simulation of the telemetry vectors so the full
Phase 1 → Action → Phase 2 flow can still be demonstrated.

## Run

```bash
streamlit run app.py
```

## What this prototype does

- **Mode A (Video):** replays a pre-recorded case-study clip, runs MediaPipe
  Face Mesh (Eye Aspect Ratio, blink-rate proxy, head pose) and DeepFace
  (facial affect) frame-by-frame, and aggregates a composite Distress /
  Engagement score.
- **Mode B (Webcam):** does the same live from `cv2.VideoCapture(0)`.
- **Action Router:** maps the composite score to one of three low-risk,
  reversible actions — a breathing visualizer, a cognitive-stimulation
  prompt, or escalation to a human supervisor — never a diagnosis.
- **Phase 2:** re-runs the same pipeline after the action and reports a
  Delta Recovery Score plus a final "Safe to Proceed" / "Escalate" badge.
- **Audit log:** every session (mode, telemetry, action, outcome) is kept
  in memory and downloadable as CSV; no video/biometric frames are ever
  written to disk.

## Non-clinical boundaries (see in-app banner)

This tool infers nothing about ASD, ADHD, or clinical depression. It only
reasons over surface-level behavioral signals (blink rate, gaze/head pose,
coarse facial affect) to select a supportive, reversible, low-stimulation
action, and defaults to human escalation whenever detection confidence is
low or distress does not measurably improve after the action. It is a
technical prototype, not a validated clinical or safeguarding instrument.