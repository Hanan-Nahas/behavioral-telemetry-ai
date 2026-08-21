# BRAINTEK — Child Initial Case Intake & Behaviour Analysis

A non-clinical Streamlit prototype for **children aged 5–15**. The application reports **observable behavioural indicators / possible states**, never confirmed medical diagnoses.

## Supported indicator categories

- Autism-related behavioural indicators
- Stress
- Hyperactivity
- Lack of focus
- Low mood/depression-related indicators

## What the revised demo demonstrates

- BRAINTEK-branded introduction and requested title
- Child age intake restricted to 5–15
- Face + full-body pose analysis using MediaPipe
- Temporal analysis over an approximately one-minute prerecorded clip
- Rolling result updates while the video plays
- Explanations covering facial cues, body movement, posture, activity and attention proxies
- Child-appropriate recommendations: suitable videos, simple games, stories, calming play, drawing/colouring
- Public YouTube URL input or local MP4 upload
- Live screen-recording friendly UI; the presenter's camera overlay is not part of the application

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the strongest demo, use a family-friendly prerecorded clip in which a child remains visible from face through body for at least one minute. The app can download a public YouTube URL with `yt-dlp`, or you can upload an MP4 directly.

## Safety / scope

This prototype does **not** diagnose autism, ADHD, depression, stress disorders, or other medical conditions. Results are surface-level behavioural observations and possible states only. A qualified clinician/caregiver remains responsible for interpretation and decisions involving a child.
