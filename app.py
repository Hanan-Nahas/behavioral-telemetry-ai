"""
BRAINTEK - AI-POWERED CHILD INITIAL CASE INTAKE AND BEHAVIOUR ANALYSIS SOLUTION
Non-clinical prototype for children aged 5-15.

The system reports observable behavioural indicators / possible states only.
It is NOT a medical diagnostic system and must not be used to diagnose autism,
ADHD, depression, stress disorders, or any other condition.
"""

import html
import math
import os
import subprocess
import tempfile
import time
import urllib.parse
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st

# Optional MediaPipe: face + full-body pose.
try:
    import mediapipe as mp
    MP_AVAILABLE = hasattr(mp, "solutions")
    if MP_AVAILABLE:
        MP_FACE = mp.solutions.face_mesh
        MP_POSE = mp.solutions.pose
    else:
        MP_FACE = MP_POSE = None
except Exception:
    MP_AVAILABLE = False
    MP_FACE = MP_POSE = None

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

st.set_page_config(
    page_title="BrainTek Child Behaviour Analysis",
    page_icon="🧠",
    layout="wide",
)

TITLE = "AI-POWERED CHILD INITIAL CASE INTAKE AND BEHAVIOUR ANALYSIS SOLUTION"
AGE_RANGE = "5–15 years"
CASES = [
    "Autism-related behavioural indicators",
    "Stress",
    "Hyperactivity",
    "Lack of focus",
    "Low mood/depression-related indicators",
]

# Public, family-friendly starting point. The demo also accepts another URL or
# an uploaded MP4 so the evaluator can select a clip that visibly contains a
# child's face and body for approximately one minute.
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=7SWO81mvZAc"

CASE_RECOMMENDATIONS = {
    "Autism-related behavioural indicators": [
        "Read or listen to a short familiar story",
        "Do a predictable drawing or colouring activity",
        "Play a simple, low-pressure matching game",
    ],
    "Stress": [
        "Watch a suitable calming video",
        "Do a short calming play activity",
        "Try slow breathing with an adult",
    ],
    "Hyperactivity": [
        "Play a short movement-based game",
        "Do a simple stretch or movement break",
        "Use a structured turn-taking game",
    ],
    "Lack of focus": [
        "Play a simple matching or memory game",
        "Read or listen to a short story",
        "Try a short drawing or colouring task",
    ],
    "Low mood/depression-related indicators": [
        "Watch a suitable positive video with a caregiver",
        "Read or listen to a comforting story",
        "Do a gentle drawing or colouring activity",
    ],
}

CASE_DESCRIPTIONS = {
    "Autism-related behavioural indicators": "Possible pattern of reduced social-orienting cues, repetitive movement or limited variation in attention. These observations are not evidence of autism and require qualified professional assessment if concerns persist.",
    "Stress": "Possible short-term stress-related pattern based on observable tension, restlessness, gaze changes, posture and facial affect during the analysed period.",
    "Hyperactivity": "Possible elevated activity pattern based on movement frequency, body displacement and repeated posture changes over time.",
    "Lack of focus": "Possible reduced-attention pattern based on repeated gaze diversion, head orientation changes and low sustained engagement with the viewed activity.",
    "Low mood/depression-related indicators": "Possible low-mood-related pattern based on sustained subdued facial affect, reduced movement and downward or withdrawn posture. This is not a diagnosis of depression.",
}


def init_state():
    defaults = {
        "history": deque(maxlen=30),
        "last_result": None,
        "video_source": DEFAULT_YOUTUBE_URL,
        "selected_case": CASES[0],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def esc(value):
    return html.escape(str(value))


def render_intro():
    st.markdown(
        """
        <div style="padding:34px 30px 28px;border-radius:18px;
                    background:linear-gradient(135deg,#071a2d 0%,#123b52 52%,#0d5c55 100%);
                    color:white;text-align:center;margin-bottom:18px">
            <div style="font-size:26px;font-weight:900;letter-spacing:2px">BRAINTEK</div>
            <div style="font-size:30px;font-weight:900;line-height:1.18;margin-top:14px">AI-POWERED CHILD INITIAL CASE INTAKE AND BEHAVIOUR ANALYSIS SOLUTION</div>
            <div style="font-size:15px;margin-top:12px;opacity:.88">Non-clinical behavioural intake prototype • Target age: children 5–15 years</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.warning(
        "NON-CLINICAL PROTOTYPE: Results are observable behavioural indicators or possible states, not confirmed medical diagnoses. "
        "The system should support — never replace — a qualified clinician, caregiver or safeguarding professional."
    )


def face_features(face_result, frame_shape):
    h, w = frame_shape[:2]
    if not face_result or not face_result.multi_face_landmarks:
        return {"face_present": False, "ear": 0.0, "yaw": 0.0, "pitch": 0.0, "smile": 0.0, "face_area": 0.0}

    lm = face_result.multi_face_landmarks[0].landmark
    def p(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=float)

    # Eye aspect ratio.
    left = [33, 160, 158, 133, 153, 144]
    right = [362, 385, 387, 263, 373, 380]
    def ear(ids):
        a, b, c, d, e, f = [p(i) for i in ids]
        horizontal = np.linalg.norm(a - d) or 1.0
        return float((np.linalg.norm(b-f) + np.linalg.norm(c-e)) / (2*horizontal))

    # Normalized face geometry provides a lightweight head orientation cue.
    nose = p(1)
    chin = p(152)
    le = p(33)
    re = p(263)
    mouth_l = p(61)
    mouth_r = p(291)
    eye_mid = (le + re) / 2
    eye_width = np.linalg.norm(le-re) or 1.0
    yaw = float(np.clip((nose[0]-eye_mid[0]) / eye_width * 90, -90, 90))
    pitch = float(np.clip((nose[1]-eye_mid[1]) / eye_width * 90, -90, 90))
    mouth_width = np.linalg.norm(mouth_l-mouth_r) or 1.0
    mouth_open = np.linalg.norm(p(13)-p(14)) / mouth_width
    face_x = [q.x for q in lm]
    face_y = [q.y for q in lm]
    face_area = max(0.0, (max(face_x)-min(face_x)) * (max(face_y)-min(face_y)))

    return {
        "face_present": True,
        "ear": round((ear(left)+ear(right))/2, 4),
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "smile": round(float(np.clip(mouth_open, 0, 1)), 3),
        "face_area": round(face_area, 4),
    }


def body_features(pose_result, frame_shape):
    h, w = frame_shape[:2]
    if not pose_result or not pose_result.pose_landmarks:
        return {"body_present": False, "movement": 0.0, "posture": "not detected", "activity": 0.0, "attention_proxy": 0.0}

    pts = pose_result.pose_landmarks.landmark
    ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    xy = np.array([[pts[i].x, pts[i].y] for i in ids if pts[i].visibility > 0.35])
    if len(xy) < 4:
        return {"body_present": False, "movement": 0.0, "posture": "not detected", "activity": 0.0, "attention_proxy": 0.0}

    shoulder_y = np.mean([pts[11].y, pts[12].y])
    hip_y = np.mean([pts[23].y, pts[24].y])
    torso = max(abs(hip_y-shoulder_y), 0.05)
    posture = "upright" if shoulder_y < hip_y else "reclined/low"
    activity = float(np.clip(np.std(xy[:, 1]) * 420, 0, 100))
    return {
        "body_present": True,
        "movement": 0.0,  # filled from temporal difference by the caller
        "posture": posture,
        "activity": round(activity, 1),
        "attention_proxy": round(float(np.clip(100 - abs(shoulder_y-0.45)*130, 0, 100)), 1),
        "body_center": (float(np.mean(xy[:,0])), float(np.mean(xy[:,1]))),
    }


def analyze_frame(frame, face_mesh, pose):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb) if face_mesh else None
    pose_result = pose.process(rgb) if pose else None
    face = face_features(face_result, frame.shape)
    body = body_features(pose_result, frame.shape)

    if face["face_present"]:
        face_score = 100.0
        if abs(face["yaw"]) > 30:
            face_score -= 20
        if abs(face["pitch"]) > 30:
            face_score -= 10
    else:
        face_score = 15.0

    confidence = round(float(np.clip((face_score + (85 if body["body_present"] else 20))/2, 0, 99)), 1)
    return face, body, confidence


def classify(samples):
    if not samples:
        return CASES[1], 0.0, "Insufficient observable data."

    face_present = np.mean([s["face"]["face_present"] for s in samples])
    body_present = np.mean([s["body"]["body_present"] for s in samples])
    avg_activity = np.mean([s["body"].get("activity", 0) for s in samples])
    avg_movement = np.mean([s["body"].get("movement", 0) for s in samples])
    avg_yaw = np.mean([abs(s["face"].get("yaw", 0)) for s in samples])
    avg_pitch = np.mean([s["face"].get("pitch", 0) for s in samples])
    avg_ear = np.mean([s["face"].get("ear", 0.25) for s in samples])
    low_affect = np.mean([1 if s["face"].get("smile", 0) < 0.12 else 0 for s in samples])
    looking_away = np.mean([1 if abs(s["face"].get("yaw", 0)) > 25 else 0 for s in samples])
    low_posture = np.mean([1 if s["body"].get("posture") == "reclined/low" else 0 for s in samples])

    scores = {
        "Stress": 0.0,
        "Hyperactivity": 0.0,
        "Lack of focus": 0.0,
        "Low mood/depression-related indicators": 0.0,
        "Autism-related behavioural indicators": 0.0,
    }
    scores["Hyperactivity"] = min(100, avg_activity*0.65 + avg_movement*0.8)
    scores["Stress"] = min(100, avg_movement*0.45 + looking_away*35 + max(0, 0.22-avg_ear)*140)
    scores["Lack of focus"] = min(100, looking_away*55 + max(0, 55-avg_movement)*0.35 + max(0, 45-avg_ear*150)*0.25)
    scores["Low mood/depression-related indicators"] = min(100, low_affect*45 + low_posture*35 + max(0, 25-avg_movement)*0.45 + max(0, avg_pitch-5)*0.35)
    scores["Autism-related behavioural indicators"] = min(100, looking_away*25 + max(0, 30-avg_movement)*0.35 + max(0, 0.23-avg_ear)*90)

    label = max(scores, key=scores.get)
    confidence = round(float(np.clip(scores[label] * 0.72 + face_present*18 + body_present*10, 0, 96)), 1)
    if face_present < 0.5 or body_present < 0.5:
        confidence = min(confidence, 55.0)

    return label, confidence, scores


def observed_description(samples, label):
    if not samples:
        return "No sustained video evidence was available."
    face = np.mean([s["face"]["face_present"] for s in samples])
    body = np.mean([s["body"]["body_present"] for s in samples])
    movement = np.mean([s["body"].get("movement",0) for s in samples])
    activity = np.mean([s["body"].get("activity",0) for s in samples])
    yaw = np.mean([abs(s["face"].get("yaw",0)) for s in samples])
    pitch = np.mean([s["face"].get("pitch",0) for s in samples])
    smile = np.mean([s["face"].get("smile",0) for s in samples])
    posture_counts = Counter(s["body"].get("posture","not detected") for s in samples)
    posture = posture_counts.most_common(1)[0][0]
    return (
        f"Across {len(samples)} sampled observations, the system tracked the face ({face*100:.0f}% of samples) "
        f"and body ({body*100:.0f}% of samples). Average body activity was {activity:.0f}/100 with temporal movement "
        f"at {movement:.0f}/100. Head orientation varied by about {yaw:.0f}° on average, average pitch was {pitch:.0f}°, "
        f"the dominant posture was {posture}, and the facial-expression proxy remained at {smile:.2f}. "
        f"These observable cues contributed to the possible state '{label}'."
    )


def download_youtube(url):
    """Download a single progressive MP4 when available; returns local path."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in {"www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com"}:
        raise ValueError("Please provide a YouTube URL.")
    folder = tempfile.mkdtemp(prefix="braintek_video_")
    output = os.path.join(folder, "source.%(ext)s")
    cmd = [
        "yt-dlp", "--no-playlist", "-f", "best[ext=mp4][height<=480]/best[ext=mp4]",
        "-o", output, url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
    candidates = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]
    if not candidates:
        raise RuntimeError("YouTube download did not return a progressive MP4. Upload an MP4 instead.")
    return candidates[0]


def draw_overlay(frame, result, elapsed, label):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w, 88), (8,20,35), -1)
    cv2.putText(out, f"BRAINTEK | {AGE_RANGE} | LIVE ANALYSIS", (18,28), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2)
    cv2.putText(out, f"{elapsed:04.1f}s  Face+Body  Possible state: {label}", (18,58), cv2.FONT_HERSHEY_SIMPLEX, .55, (120,235,220), 2)
    cv2.putText(out, f"Confidence: {result:.0f}%", (18,80), cv2.FONT_HERSHEY_SIMPLEX, .46, (210,220,230), 1)
    return out


def run_video(path, duration_seconds=60):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    native_duration = total / fps if total else 0
    if native_duration < 1:
        raise RuntimeError("The video is too short or has no readable duration.")

    face_mesh = MP_FACE.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5) if MP_AVAILABLE else None
    pose = MP_POSE.Pose(model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5) if MP_AVAILABLE else None
    previous_center = None
    samples = []
    start = time.time()
    display = st.empty()
    metrics = st.empty()
    progress = st.progress(0)
    max_duration = min(float(duration_seconds), native_duration)
    last_render = 0.0
    frame_index = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            video_elapsed = frame_index / fps
            if video_elapsed >= max_duration:
                break
            face, body, confidence = analyze_frame(frame, face_mesh, pose)
            center = body.get("body_center")
            movement = 0.0
            if center and previous_center:
                movement = float(np.clip(np.linalg.norm(np.array(center)-np.array(previous_center))*600, 0, 100))
            previous_center = center
            body["movement"] = movement
            sample = {"face":face, "body":body, "confidence":confidence}
            samples.append(sample)

            # Recompute a rolling state every ~1 second so the result develops over time.
            if video_elapsed - last_render >= 1.0 or frame_index == 0:
                label, conf, _ = classify(samples[-min(len(samples), 80):])
                annotated = draw_overlay(frame, conf, video_elapsed, label)
                display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                metrics.markdown(
                    f"**Possible state:** {label}  \n"
                    f"**Detection confidence:** {conf:.0f}%  \n"
                    f"**Face:** {'detected' if face['face_present'] else 'not detected'} • "
                    f"**Body:** {'detected' if body['body_present'] else 'not detected'} • "
                    f"**Posture:** {body.get('posture','not detected')} • "
                    f"**Activity:** {body.get('activity',0):.0f}/100 • **Movement:** {movement:.0f}/100"
                )
                progress.progress(min(1.0, video_elapsed/max_duration))
                last_render = video_elapsed
            frame_index += 1
            # Make the screen recording visibly demonstrate continuous real-time progression.
            time.sleep(max(0.0, 0.025 - (time.time()-start-frame_index/fps)))
    finally:
        cap.release()
        if face_mesh: face_mesh.close()
        if pose: pose.close()
    progress.progress(1.0)
    return samples


def result_panel(samples):
    label, confidence, scores = classify(samples)
    description = observed_description(samples, label)
    recommendations = CASE_RECOMMENDATIONS[label]
    st.session_state.last_result = {"label":label,"confidence":confidence,"description":description}
    st.markdown("### AI Behavioural Result")
    c1, c2 = st.columns([1,2])
    c1.metric("Possible state / indicator", label)
    c2.metric("Detection confidence", f"{confidence:.0f}%")
    st.info(description)
    st.caption("Interpretation: behavioural indicator / possible state only — not a confirmed diagnosis.")

    st.markdown("### Child-Appropriate Support Recommendations")
    for rec in recommendations:
        st.write(f"• {rec}")

    st.markdown("### Observed Cue Breakdown")
    recent = samples[-min(len(samples),120):]
    if recent:
        face_rate = np.mean([s['face']['face_present'] for s in recent])*100
        body_rate = np.mean([s['body']['body_present'] for s in recent])*100
        movement = np.mean([s['body'].get('movement',0) for s in recent])
        activity = np.mean([s['body'].get('activity',0) for s in recent])
        yaw = np.mean([abs(s['face'].get('yaw',0)) for s in recent])
        posture = Counter(s['body'].get('posture','not detected') for s in recent).most_common(1)[0][0]
        st.dataframe({
            "Cue":["Face visibility","Body visibility","Facial orientation","Posture","Movement","Activity level"],
            "Observed over video":[f"{face_rate:.0f}%",f"{body_rate:.0f}%",f"{yaw:.0f}° average variation",posture,f"{movement:.0f}/100",f"{activity:.0f}/100"],
        }, use_container_width=True, hide_index=True)


def render_video_controls():
    st.subheader("🎥 Continuous prerecorded-video analysis")
    st.caption("The demo analyses a real video continuously for approximately one minute and updates the possible state over time. Use a clip where the child’s face and body are both visible.")
    source_type = st.radio("Video source", ["YouTube URL", "Upload MP4"], horizontal=True)
    path = None
    if source_type == "YouTube URL":
        url = st.text_input("Public YouTube video", value=st.session_state.video_source)
        st.session_state.video_source = url
        st.caption("Suggested starting point: a child-friendly drawing video. For the face+body requirement, choose a clip where the child is visibly on camera.")
        if st.button("⬇️ Prepare YouTube video"):
            try:
                with st.spinner("Downloading a progressive MP4 for local analysis…"):
                    path = download_youtube(url)
                st.session_state.video_path = path
                st.success("Video prepared. Click Start 60-Second Analysis.")
            except Exception as e:
                st.error(f"Could not prepare the YouTube video: {e}")
    else:
        uploaded = st.file_uploader("Upload a prerecorded MP4", type=["mp4","mov","avi"])
        if uploaded:
            folder = tempfile.mkdtemp(prefix="braintek_upload_")
            path = os.path.join(folder, uploaded.name)
            with open(path,"wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.video_path = path

    if st.session_state.get("video_path") and st.button("▶️ Start approximately 1-minute continuous analysis", type="primary"):
        try:
            with st.spinner("Running continuous face + body analysis…"):
                samples = run_video(st.session_state.video_path, 60)
            st.session_state.analysis_samples = samples
            result_panel(samples)
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    if st.session_state.get("analysis_samples"):
        if st.button("🔄 Show final result again"):
            result_panel(st.session_state.analysis_samples)


def render_case_selector():
    st.subheader("Initial Case Intake")
    age = st.selectbox("Child age", list(range(5,16)), index=5)
    case = st.selectbox("Behavioural indicator to demonstrate", CASES)
    st.session_state.selected_case = case
    st.info(f"Target population: children aged 5–15. Current intake age: {age}. The selected case is a demonstration category; the AI does not confirm a diagnosis.")
    st.markdown(f"**Case description:** {CASE_DESCRIPTIONS[case]}")


def main():
    init_state()
    render_intro()
    render_disclaimer()
    render_case_selector()
    st.divider()
    render_video_controls()
    st.divider()
    st.markdown("### Five supported behavioural-indicator categories")
    st.write(" • ".join(CASES))
    st.caption("Live screen recording recommendation: record only the browser/demo screen with your voice-over; keep your webcam/camera overlay off.")
    st.caption(f"MediaPipe face + body pose: {'available' if MP_AVAILABLE else 'unavailable — install mediapipe'} • DeepFace facial affect: {'available' if DEEPFACE_AVAILABLE else 'optional'}")


if __name__ == "__main__":
    main()
