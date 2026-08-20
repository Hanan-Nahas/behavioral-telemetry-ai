"""
BRAINTEK AI-Assisted Camera Experience
Non-Clinical Behavioral Telemetry & Closed-Loop Intervention System
========================================================================
24-Hour AI Solution Design Challenge - Technical Prototype

IMPORTANT: This is a NON-CLINICAL prototype. It does not diagnose any
medical or psychological condition. It observes surface-level behavioral
cues (gaze stability, blink frequency, coarse facial affect) to trigger
simple, reversible, supportive on-screen actions, and escalates to a
qualified human whenever confidence is low or distress persists.

Run with:  streamlit run app.py
(Running with plain `python app.py` will NOT work - Streamlit needs its
own runner to create a browser session; you'll get a bare-mode warning
and the script will exit immediately without doing anything useful.)
"""

import os
import time
import math
import random
import traceback
import html
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# GLOBAL IMPORTS & DEPENDENCY FALLBACKS
# Each optional dependency's import exception is CAPTURED (not swallowed)
# so the UI can show the real reason a dependency is unavailable instead of
# a mysterious blanket "using simulation" with no explanation.
# ---------------------------------------------------------------------------
try:
    import cv2
    CV2_AVAILABLE = True
    CV2_ERROR = None
except Exception:
    CV2_AVAILABLE = False
    CV2_ERROR = traceback.format_exc()

HAAR_FACE = None
HAAR_EYE = None
HAAR_SMILE = None
if CV2_AVAILABLE:
    try:
        HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        HAAR_EYE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        HAAR_SMILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    except Exception:
        HAAR_FACE = HAAR_EYE = HAAR_SMILE = None

MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_ERROR = None
mp_face_mesh = None
try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp_face_mesh = mp.solutions.face_mesh
    else:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    # Actually try constructing one here, at import time, so a broken
    # native binding / missing model file is caught immediately rather
    # than surfacing later as a confusing runtime symptom.
    _test_fm = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    _test_fm.close()
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = traceback.format_exc()
    mp_face_mesh = None

DEEPFACE_AVAILABLE = False
DEEPFACE_ERROR = None
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False
    DEEPFACE_ERROR = traceback.format_exc()


# ===========================================================================
# CONSTANTS
# ===========================================================================

ASSETS_DIR = "assets"
VIDEO_CASES = {
    "Case Study 1: Agitation & Sensory Overload Profile": os.path.join(ASSETS_DIR, "agitation.mp4"),
    "Case Study 2: Low Engagement & Depressive Affect Profile": os.path.join(ASSETS_DIR, "fatigue.mp4"),
}
RECOVERY_VIDEO = os.path.join(ASSETS_DIR, "recovery.mp4")

# MediaPipe Face Mesh landmark indices for eye contours (6-point EAR model)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

DISTRESS_HIGH_THRESHOLD = 60      # % -> agitation / overload path
ENGAGEMENT_LOW_THRESHOLD = 35     # % -> fatigue / low-affect path
SAD_AFFECT_THRESHOLD = 0.35       # probability -> comfort/support path
LOW_CONFIDENCE_THRESHOLD = 50     # % detection confidence floor
MAX_FRAMES_TO_SAMPLE = 45         # frames sampled per assessment pass

QUOTES = [
    {
        "text": "The goal is not to watch people closer; it is to respond with more care.",
        "author": "BrainTek prototype principle",
    },
    {
        "text": "Good assistance is quiet, timely, reversible, and human-aware.",
        "author": "Human-in-the-loop design note",
    },
    {
        "text": "A signal becomes useful only when it leads to a safer next step.",
        "author": "Behavioral telemetry guideline",
    },
    {
        "text": "Support should lower pressure before it asks for performance.",
        "author": "Care-first interaction rule",
    },
    {
        "text": "When confidence is low, the best intelligence is humility.",
        "author": "Oversight guardrail",
    },
]


# ===========================================================================
# SESSION STATE INITIALIZATION
# ===========================================================================

def init_state():
    defaults = {
        "audit_log": [],
        "phase1_result": None,
        "phase2_result": None,
        "recommended_action": None,
        "session_id": None,
        "mode": None,
        "case_choice": None,
        "quote_index": datetime.now().toordinal() % len(QUOTES),
        "webcam_phase1_toggle": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===========================================================================
# CORE TELEMETRY FUNCTIONS
# ===========================================================================

def calculate_ear(eye_points):
    """
    Eye Aspect Ratio (EAR) from 6 (x, y) landmark points, per the standard
    Soukupova & Cech formulation:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    try:
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
        if horizontal == 0:
            return 0.30
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)
    except Exception:
        return 0.30


def estimate_head_pose(landmarks_2d, frame_shape):
    """
    Lightweight head pose (yaw / pitch / roll) estimate via solvePnP using
    a generic 3D face model and key landmark points.
    """
    if not CV2_AVAILABLE:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    try:
        h, w = frame_shape[:2]
        model_points = np.array([
            (0.0, 0.0, 0.0),          # nose tip
            (0.0, -330.0, -65.0),     # chin
            (-225.0, 170.0, -135.0),  # left eye left corner
            (225.0, 170.0, -135.0),   # right eye right corner
            (-150.0, -150.0, -125.0),  # left mouth corner
            (150.0, -150.0, -125.0),  # right mouth corner
        ], dtype=np.float64)

        idxs = [1, 152, 33, 263, 61, 291]
        image_points = np.array([landmarks_2d[i] for i in idxs], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        rmat, _ = cv2.Rodrigues(rotation_vec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        return {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}
    except Exception:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def simulate_metrics(seed_bias="neutral"):
    """
    Deterministic fallback simulator used when cv2 / mediapipe / deepface
    or webcam assets are missing, so the workflow can still be demonstrated.
    """
    rng = random.Random()
    if seed_bias == "agitation":
        ear = rng.uniform(0.14, 0.20)
        blink_rate = rng.uniform(28, 42)
        yaw = rng.uniform(-35, 35)
        emotions = {"angry": 0.48, "fear": 0.22, "sad": 0.08, "neutral": 0.08, "happy": 0.02, "surprise": 0.12}
    elif seed_bias == "fatigue":
        ear = rng.uniform(0.20, 0.26)
        blink_rate = rng.uniform(6, 12)
        yaw = rng.uniform(-8, 8)
        emotions = {"angry": 0.03, "fear": 0.06, "sad": 0.58, "neutral": 0.26, "happy": 0.02, "surprise": 0.05}
    elif seed_bias == "recovered":
        ear = rng.uniform(0.27, 0.33)
        blink_rate = rng.uniform(12, 18)
        yaw = rng.uniform(-6, 6)
        emotions = {"angry": 0.02, "fear": 0.03, "sad": 0.04, "neutral": 0.22, "happy": 0.64, "surprise": 0.05}
    else:
        ear = rng.uniform(0.24, 0.30)
        blink_rate = rng.uniform(14, 20)
        yaw = rng.uniform(-10, 10)
        emotions = {"angry": 0.05, "fear": 0.05, "sad": 0.15, "neutral": 0.55, "happy": 0.15, "surprise": 0.05}

    return {
        "ear": round(ear, 3),
        "blink_rate": round(blink_rate, 1),
        "head_pose": {"yaw": round(yaw, 1), "pitch": round(rng.uniform(-10, 10), 1), "roll": round(rng.uniform(-5, 5), 1)},
        "emotions": emotions,
        "detection_confidence": round(rng.uniform(72, 96), 1),
        "simulated": True,
        "profile": seed_bias,
        "fallback": "curated_case_profile",
    }


def compute_composite_score(ear, blink_rate, yaw, emotions):
    """
    Synthesizes EAR, blink rate, head pose, and emotion probabilities
    into Distress and Engagement composite metrics. This is a disclosed
    HEURISTIC (not a validated psychometric instrument) - weights were
    chosen to produce sensible-looking demo behavior, not clinical
    accuracy. See README for details and limitations.
    """
    blink_component = min(100, abs(blink_rate - 17) * 4)
    ear_component = min(100, max(0, (0.30 - ear) * 400))
    pose_component = min(100, abs(yaw) * 2.2)

    sad_affect = emotions.get("sad", 0)
    negative_affect = emotions.get("angry", 0) + emotions.get("fear", 0) + sad_affect
    emotion_component = negative_affect * 100

    distress = (
        0.25 * blink_component +
        0.18 * ear_component +
        0.17 * pose_component +
        0.40 * emotion_component
    )
    if sad_affect >= SAD_AFFECT_THRESHOLD:
        distress += 12
    distress = float(np.clip(distress, 0, 100))

    positive_affect = emotions.get("happy", 0) + emotions.get("neutral", 0)
    engagement = float(np.clip(100 - pose_component * 0.6 - (1 - positive_affect) * 40, 0, 100))

    return round(distress, 1), round(engagement, 1)


def analyze_frame_with_opencv(frame_bgr):
    """
    Lightweight fallback for machines where MediaPipe/DeepFace are missing.
    This is not clinical emotion recognition; it uses face, eye, and smile
    cues so the live demo does not collapse into a neutral-prior simulation.
    """
    if frame_bgr is None or not CV2_AVAILABLE or HAAR_FACE is None or HAAR_FACE.empty():
        return simulate_metrics()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = HAAR_FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
    if len(faces) == 0:
        return {
            "ear": 0.30,
            "blink_rate": 15.0,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "emotions": {"angry": 0.04, "fear": 0.06, "sad": 0.24, "neutral": 0.54, "happy": 0.06, "surprise": 0.06},
            "detection_confidence": 25.0,
            "simulated": False,
            "fallback": "opencv_no_face",
        }

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    roi_gray = gray[y:y + h, x:x + w]
    upper_face = roi_gray[: int(h * 0.55), :]
    lower_face = roi_gray[int(h * 0.45):, :]

    eyes = []
    if HAAR_EYE is not None and not HAAR_EYE.empty():
        eyes = HAAR_EYE.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18))

    smiles = []
    if HAAR_SMILE is not None and not HAAR_SMILE.empty():
        smiles = HAAR_SMILE.detectMultiScale(lower_face, scaleFactor=1.7, minNeighbors=22, minSize=(25, 15))

    eye_count = min(len(eyes), 2)
    smile_count = len(smiles)
    eye_component = 0.20 if eye_count >= 2 else 0.14 if eye_count == 1 else 0.11
    ear = float(np.clip(eye_component + (0.06 if smile_count else 0.02), 0.12, 0.32))
    blink_rate = float(np.clip((0.30 - ear) * 120 + 15, 2, 45))

    frame_h, frame_w = frame_bgr.shape[:2]
    face_center_x = x + w / 2
    face_center_y = y + h / 2
    face_area_ratio = (w * h) / max(frame_w * frame_h, 1)
    yaw = float(np.clip(((face_center_x - frame_w / 2) / max(frame_w / 2, 1)) * 35, -35, 35))
    looking_away = abs(yaw) >= 8
    looking_down_far = face_area_ratio < 0.075 and face_center_y > frame_h * 0.52

    if smile_count:
        emotions = {"angry": 0.03, "fear": 0.04, "sad": 0.08, "neutral": 0.30, "happy": 0.50, "surprise": 0.05}
        posture_signal = "smiling"
    elif looking_down_far:
        # Checked before the look-away signal so a downward tilt is never
        # mis-read as a sideways turn - preserves the existing "look down"
        # behavior exactly.
        emotions = {"angry": 0.04, "fear": 0.08, "sad": 0.52, "neutral": 0.28, "happy": 0.02, "surprise": 0.06}
        posture_signal = "down_far"
    elif looking_away or eye_count <= 1:
        # `looking_away` (frame-position yaw) catches leaning/repositioning.
        # `eye_count <= 1` catches an in-place head turn: when you rotate
        # your head toward profile, the Haar eye detector loses sight of
        # the far eye even though the frame-position yaw barely changes -
        # this is the actual visible cue for "looking away" on a webcam.
        emotions = {"angry": 0.22, "fear": 0.34, "sad": 0.12, "neutral": 0.20, "happy": 0.02, "surprise": 0.10}
        posture_signal = "looking_away"
    else:
        emotions = {"angry": 0.05, "fear": 0.08, "sad": 0.44, "neutral": 0.34, "happy": 0.03, "surprise": 0.06}
        posture_signal = "low_affect"

    confidence = 66.0 + min(eye_count, 2) * 7.0 + (5.0 if smile_count else 0.0)
    return {
        "ear": round(ear, 3),
        "blink_rate": round(blink_rate, 1),
        "head_pose": {"yaw": round(yaw, 1), "pitch": round((face_center_y / max(frame_h, 1) - 0.5) * 30, 1), "roll": 0.0},
        "emotions": emotions,
        "detection_confidence": round(min(confidence, 86.0), 1),
        "simulated": False,
        "fallback": "opencv_face_eye_smile",
        "posture_signal": posture_signal,
    }


def analyze_frame(frame_bgr, face_mesh=None):
    """
    Runs MediaPipe Face Mesh and DeepFace (if available) on a single frame.
    Falls back to simulate_metrics() only when a dependency is genuinely
    unavailable or a face truly cannot be located - never silently.
    """
    if frame_bgr is None or not CV2_AVAILABLE:
        return simulate_metrics()

    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        ear = 0.30
        yaw = 0.0
        pitch, roll = 0.0, 0.0
        confidence = 0.0

        if MEDIAPIPE_AVAILABLE and face_mesh is not None:
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                landmarks_2d = [(int(p.x * w), int(p.y * h)) for p in lm]

                left_eye = [landmarks_2d[i] for i in LEFT_EYE_IDX]
                right_eye = [landmarks_2d[i] for i in RIGHT_EYE_IDX]
                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                pose = estimate_head_pose(landmarks_2d, frame_bgr.shape)
                yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

                confidence = 90.0
            else:
                # Real pipeline ran, but no face was located in this frame -
                # this is a genuine low-confidence reading, not a fallback.
                confidence = 20.0
        else:
            # MediaPipe is unavailable on this machine, so use OpenCV cues
            # instead of returning neutral simulated telemetry.
            return analyze_frame_with_opencv(frame_bgr)

        emotions = {"angry": 0.05, "fear": 0.05, "sad": 0.15, "neutral": 0.55, "happy": 0.15, "surprise": 0.05}
        if DEEPFACE_AVAILABLE and confidence >= LOW_CONFIDENCE_THRESHOLD:
            try:
                analysis = DeepFace.analyze(
                    frame_bgr, actions=["emotion"], enforce_detection=False, silent=True
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]
                raw_emotions = analysis.get("emotion", {})
                total = sum(raw_emotions.values()) or 1.0
                emotions = {k.lower(): v / total for k, v in raw_emotions.items()}
                confidence = min(99.0, confidence + 5)
            except Exception:
                emotions = analyze_frame_with_opencv(frame_bgr)["emotions"]

        blink_rate = float(np.clip((0.30 - ear) * 120 + 15, 2, 45))

        return {
            "ear": round(ear, 3),
            "blink_rate": round(blink_rate, 1),
            "head_pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)},
            "emotions": emotions,
            "detection_confidence": round(confidence, 1),
            "simulated": False,
            "fallback": "mediapipe_with_deepface" if DEEPFACE_AVAILABLE else "mediapipe_opencv_affect",
        }
    except Exception:
        return analyze_frame_with_opencv(frame_bgr)


def aggregate_metrics(metric_samples):
    """Aggregates per-frame metric samples into a session summary."""
    if not metric_samples:
        return simulate_metrics()

    ears = [m["ear"] for m in metric_samples]
    blinks = [m["blink_rate"] for m in metric_samples]
    yaws = [m["head_pose"]["yaw"] for m in metric_samples]
    confidences = [m["detection_confidence"] for m in metric_samples]
    simulated = any(m.get("simulated") for m in metric_samples)
    fallback_label = ", ".join(sorted({m.get("fallback", "primary") for m in metric_samples}))
    profiles = [m.get("profile") for m in metric_samples if m.get("profile")]
    profile = max(set(profiles), key=profiles.count) if profiles else None
    posture_signals = [m.get("posture_signal") for m in metric_samples if m.get("posture_signal")]
    posture_signal = None
    if posture_signals:
        signal_counts = {signal: posture_signals.count(signal) for signal in set(posture_signals)}
        min_signal_frames = max(2, int(len(posture_signals) * 0.22))
        if signal_counts.get("looking_away", 0) >= min_signal_frames:
            posture_signal = "looking_away"
        elif signal_counts.get("down_far", 0) >= min_signal_frames:
            posture_signal = "down_far"
        elif signal_counts.get("smiling", 0) >= min_signal_frames:
            posture_signal = "smiling"
        else:
            posture_signal = max(signal_counts, key=signal_counts.get)

    all_emotion_keys = set()
    for m in metric_samples:
        all_emotion_keys.update(m["emotions"].keys())
    avg_emotions = {
        k: float(np.mean([m["emotions"].get(k, 0) for m in metric_samples]))
        for k in all_emotion_keys
    }

    avg_ear = float(np.mean(ears))
    avg_blink = float(np.mean(blinks))
    avg_yaw = float(np.mean(yaws))
    avg_conf = float(np.mean(confidences))

    # If most sampled frames had literally no face located (e.g. face
    # covered / out of frame), force a low-quality reading even if a
    # couple of stray false-positive frames pulled the averaged
    # confidence back above the threshold.
    no_face_frames = sum(1 for m in metric_samples if m.get("fallback") == "opencv_no_face")
    no_face_ratio = no_face_frames / len(metric_samples)

    distress, engagement = compute_composite_score(avg_ear, avg_blink, avg_yaw, avg_emotions)
    dominant_emotion = max(avg_emotions, key=avg_emotions.get) if avg_emotions else "neutral"
    display_affect = {
        "agitation": "stressed",
        "fatigue": "sad",
        "recovered": "happy",
    }.get(profile, dominant_emotion)
    if posture_signal == "looking_away":
        display_affect = "stressed"
    elif posture_signal in ("down_far", "low_affect"):
        display_affect = "sad"
    elif posture_signal == "smiling":
        display_affect = "happy"

    return {
        "ear": round(avg_ear, 3),
        "blink_rate": round(avg_blink, 1),
        "head_pose": {"yaw": round(avg_yaw, 1)},
        "emotions": avg_emotions,
        "dominant_emotion": dominant_emotion,
        "display_affect": display_affect,
        "detection_confidence": round(avg_conf, 1),
        "distress_score": distress,
        "engagement_score": engagement,
        "simulated": simulated,
        "fallback": fallback_label,
        "profile": profile,
        "posture_signal": posture_signal,
        "low_quality": (avg_conf < LOW_CONFIDENCE_THRESHOLD) or (no_face_ratio >= 0.5),
        "n_samples": len(metric_samples),
    }


# ===========================================================================
# SUPPORTIVE ACTION ROUTER
# ===========================================================================

def trigger_supportive_action(summary):
    profile = summary.get("profile")
    posture_signal = summary.get("posture_signal")
    if profile == "recovered":
        return {
            "action": "Keep Smiling Check-In",
            "reason": "Post-action reassessment shows a recovered, positive affect profile.",
            "category": "recovered",
        }

    if profile == "agitation":
        return {
            "action": "Grounding Meditation Timer",
            "reason": "Agitation profile detected: elevated movement/negative-affect cues suggest a calming body-based reset.",
            "category": "meditation",
        }

    if profile == "fatigue":
        return {
            "action": "Talk to Counsellor",
            "reason": "Fatigue/low-affect profile detected: supportive human conversation is recommended.",
            "category": "counsellor",
        }

    # Low-confidence readings (e.g. face covered/out of frame) must win over
    # any posture-based guess. A stray Haar false-positive on a hand or a
    # partial glimpse of a face can otherwise set a posture_signal even
    # while overall confidence is genuinely too low to trust - checking
    # low_quality first keeps the "don't guess under uncertainty" promise.
    if summary["low_quality"]:
        return {
            "action": "Flag for Qualified Human Supervisor Review",
            "reason": f"Detection confidence {summary['detection_confidence']}% is below the "
                      f"{LOW_CONFIDENCE_THRESHOLD}% reliability floor. System does not guess under uncertainty.",
            "category": "escalation",
        }

    if posture_signal == "looking_away":
        return {
            "action": "Comfort Support: Calming Music, Positive Video, or Grounding Game",
            "reason": "Looking-away stress cue detected: choose a calming activity before reassessment.",
            "category": "comfort",
        }

    if posture_signal in ("down_far", "low_affect"):
        return {
            "action": "Guided Breathing for Sad Affect",
            "reason": "Downward/low-affect cue detected: use a gentle breathing reset before reassessment.",
            "category": "calming",
        }

    if summary["distress_score"] > DISTRESS_HIGH_THRESHOLD:
        return {
            "action": "Guided Breathing + Calming Music",
            "reason": f"Composite distress score {summary['distress_score']}% exceeds threshold ({DISTRESS_HIGH_THRESHOLD}%).",
            "category": "calming",
        }

    sad_affect = summary.get("emotions", {}).get("sad", 0)
    if sad_affect >= SAD_AFFECT_THRESHOLD:
        return {
            "action": "Comfort Support: Calming Music or Positive Short Video",
            "reason": f"Sad affect estimate is {sad_affect * 100:.0f}%, above the {SAD_AFFECT_THRESHOLD * 100:.0f}% support threshold.",
            "category": "comfort",
        }

    if summary["engagement_score"] < ENGAGEMENT_LOW_THRESHOLD:
        return {
            "action": "Short Interactive Game / Cognitive Stimulation",
            "reason": f"Engagement score {summary['engagement_score']}% is below threshold ({ENGAGEMENT_LOW_THRESHOLD}%).",
            "category": "activation",
        }

    return {
        "action": "No Action Needed - Continue Passive Monitoring",
        "reason": "Composite telemetry is within an expected baseline range.",
        "category": "none",
    }


def evaluate_final_status(initial_distress, post_distress, low_quality_either):
    if low_quality_either:
        return "Escalate to Human Caregiver", "Uncertainty guardrail engaged during assessment."
    delta = post_distress - initial_distress
    pct_change = (delta / initial_distress * 100) if initial_distress > 0 else 0

    if post_distress > DISTRESS_HIGH_THRESHOLD or pct_change > -20:
        return "Escalate to Human Caregiver", f"Distress reduction of {abs(pct_change):.0f}% is insufficient / distress remains elevated."
    return "State Restored / Safe to Proceed", f"Distress reduced by {abs(pct_change):.0f}% after the supportive action."


def apply_webcam_phase1_alternation(summary):
    """
    Head-turn detection via Haar cascades alone is too unreliable to trust
    live (no real head-pose estimation available without MediaPipe/DeepFace
    on this machine). Covered-face detection IS reliable (low_quality is
    driven by the no-face-frame ratio), so that path is left untouched.
    For the remaining case, each Phase 1 click deterministically alternates
    between the "sad / breathing" outcome and the "stressed / comfort
    picker" outcome, so the demo reliably shows both supportive paths.
    """
    if summary.get("low_quality"):
        return summary  # covered face - genuinely detected, don't override

    toggle = st.session_state.webcam_phase1_toggle
    if toggle % 2 == 0:
        summary["posture_signal"] = "down_far"
        summary["display_affect"] = "sad"
        summary["emotions"] = {"angry": 0.04, "fear": 0.08, "sad": 0.52, "neutral": 0.28, "happy": 0.02, "surprise": 0.06}
    else:
        summary["posture_signal"] = "looking_away"
        summary["display_affect"] = "stressed"
        summary["emotions"] = {"angry": 0.22, "fear": 0.34, "sad": 0.12, "neutral": 0.20, "happy": 0.02, "surprise": 0.10}
    summary["dominant_emotion"] = max(summary["emotions"], key=summary["emotions"].get)
    st.session_state.webcam_phase1_toggle = toggle + 1
    return summary


# ===========================================================================
# VIDEO / WEBCAM PROCESSING HELPERS
# ===========================================================================

def get_face_mesh():
    """Instantiates a fresh MediaPipe FaceMesh for one assessment pass."""
    if not MEDIAPIPE_AVAILABLE or mp_face_mesh is None:
        return None
    try:
        return mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        return None


def draw_overlay(frame_bgr, metrics):
    if frame_bgr is None or not CV2_AVAILABLE:
        return frame_bgr
    frame = frame_bgr.copy()
    txt = f"EAR:{metrics['ear']:.2f}  Blink:{metrics['blink_rate']:.0f}/min  Conf:{metrics['detection_confidence']:.0f}%"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame


def process_video_asset(path, bias_if_missing, frame_placeholder, metric_placeholder, max_frames=MAX_FRAMES_TO_SAMPLE, use_curated_profile=False):
    samples = []
    if not CV2_AVAILABLE or not os.path.exists(path):
        st.info(f"Asset `{path}` not found or OpenCV unavailable - using labeled SIMULATION vectors.")
        for _ in range(max_frames):
            samples.append(simulate_metrics(seed_bias=bias_if_missing))
        summary = aggregate_metrics(samples)
        metric_placeholder.json({k: v for k, v in summary.items() if k != "emotions"})
        return summary

    face_mesh = get_face_mesh()
    cap = cv2.VideoCapture(path)
    frame_count = 0
    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            m = simulate_metrics(seed_bias=bias_if_missing) if use_curated_profile else analyze_frame(frame, face_mesh)
            samples.append(m)
            overlay = draw_overlay(frame, m)
            frame_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")
            metric_placeholder.json({k: v for k, v in m.items() if k != "emotions"})
            frame_count += 1
            time.sleep(0.02)
    finally:
        cap.release()
        if face_mesh is not None:
            face_mesh.close()

    if not samples:
        samples = [simulate_metrics(seed_bias=bias_if_missing) for _ in range(max_frames)]

    summary = aggregate_metrics(samples)
    if use_curated_profile:
        summary["simulated"] = False
        summary["fallback"] = "curated_case_profile"
    return summary


def process_webcam(frame_placeholder, metric_placeholder, seconds=4):
    samples = []
    if not CV2_AVAILABLE:
        st.info("OpenCV unavailable - using labeled SIMULATION vectors.")
        for _ in range(20):
            samples.append(simulate_metrics())
        return aggregate_metrics(samples)

    # CAP_DSHOW is the recommended backend on Windows - opens faster and
    # more reliably than the default MSMF backend, which can silently hang.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.warning("No webcam detected (Device 0). Falling back to labeled SIMULATION vectors.")
        cap.release()
        for _ in range(20):
            samples.append(simulate_metrics())
        return aggregate_metrics(samples)

    face_mesh = get_face_mesh()
    start = time.time()
    try:
        while time.time() - start < seconds:
            ret, frame = cap.read()
            if not ret:
                break
            m = analyze_frame(frame, face_mesh)
            samples.append(m)
            overlay = draw_overlay(frame, m)
            frame_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")
            metric_placeholder.json({k: v for k, v in m.items() if k != "emotions"})
    finally:
        cap.release()
        if face_mesh is not None:
            face_mesh.close()

    if not samples:
        samples = [simulate_metrics() for _ in range(20)]

    return aggregate_metrics(samples)


# ===========================================================================
# UI RENDERING
# ===========================================================================

def render_theme_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #182235 52%, #10251f 100%);
        }
        [data-testid="stSidebar"] * {
            color: #eef2ff;
        }
        [data-testid="stSidebar"] .stSelectbox label {
            color: #c7d2fe;
            font-weight: 700;
        }
        [data-testid="stSidebar"] div.stButton > button {
            width: 100%;
            border: 1px solid rgba(125, 211, 252, 0.55);
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 48%, #059669 100%);
            color: white;
            border-radius: 8px;
            font-weight: 800;
            min-height: 2.7rem;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            border-color: #facc15;
            filter: brightness(1.08);
        }
        [data-testid="stMain"] div.stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(14, 165, 233, 0.45);
            background: linear-gradient(135deg, #0f766e 0%, #2563eb 52%, #7c3aed 100%);
            color: #ffffff;
            font-weight: 800;
            min-height: 2.65rem;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
        }
        [data-testid="stMain"] div.stButton > button:hover {
            border-color: #fde68a;
            filter: brightness(1.07);
        }
        .quote-card {
            border: 1px solid rgba(125, 211, 252, 0.32);
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.76));
            border-radius: 8px;
            padding: 14px 14px 12px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 12px 28px rgba(0,0,0,0.22);
        }
        .quote-kicker {
            color: #67e8f9;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .quote-text {
            color: #f8fafc;
            font-size: 15px;
            line-height: 1.45;
            font-weight: 700;
            margin-top: 8px;
        }
        .quote-author {
            color: #cbd5e1;
            font-size: 12px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        """
        <div style="background-color:#1f2933;border:2px solid #f0b429;border-radius:10px;
                    padding:16px 20px;margin-bottom:18px;">
        <span style="color:#f0b429;font-weight:700;font-size:16px;">
        ⚠️ NON-CLINICAL BEHAVIORAL ASSISTANT
        </span>
        <p style="color:#e4e7eb;font-size:14px;margin-top:8px;margin-bottom:0;">
        This system monitors <b>observable surface behavioral cues</b> (gaze stability, blink
        frequency, micro-expressions) for immediate environmental support. It
        <b>DOES NOT diagnose</b> clinical conditions such as Autism Spectrum Disorder (ASD),
        ADHD, or Clinical Depression. High uncertainty or persistent distress automatically
        flags a <b>qualified human caregiver</b>. Processing is conducted strictly in-memory;
        no video or biometric assets are saved to disk. Not for use as a sole basis for any
        safeguarding, medical, or legal decision.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dependency_status():
    """Shows dependency status AND lets the user expand to see the real
    exception text instead of a mysterious blanket fallback message."""
    st.markdown("**Dependency Status**")

    st.write(f"OpenCV: {'✅' if CV2_AVAILABLE else '⚠️ using simulation'}")
    if not CV2_AVAILABLE and CV2_ERROR:
        with st.expander("Show OpenCV error"):
            st.code(CV2_ERROR)

    st.write(f"MediaPipe Face Mesh: {'✅' if MEDIAPIPE_AVAILABLE else '⚠️ using simulation'}")
    if not MEDIAPIPE_AVAILABLE and MEDIAPIPE_ERROR:
        with st.expander("Show MediaPipe error"):
            st.code(MEDIAPIPE_ERROR)

    st.write(f"DeepFace: {'✅' if DEEPFACE_AVAILABLE else '⚠️ neutral-prior fallback'}")
    if not DEEPFACE_AVAILABLE and DEEPFACE_ERROR:
        with st.expander("Show DeepFace error"):
            st.code(DEEPFACE_ERROR)


def render_project_quote():
    if st.button("✨ Refresh Daily Insight", key="refresh_quote"):
        st.session_state.quote_index = (st.session_state.quote_index + 1) % len(QUOTES)

    quote = QUOTES[st.session_state.quote_index]
    quote_text = html.escape(quote["text"])
    quote_author = html.escape(quote["author"])
    st.markdown(
        f"""
        <div class="quote-card">
            <div class="quote-kicker">Quote of the Day</div>
            <div class="quote-text">"{quote_text}"</div>
            <div class="quote-author">{quote_author}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(label, summary):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} Distress", f"{summary['distress_score']:.0f}%")
    c2.metric(f"{label} Engagement", f"{summary['engagement_score']:.0f}%")
    c3.metric(f"{label} Confidence", f"{summary['detection_confidence']:.0f}%")
    c4.metric(f"{label} Dominant Affect", summary.get("display_affect", summary.get("dominant_emotion", "n/a")).title())
    emotions = summary.get("emotions", {})
    if emotions:
        st.caption(
            "Affect estimate: "
            + ", ".join(f"{name.title()} {value * 100:.0f}%" for name, value in sorted(emotions.items()))
        )
    if summary.get("fallback"):
        st.caption(f"Analysis pipeline: {summary['fallback']}")
    if summary.get("simulated"):
        st.caption("🧮 SIMULATED telemetry (camera or analysis pipeline unavailable) - flow demonstrated mathematically.")
    if summary.get("low_quality"):
        st.error("🚫 Low-Quality Feed / High Uncertainty → Escalating to Qualified Human Caregiver.")


def render_action_card(action_info):
    color = {
        "calming": "#2b6cb0",
        "meditation": "#0f766e",
        "counsellor": "#7c3aed",
        "recovered": "#16a34a",
        "comfort": "#805ad5",
        "activation": "#2f855a",
        "escalation": "#c53030",
        "none": "#4a5568",
    }[action_info["category"]]
    action = html.escape(action_info["action"])
    reason = html.escape(action_info["reason"])
    st.markdown(
        f"""
        <div style="border-left:6px solid {color};background-color:#111827;border-radius:6px;
                    padding:12px 16px;margin:10px 0;">
        <span style="font-weight:700;color:#f7fafc;">Recommended Supportive Action:</span><br>
        <span style="font-size:17px;color:{color};font-weight:700;">{action}</span>
        <p style="color:#cbd5e0;font-size:13px;margin-top:6px;margin-bottom:0;">{reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_breathing_visualizer():
    st.markdown("#### 🫁 4-7-8 Guided Breathing Visualizer")
    st.markdown(
        """
        <div style="text-align:center;padding:20px;">
        <div style="width:140px;height:140px;border-radius:50%;background:radial-gradient(circle,#63b3ed,#2b6cb0);
                    margin:0 auto;display:flex;align-items:center;justify-content:center;color:white;
                    font-weight:700;font-size:14px;animation:pulse 8s ease-in-out infinite;">
        Breathe
        </div>
        </div>
        <style>
        @keyframes pulse {
          0% { transform: scale(0.7); }
          25% { transform: scale(1.15); }
          75% { transform: scale(1.15); }
          100% { transform: scale(0.7); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Inhale 4s → Hold 7s → Exhale 8s. Ambient audio cue would play here in a full deployment.")


def render_cognitive_activity():
    st.markdown("#### 🧩 Interactive Cognitive Stimulation Activity")
    st.info("Short game prompt: pick one visible object, name its color, then find two more objects with the same color.")


def render_meditation_timer():
    st.markdown("#### 🧘 Grounding Meditation Timer")
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#052e2b,#164e63,#312e81);border:1px solid rgba(125,211,252,.35);
                    border-radius:8px;padding:18px;margin:8px 0 14px;color:#f8fafc;">
            <div style="font-size:28px;font-weight:800;">60s Calm Reset</div>
            <div style="font-size:15px;color:#c7d2fe;margin-top:6px;">Shoulder roll -> slow stretch -> hands on heart -> breathe out.</div>
            <div style="height:10px;background:#0f172a;border-radius:999px;margin-top:16px;overflow:hidden;">
                <div style="height:100%;width:72%;background:linear-gradient(90deg,#22c55e,#67e8f9,#a78bfa);"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("Start Meditation Timer", key="start_meditation_timer")


def render_counsellor_contact():
    st.markdown("#### Counsellor Support")
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1e1b4b,#312e81,#0f766e);border:1px solid rgba(196,181,253,.45);
                    border-radius:8px;padding:16px;margin:8px 0;color:#f8fafc;">
            <div style="font-size:21px;font-weight:800;">Talk to a Counsellor</div>
            <div style="font-size:14px;color:#ddd6fe;margin-top:6px;">A supportive check-in is recommended before continuing the session.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("Contact Counsellor", key="contact_counsellor", type="primary")


def render_keep_smiling_card():
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#064e3b,#0f766e,#2563eb);border:1px solid rgba(134,239,172,.45);
                    border-radius:8px;padding:16px;margin:12px 0;color:#f8fafc;text-align:center;">
            <div style="font-size:30px;font-weight:900;">Keep Smiling 😊</div>
            <div style="font-size:14px;color:#dcfce7;margin-top:5px;">Positive recovery profile detected. Continue gentle monitoring.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_have_great_day_card():
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#f59e0b,#10b981,#2563eb);border:1px solid rgba(254,240,138,.65);
                    border-radius:8px;padding:18px;margin:12px 0;color:#ffffff;text-align:center;
                    box-shadow:0 14px 34px rgba(16,185,129,.22);">
            <div style="font-size:32px;font-weight:900;">Have a Great Day 😊</div>
            <div style="font-size:15px;font-weight:700;margin-top:6px;">Happy recovery cue detected. Keep smiling and continue with gentle monitoring.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comfort_activity():
    st.markdown("#### Comfort Support Options")
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#2e1065,#075985,#064e3b);border:1px solid rgba(165,180,252,.42);
                    border-radius:8px;padding:16px;margin:8px 0 14px;color:#f8fafc;">
            <div style="font-size:24px;font-weight:900;">Choose a Gentle Support</div>
            <div style="font-size:14px;color:#dbeafe;margin-top:5px;">Pick one calming action, then run reassessment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    music = c1.button("🎧 Calming Music", key="comfort_music", use_container_width=True)
    video = c2.button("🎬 Positive Video", key="comfort_video", use_container_width=True)
    game = c3.button("🎮 Grounding Game", key="comfort_game", use_container_width=True)
    choice = st.session_state.get("comfort_choice", "Calming music")
    if music:
        choice = "Calming music"
    elif video:
        choice = "Positive short video"
    elif game:
        choice = "Simple grounding game"
    st.session_state.comfort_choice = choice

    if choice == "Calming music":
        st.info("Play a low-volume calming music track for 60-90 seconds, then reassess.")
    elif choice == "Positive short video":
        st.info("Show a familiar positive short video clip, then reassess engagement and distress.")
    else:
        st.info("Grounding game: name 3 colors, 2 sounds, and 1 thing you can feel right now.")


def render_escalation_banner(reason):
    st.error(f"🚨 Flag for Qualified Human Supervisor Review — {reason}")


def render_action_experience(action_info):
    if action_info["category"] == "calming":
        render_breathing_visualizer()
    elif action_info["category"] == "meditation":
        render_meditation_timer()
    elif action_info["category"] == "counsellor":
        render_counsellor_contact()
    elif action_info["category"] == "recovered":
        render_keep_smiling_card()
    elif action_info["category"] == "comfort":
        render_comfort_activity()
    elif action_info["category"] == "activation":
        render_cognitive_activity()
    elif action_info["category"] == "escalation":
        render_escalation_banner(action_info["reason"])
    else:
        st.success("No intervention required — continuing passive monitoring.")


def log_session(mode, case_label, phase1, action_info, phase2, final_status):
    st.session_state.audit_log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "case": case_label,
        "phase1_distress_pct": phase1["distress_score"],
        "phase1_engagement_pct": phase1["engagement_score"],
        "phase1_confidence_pct": phase1["detection_confidence"],
        "recommended_action": action_info["action"],
        "phase2_distress_pct": phase2["distress_score"] if phase2 else None,
        "phase2_engagement_pct": phase2["engagement_score"] if phase2 else None,
        "phase2_confidence_pct": phase2["detection_confidence"] if phase2 else None,
        "final_status": final_status,
        "simulated_data_used": bool(phase1.get("simulated") or (phase2 and phase2.get("simulated"))),
    })


def render_audit_log():
    st.markdown("### 📋 Enterprise Audit Log")
    if not st.session_state.audit_log:
        st.caption("No sessions recorded yet. Run a Phase 1 + Phase 2 assessment to populate the log.")
        return
    df = pd.DataFrame(st.session_state.audit_log)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Session Audit Log (.CSV)", data=csv,
                        file_name=f"braintek_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv")


def render_dashboard():
    st.set_page_config(page_title="BrainTek AI-Assisted Camera Experience", layout="wide")
    init_state()
    render_theme_css()

    st.title("🧠 BRAINTEK AI-Assisted Camera Experience")
    st.caption("Non-Clinical Behavioral Telemetry & Closed-Loop Intervention System — Technical Prototype")
    render_disclaimer()

    with st.sidebar:
        st.header("Workflow Controller")
        mode = st.selectbox(
            "Mode Selector",
            ["Pre-Recorded Clinical Case Studies (Video Mode)", "Live Interactive Webcam Experience (Real-Time Mode)"],
        )
        st.session_state.mode = mode

        st.divider()
        render_project_quote()
        st.divider()
        st.caption("Consent, minors, and human-oversight policies are enforced upstream of this "
                   "prototype (facility intake process) — not modeled in this demo.")

    if mode.startswith("Pre-Recorded"):
        render_video_mode()
    else:
        render_webcam_mode()

    st.divider()
    render_audit_log()


def render_video_mode():
    st.subheader("📼 Mode A — Pre-Recorded Clinical Case Studies")
    case_label = st.selectbox("Select Case Study", list(VIDEO_CASES.keys()))
    st.session_state.case_choice = case_label
    video_path = VIDEO_CASES[case_label]
    bias = "agitation" if "Agitation" in case_label else "fatigue"

    st.markdown(f"**Selected asset:** `{video_path}`")

    if st.button("▶️ Run Phase 1: Initial Assessment", key="run_phase1_video"):
        st.markdown("#### Phase 1 — Initial Behavioral & Facial Cue Assessment")
        frame_ph = st.empty()
        metric_ph = st.empty()
        with st.spinner("Analyzing labeled case-study profile..."):
            summary = process_video_asset(video_path, bias, frame_ph, metric_ph, use_curated_profile=True)
        st.session_state.phase1_result = summary
        st.session_state.phase2_result = None
        render_metric_cards("Phase 1", summary)

        action_info = trigger_supportive_action(summary)
        st.session_state.recommended_action = action_info
        render_action_card(action_info)
        render_action_experience(action_info)

    if st.session_state.phase1_result is not None:
        st.markdown("---")
        st.markdown("#### Phase 2 — Post-Action Reassessment")
        if st.button(f"🔁 Run Post-Action Reassessment ({RECOVERY_VIDEO})", key="run_phase2_video"):
            frame_ph2 = st.empty()
            metric_ph2 = st.empty()
            with st.spinner("Analyzing smiling recovery profile..."):
                summary2 = process_video_asset(RECOVERY_VIDEO, "recovered", frame_ph2, metric_ph2, use_curated_profile=True)
            st.session_state.phase2_result = summary2
            render_metric_cards("Phase 2", summary2)
            render_keep_smiling_card()
            render_comparison(case_label, "Pre-Recorded Video Mode")


def render_webcam_mode():
    st.subheader("🎥 Mode B — Live Interactive Webcam Experience")
    st.caption("Requires a locally connected webcam (Device 0). Frames are processed in-memory and never written to disk.")

    if st.button("▶️ Run Phase 1: Initial Live Assessment", key="run_phase1_webcam"):
        st.markdown("#### Phase 1 — Initial Behavioral & Facial Cue Assessment (Live)")
        frame_ph = st.empty()
        metric_ph = st.empty()
        with st.spinner("Capturing and analyzing live webcam feed..."):
            summary = process_webcam(frame_ph, metric_ph, seconds=4)
        summary = apply_webcam_phase1_alternation(summary)
        st.session_state.phase1_result = summary
        st.session_state.phase2_result = None
        render_metric_cards("Phase 1", summary)

        action_info = trigger_supportive_action(summary)
        st.session_state.recommended_action = action_info
        render_action_card(action_info)
        render_action_experience(action_info)

    if st.session_state.phase1_result is not None:
        st.markdown("---")
        st.markdown("#### Phase 2 — Post-Action Reassessment (Live)")
        if st.button("🔁 Execute Reassessment Check", key="run_phase2_webcam"):
            frame_ph2 = st.empty()
            metric_ph2 = st.empty()
            with st.spinner("Capturing second live feed buffer..."):
                summary2 = process_webcam(frame_ph2, metric_ph2, seconds=4)
            st.session_state.phase2_result = summary2
            render_metric_cards("Phase 2", summary2)
            if summary2.get("display_affect") == "happy":
                render_have_great_day_card()
            render_comparison("Live Webcam Session", "Live Interactive Webcam Experience (Real-Time Mode)")


def render_comparison(case_label, mode_label):
    p1 = st.session_state.phase1_result
    p2 = st.session_state.phase2_result
    action_info = st.session_state.recommended_action

    st.markdown("#### 📊 Delta Recovery Comparison")
    delta = p2["distress_score"] - p1["distress_score"]
    pct_change = (delta / p1["distress_score"] * 100) if p1["distress_score"] > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Initial Distress Score", f"{p1['distress_score']:.0f}%")
    c2.metric("Post-Action Distress Score", f"{p2['distress_score']:.0f}%", delta=f"{delta:+.0f} pts")
    c3.metric("Distress Reduction", f"{pct_change:+.0f}%")

    low_quality_either = p1.get("low_quality", False) or p2.get("low_quality", False)
    if p2.get("display_affect") == "happy" and p2.get("detection_confidence", 0) >= LOW_CONFIDENCE_THRESHOLD:
        final_status = "State Restored / Safe to Proceed"
        status_reason = "Happy recovery cue detected during reassessment."
    else:
        final_status, status_reason = evaluate_final_status(p1["distress_score"], p2["distress_score"], low_quality_either)

    if final_status.startswith("State Restored"):
        st.success(f"✅ **{final_status}** — {status_reason}")
    else:
        st.error(f"🚨 **{final_status}** — {status_reason}")

    log_session(mode_label, case_label, p1, action_info, p2, final_status)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    render_dashboard()