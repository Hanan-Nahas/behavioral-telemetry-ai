"""
check_env.py — prints the REAL exception text for mediapipe/deepface
import failures instead of the app's silent True/False fallback flags.

Run:  python check_env.py
(from inside your activated venv, same folder as app.py)
"""
import sys
import traceback

print("=" * 70)
print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version}")
print("=" * 70)

def try_import(label, import_fn):
    print(f"\n--- {label} ---")
    try:
        import_fn()
        print(f"[OK] {label} imported successfully.")
    except Exception:
        print(f"[FAILED] {label} raised an exception:\n")
        traceback.print_exc()

def _cv2():
    import cv2
    print(f"    cv2 version: {cv2.__version__}")

def _mediapipe():
    import mediapipe as mp
    print(f"    mediapipe version: {getattr(mp, '__version__', 'unknown')}")
    print(f"    mediapipe file: {mp.__file__}")
    print(f"    hasattr(mp, 'solutions'): {hasattr(mp, 'solutions')}")
    if hasattr(mp, "solutions"):
        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        print("    FaceMesh() constructed OK")
        fm.close()
    else:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        fm = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        print("    FaceMesh() constructed OK via fallback path")
        fm.close()

def _deepface():
    from deepface import DeepFace
    import numpy as np
    dummy = (np.random.rand(100, 100, 3) * 255).astype("uint8")
    result = DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
    print(f"    DeepFace.analyze() ran OK, result type: {type(result)}")

try_import("opencv-python (cv2)", _cv2)
try_import("mediapipe", _mediapipe)
try_import("deepface", _deepface)

print("\n" + "=" * 70)
print("Done. Whatever [FAILED] traceback printed above is the real cause")
print("your Streamlit app's Dependency Status panel is hiding from you.")
print("=" * 70)