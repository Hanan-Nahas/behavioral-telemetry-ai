import cv2
import time


def test_webcam_indices():
    """Tests indices 0, 1, and -1 using the DirectShow backend on Windows."""
    indices = [0, 1, -1]

    for index in indices:
        print(f"\n--- Testing Camera Index: {index} ---")
        # cv2.CAP_DSHOW prevents silent hangs on Windows MSMF drivers
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"❌ Camera Index {index} could NOT be opened.")
            continue

        print(f"✅ Camera Index {index} opened successfully! Starting preview...")
        print("Press 'q' inside the video window to stop testing this camera.")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"⚠️ Index {index} opened, but failed to read frames (black screen or blocked).")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Display frame count and FPS overlay on live feed
            cv2.putText(
                frame,
                f"Index: {index} | Frames: {frame_count} | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(f"Camera Test - Index {index}", frame)

            # Press 'q' on keyboard to exit window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Closed Camera Index {index}.")


if __name__ == "__main__":
    test_webcam_indices()
