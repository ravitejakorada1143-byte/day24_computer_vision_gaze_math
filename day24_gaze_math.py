
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------
# 1. MediaPipe Model Configuration
# ----------------------------------

MODEL_PATH = "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# ----------------------------------
# 2. Gaze Mathematics
# ----------------------------------

def calculate_gaze(
    iris_x,
    iris_y,
    left_x,
    right_x,
    top_y,
    bottom_y
):

    eye_width = abs(right_x - left_x)
    eye_height = abs(bottom_y - top_y)

    if eye_width == 0 or eye_height == 0:
        return None, None

    gaze_x = (
        iris_x - min(left_x, right_x)
    ) / eye_width

    gaze_y = (
        iris_y - min(top_y, bottom_y)
    ) / eye_height

    return gaze_x, gaze_y


# ----------------------------------
# 3. Calculate Iris Centre
# ----------------------------------

def get_iris_center(landmarks, indices):

    x = sum(
        landmarks[i].x for i in indices
    ) / len(indices)

    y = sum(
        landmarks[i].y for i in indices
    ) / len(indices)

    return x, y


# ----------------------------------
# 4. Webcam Initialization
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Webcam could not be opened.")

    detector.close()

    raise SystemExit


print("Day 24 - Computer Vision Gaze Math")
print("Real-Time Gaze Tracking Started")


# ----------------------------------
# 5. Anomaly Tracking Variables
# ----------------------------------

off_center_start = None

anomaly_logged = False

previous_timestamp_ms = 0


# ----------------------------------
# 6. Real-Time Processing
# ----------------------------------

try:

    while True:

        success, frame = cap.read()

        if not success:

            print("Unable to capture webcam frame.")

            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Ensure strictly increasing timestamps
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            previous_timestamp_ms + 1
        )

        previous_timestamp_ms = timestamp_ms

        # Detect facial landmarks
        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:

            face = result.face_landmarks[0]

            # Left and right iris landmark groups
            iris_groups = [
                ([468, 469, 470, 471, 472],
                 33, 133, 159, 145),

                ([473, 474, 475, 476, 477],
                 362, 263, 386, 374)
            ]

            gaze_values = []

            # ----------------------------------
            # 7. Calculate Both Eye Positions
            # ----------------------------------

            for iris_ids, corner1, corner2, upper, lower in iris_groups:

                iris_x, iris_y = get_iris_center(
                    face,
                    iris_ids
                )

                gaze_x, gaze_y = calculate_gaze(
                    iris_x,
                    iris_y,
                    face[corner1].x,
                    face[corner2].x,
                    face[upper].y,
                    face[lower].y
                )

                if gaze_x is None:
                    continue

                gaze_values.append(
                    (gaze_x, gaze_y)
                )

                # Convert normalized coordinates to pixels
                pixel_x = int(iris_x * width)
                pixel_y = int(iris_y * height)

                # Draw iris centre
                cv2.circle(
                    frame,
                    (pixel_x, pixel_y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # ----------------------------------
            # 8. Average Gaze Calculation
            # ----------------------------------

            if gaze_values:

                avg_x = sum(
                    value[0] for value in gaze_values
                ) / len(gaze_values)

                avg_y = sum(
                    value[1] for value in gaze_values
                ) / len(gaze_values)

                # ----------------------------------
                # 9. Off-Centre Gaze Detection
                # ----------------------------------

                off_center = (
                    avg_x < 0.35 or
                    avg_x > 0.65 or
                    avg_y < 0.30 or
                    avg_y > 0.70
                )

                cv2.putText(
                    frame,
                    f"Gaze X: {avg_x:.2f} Y: {avg_y:.2f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

                # ----------------------------------
                # 10. Two-Second Anomaly Detection
                # ----------------------------------

                if off_center:

                    if off_center_start is None:

                        off_center_start = time.monotonic()

                    duration = (
                        time.monotonic() - off_center_start
                    )

                    cv2.putText(
                        frame,
                        f"Off-center: {duration:.1f}s",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2
                    )

                    if duration > 2.0 and not anomaly_logged:

                        print(
                            "GAZE ANOMALY: Off-center gaze detected "
                            f"for {duration:.2f} seconds"
                        )

                        anomaly_logged = True

                else:

                    off_center_start = None

                    anomaly_logged = False

                    cv2.putText(
                        frame,
                        "Gaze: CENTERED",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

        else:

            off_center_start = None

            anomaly_logged = False

            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Display webcam preview
        cv2.imshow(
            "Day 24 - Gaze Tracking",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break


# ----------------------------------
# 11. Release Resources
# ----------------------------------

finally:

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    print("Gaze tracking stopped successfully.")
