"""
Hand Tracking Minimum
=====================
Minimal hand tracking implementation using MediaPipe.
Part of the AI Virtual Painter project.
"""

import cv2
import mediapipe as mp
import time


def main():
    """Minimal hand tracking demo."""
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    prev_time = 0
    curr_time = 0

    print("Starting Hand Tracking Minimum...")
    print("Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Highlight specific landmarks
                    if id == 0:  # Wrist
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
        )

        cv2.imshow("Hand Tracking Min", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
