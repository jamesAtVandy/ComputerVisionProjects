"""
Finger Counting Project
========================
OpenCV-based finger counting using hand tracking.
Part of the AI Virtual Painter project.
"""

import cv2
import mediapipe as mp
import time


class FingerCounter:
    """
    A class to detect and count fingers using MediaPipe hand tracking.
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        Initialize the FingerCounter.

        Args:
            mode: Static image mode (False for video)
            max_hands: Maximum number of hands to detect
            detection_con: Minimum detection confidence
            track_con: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Finger tip landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """
        Find hands in the image.

        Args:
            img: Input image (BGR format)
            draw: Whether to draw landmarks on the image

        Returns:
            Image with or without drawn landmarks
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )

        return img

    def find_position(self, img, hand_no=0, draw=True):
        """
        Find the position of hand landmarks.

        Args:
            img: Input image
            hand_no: Hand index to track
            draw: Whether to draw circles on landmarks

        Returns:
            List of landmark positions [(id, x, y), ...]
        """
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lm_list

    def count_fingers(self):
        """
        Count the number of extended fingers.

        Returns:
            Number of extended fingers (0-5)
        """
        fingers = []

        if len(self.lm_list) == 0:
            return 0

        # Thumb - check x-axis
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers - check y-axis
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)


def main():
    """Main function to run finger counting demo."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    counter = FingerCounter()
    prev_time = 0

    print("Starting Finger Counting Project...")
    print("Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        img = cv2.flip(img, 1)  # Mirror the image
        img = counter.find_hands(img)
        lm_list = counter.find_position(img, draw=False)

        if len(lm_list) != 0:
            finger_count = counter.count_fingers()

            # Display finger count
            cv2.rectangle(img, (20, 20), (170, 120), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img, str(finger_count), (45, 100),
                cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 5
            )

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 160),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
        )

        cv2.imshow("Finger Counter", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
