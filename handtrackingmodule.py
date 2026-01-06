"""
Hand Tracking Module
====================
Reusable hand tracking module for computer vision projects.
Part of the AI Virtual Painter project.
"""

import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    """
    A comprehensive hand detector class using MediaPipe.
    Provides hand detection, landmark tracking, and gesture recognition utilities.
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        Initialize the HandDetector.

        Args:
            mode: Static image mode (False for video stream)
            max_hands: Maximum number of hands to detect
            detection_con: Minimum detection confidence threshold
            track_con: Minimum tracking confidence threshold
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
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Finger tip IDs: thumb, index, middle, ring, pinky
        self.tip_ids = [4, 8, 12, 16, 20]

        # Store results
        self.results = None
        self.lm_list = []
        self.bbox = None

    def find_hands(self, img, draw=True, flip_type=True):
        """
        Find hands in the image.

        Args:
            img: Input image (BGR format)
            draw: Whether to draw landmarks on the image
            flip_type: Whether to correct for flipped image

        Returns:
            all_hands: List of detected hands with their data
            img: Image with or without drawn landmarks
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(
                self.results.multi_handedness,
                self.results.multi_hand_landmarks
            ):
                my_hand = {}
                lm_list = []
                x_list = []
                y_list = []

                for id, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), lm.z
                    lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                # Bounding box
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                box_w, box_h = x_max - x_min, y_max - y_min
                bbox = x_min, y_min, box_w, box_h
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                my_hand["lmList"] = lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                # Determine hand type (left/right)
                if flip_type:
                    if hand_type.classification[0].label == "Right":
                        my_hand["type"] = "Left"
                    else:
                        my_hand["type"] = "Right"
                else:
                    my_hand["type"] = hand_type.classification[0].label

                all_hands.append(my_hand)

                # Draw
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )
                    cv2.rectangle(
                        img, (bbox[0] - 20, bbox[1] - 20),
                        (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                        (255, 0, 255), 2
                    )
                    cv2.putText(
                        img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
                    )

        return all_hands, img

    def fingers_up(self, my_hand):
        """
        Check which fingers are up/extended.

        Args:
            my_hand: Hand data dictionary from find_hands()

        Returns:
            List of 5 values (0 or 1) indicating if each finger is up
        """
        fingers = []
        lm_list = my_hand["lmList"]
        hand_type = my_hand["type"]

        # Thumb
        if hand_type == "Right":
            if lm_list[self.tip_ids[0]][0] > lm_list[self.tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lm_list[self.tip_ids[0]][0] < lm_list[self.tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][1] < lm_list[self.tip_ids[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img=None, draw=True, color=(255, 0, 255), scale=5):
        """
        Calculate and optionally visualize the distance between two landmarks.

        Args:
            p1: First point (x, y) or landmark index
            p2: Second point (x, y) or landmark index
            img: Image to draw on
            draw: Whether to draw the visualization
            color: Line and circle color
            scale: Size scale for drawing

        Returns:
            length: Distance between the two points
            info: Tuple of (x1, y1, x2, y2, cx, cy)
            img: Image with visualization
        """
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None and draw:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img


def main():
    """Demo function to test the hand tracking module."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(detection_con=0.8, max_hands=2)
    prev_time = 0

    print("Starting Hand Tracking Module Demo...")
    print("Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        img = cv2.flip(img, 1)
        hands, img = detector.find_hands(img)

        if hands:
            # First hand
            hand1 = hands[0]
            lm_list1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            center1 = hand1["center"]
            hand_type1 = hand1["type"]

            fingers1 = detector.fingers_up(hand1)
            finger_count = sum(fingers1)

            # Display finger count
            cv2.putText(
                img, f"Fingers: {finger_count}", (10, 140),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
            )

            # If two hands detected
            if len(hands) == 2:
                hand2 = hands[1]
                lm_list2 = hand2["lmList"]

                # Calculate distance between index fingers
                length, info, img = detector.find_distance(
                    lm_list1[8][:2], lm_list2[8][:2], img
                )
                cv2.putText(
                    img, f"Distance: {int(length)}", (10, 180),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
                )

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
        )

        cv2.imshow("Hand Tracking Module", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
