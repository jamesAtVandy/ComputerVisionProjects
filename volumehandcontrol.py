"""
Volume Hand Control
===================
Control system volume using hand gestures with OpenCV and MediaPipe.
Part of the AI Virtual Painter project.
"""

import cv2
import numpy as np
import time
import math

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False
    print("Warning: pycaw not available. Volume control will be simulated.")

from handtrackingmodule import HandDetector


class VolumeController:
    """
    Control system volume using hand gestures.
    Uses distance between thumb and index finger to adjust volume.
    """

    def __init__(self, width=1280, height=720):
        """
        Initialize the volume controller.

        Args:
            width: Camera frame width
            height: Camera frame height
        """
        self.width = width
        self.height = height

        # Hand detector
        self.detector = HandDetector(detection_con=0.7, max_hands=1)

        # Volume control setup
        self.min_volume = 0
        self.max_volume = 100
        self.current_volume = 50

        # Distance range for volume control (in pixels)
        self.min_dist = 30   # Fingers close = min volume
        self.max_dist = 250  # Fingers apart = max volume

        # Visual settings
        self.vol_bar_x = 50
        self.vol_bar_y = 150
        self.vol_bar_width = 40
        self.vol_bar_height = 400

        # Initialize system volume control
        if PYCAW_AVAILABLE:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.vol_range = self.volume.GetVolumeRange()
                self.min_vol = self.vol_range[0]
                self.max_vol = self.vol_range[1]
            except Exception as e:
                print(f"Error initializing volume control: {e}")
                self.volume = None
        else:
            self.volume = None
            self.min_vol = -65.25
            self.max_vol = 0.0

    def get_hand_distance(self, hand):
        """
        Get distance between thumb tip and index finger tip.

        Args:
            hand: Hand data from detector

        Returns:
            Distance between thumb and index finger, center point
        """
        lm_list = hand["lmList"]

        # Get thumb tip (4) and index finger tip (8) positions
        thumb_tip = lm_list[4][:2]
        index_tip = lm_list[8][:2]

        # Calculate distance
        distance = math.hypot(index_tip[0] - thumb_tip[0],
                             index_tip[1] - thumb_tip[1])

        # Center point for visual feedback
        center = ((thumb_tip[0] + index_tip[0]) // 2,
                 (thumb_tip[1] + index_tip[1]) // 2)

        return distance, center, thumb_tip, index_tip

    def set_volume(self, volume_percent):
        """
        Set system volume.

        Args:
            volume_percent: Volume level (0-100)
        """
        self.current_volume = np.clip(volume_percent, 0, 100)

        if self.volume is not None:
            # Convert percentage to actual volume level
            vol = np.interp(
                volume_percent,
                [0, 100],
                [self.min_vol, self.max_vol]
            )
            self.volume.SetMasterVolumeLevel(vol, None)

    def draw_volume_bar(self, img, volume_percent):
        """
        Draw volume indicator bar on image.

        Args:
            img: Image to draw on
            volume_percent: Current volume percentage

        Returns:
            Image with volume bar
        """
        # Background bar
        cv2.rectangle(
            img,
            (self.vol_bar_x, self.vol_bar_y),
            (self.vol_bar_x + self.vol_bar_width,
             self.vol_bar_y + self.vol_bar_height),
            (50, 50, 50), -1
        )

        # Calculate filled height
        filled_height = int((volume_percent / 100) * self.vol_bar_height)
        fill_y = self.vol_bar_y + self.vol_bar_height - filled_height

        # Color gradient based on volume
        if volume_percent < 30:
            color = (0, 255, 0)  # Green
        elif volume_percent < 70:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Filled bar
        cv2.rectangle(
            img,
            (self.vol_bar_x, fill_y),
            (self.vol_bar_x + self.vol_bar_width,
             self.vol_bar_y + self.vol_bar_height),
            color, -1
        )

        # Border
        cv2.rectangle(
            img,
            (self.vol_bar_x, self.vol_bar_y),
            (self.vol_bar_x + self.vol_bar_width,
             self.vol_bar_y + self.vol_bar_height),
            (255, 255, 255), 2
        )

        # Volume percentage text
        cv2.putText(
            img, f"{int(volume_percent)}%",
            (self.vol_bar_x - 10, self.vol_bar_y + self.vol_bar_height + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        return img

    def draw_gesture_feedback(self, img, thumb, index, center, distance):
        """
        Draw visual feedback for the gesture.

        Args:
            img: Image to draw on
            thumb: Thumb tip position
            index: Index finger tip position
            center: Center point between fingers
            distance: Distance between fingers

        Returns:
            Image with gesture visualization
        """
        # Draw circles at fingertips
        cv2.circle(img, thumb, 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, index, 15, (255, 0, 255), cv2.FILLED)

        # Draw line between fingers
        cv2.line(img, thumb, index, (255, 0, 255), 3)

        # Draw center circle
        cv2.circle(img, center, 10, (0, 255, 0), cv2.FILLED)

        # Change color when fingers very close
        if distance < 50:
            cv2.circle(img, center, 15, (0, 255, 0), cv2.FILLED)

        return img

    def run(self):
        """Main loop for volume control."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        prev_time = 0
        smooth_volume = self.current_volume

        print("=" * 50)
        print("   VOLUME HAND CONTROL")
        print("=" * 50)
        print("Use thumb and index finger distance to control volume")
        print("Fingers close = Low volume")
        print("Fingers apart = High volume")
        print("Press 'q' to quit")
        print("=" * 50)

        if not PYCAW_AVAILABLE:
            print("\n[Simulation Mode - Install pycaw for actual volume control]")
            print("pip install pycaw\n")

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            img = cv2.flip(img, 1)

            # Detect hands
            hands, img = self.detector.find_hands(img)

            if hands:
                hand = hands[0]
                
                distance, center, thumb, index = self.get_hand_distance(hand)

                # Map distance to volume
                volume = np.interp(
                    distance,
                    [self.min_dist, self.max_dist],
                    [0, 100]
                )
                volume = np.clip(volume, 0, 100)

                # Smooth the volume change
                smooth_volume = smooth_volume * 0.8 + volume * 0.2

                # Set volume
                self.set_volume(smooth_volume)

                # Draw gesture feedback
                img = self.draw_gesture_feedback(
                    img, thumb, index, center, distance
                )

            # Draw volume bar
            img = self.draw_volume_bar(img, smooth_volume)

            # Mode indicator
            mode_text = "VOLUME CONTROL" if PYCAW_AVAILABLE else "SIMULATION"
            cv2.putText(
                img, mode_text, (self.width - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
            )

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(
                img, f"FPS: {int(fps)}", (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
            )

            cv2.imshow("Volume Hand Control", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Run the volume hand control application."""
    controller = VolumeController()
    controller.run()


if __name__ == "__main__":
    main()
