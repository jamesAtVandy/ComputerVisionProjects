"""
My New Game Hand Tracking
=========================
Hand tracking game implementation using OpenCV and MediaPipe.
Part of the AI Virtual Painter project.
"""

import cv2
import numpy as np
import time
import random
from handtrackingmodule import HandDetector


class HandTrackingGame:
    """
    A simple hand tracking game where players catch falling objects.
    """

    def __init__(self, width=1280, height=720):
        """
        Initialize the game.

        Args:
            width: Game window width
            height: Game window height
        """
        self.width = width
        self.height = height
        self.score = 0
        self.game_over = False
        self.difficulty = 1

        # Object properties
        self.objects = []
        self.object_size = 40
        self.max_objects = 5
        self.spawn_timer = 0
        self.spawn_interval = 60  # frames

        # Hand detector
        self.detector = HandDetector(detection_con=0.7, max_hands=1)

        # Colors
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    def spawn_object(self):
        """Spawn a new falling object at a random x position."""
        if len(self.objects) < self.max_objects:
            x = random.randint(self.object_size, self.width - self.object_size)
            color = random.choice(self.colors)
            speed = random.randint(3, 5 + self.difficulty)
            self.objects.append({
                'x': x,
                'y': -self.object_size,
                'color': color,
                'speed': speed
            })

    def update_objects(self):
        """Update object positions and remove off-screen objects."""
        new_objects = []
        for obj in self.objects:
            obj['y'] += obj['speed']
            if obj['y'] < self.height + self.object_size:
                new_objects.append(obj)
        self.objects = new_objects

    def check_collision(self, hand_center):
        """
        Check if hand caught any objects.

        Args:
            hand_center: (x, y) position of hand center

        Returns:
            Number of objects caught
        """
        caught = 0
        remaining = []
        catch_radius = 80  # Catch distance threshold

        for obj in self.objects:
            distance = ((hand_center[0] - obj['x']) ** 2 +
                       (hand_center[1] - obj['y']) ** 2) ** 0.5
            if distance < catch_radius:
                caught += 1
            else:
                remaining.append(obj)

        self.objects = remaining
        return caught

    def draw_game(self, img, hand_pos=None):
        """
        Draw game elements on the image.

        Args:
            img: Frame to draw on
            hand_pos: Current hand position (x, y)

        Returns:
            Image with game elements drawn
        """
        # Draw falling objects
        for obj in self.objects:
            cv2.circle(img, (obj['x'], obj['y']), self.object_size, obj['color'], -1)
            cv2.circle(img, (obj['x'], obj['y']), self.object_size, (255, 255, 255), 2)

        # Draw hand indicator
        if hand_pos:
            cv2.circle(img, hand_pos, 20, (0, 255, 0), -1)
            cv2.circle(img, hand_pos, catch_radius := 80, (0, 255, 0), 2)

        # Draw score
        cv2.rectangle(img, (10, 10), (250, 80), (50, 50, 50), -1)
        cv2.putText(
            img, f"Score: {self.score}", (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )

        # Draw difficulty level
        cv2.putText(
            img, f"Level: {self.difficulty}", (self.width - 200, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3
        )

        return img

    def run(self):
        """Main game loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        prev_time = 0

        print("=" * 50)
        print("   HAND TRACKING GAME")
        print("=" * 50)
        print("Catch the falling objects with your hand!")
        print("Press 'q' to quit, 'r' to restart")
        print("=" * 50)

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            img = cv2.flip(img, 1)

            # Detect hands
            hands, img = self.detector.find_hands(img, draw=False)

            hand_center = None
            if hands:
                hand = hands[0]
                hand_center = hand["center"]

                # Check collisions
                caught = self.check_collision(hand_center)
                self.score += caught * 10

                # Increase difficulty
                if self.score > 0 and self.score % 100 == 0:
                    self.difficulty = min(10, self.score // 100 + 1)

            # Spawn new objects
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval // self.difficulty:
                self.spawn_object()
                self.spawn_timer = 0

            # Update object positions
            self.update_objects()

            # Draw game elements
            img = self.draw_game(img, hand_center)

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(
                img, f"FPS: {int(fps)}", (10, self.height - 20),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2
            )

            cv2.imshow("Hand Tracking Game", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Restart game
                self.score = 0
                self.difficulty = 1
                self.objects = []

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nFinal Score: {self.score}")
        print(f"Final Level: {self.difficulty}")


def main():
    """Run the hand tracking game."""
    game = HandTrackingGame()
    game.run()


if __name__ == "__main__":
    main()
