import math
import cv2
import numpy as np
import mediapipe as mp

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

# Virtual Keyboard Controller
from pynput.keyboard import Key, Controller

keyboard = Controller()


class HandGestureController:
    """Handles MediaPipe Hand Landmark Processing."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def is_fist(self, landmarks):
        wrist = landmarks[0]
        fingertip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]

        closed_fingers = 0
        for tip, pip in zip(fingertip_ids, pip_ids):
            dist_tip = math.hypot(landmarks[tip].x - wrist.x, landmarks[tip].y - wrist.y)
            dist_pip = math.hypot(landmarks[pip].x - wrist.x, landmarks[pip].y - wrist.y)
            if dist_tip < dist_pip:
                closed_fingers += 1

        return closed_fingers >= 3

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                cx, cy = int(landmarks[0].x * w), int(landmarks[0].y * h)
                fist_state = self.is_fist(landmarks)

                hand_data.append({
                    "center": (cx, cy),
                    "is_fist": fist_state,
                    "landmarks": landmarks
                })

        return hand_data


class SplitScreenSteeringAppLayout(BoxLayout):
    """
    Split-screen application:
    - Left 50%: Camera Feed & Vision HUD
    - Right 50%: Keyboard Control Panel & Game Telemetry Instructions
    """
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)

        # Active Pressed Key Tracker (Prevents Key Spamming)
        self.pressed_keys = set()

        # --- LEFT SIDE: Camera Feed (50% Width) ---
        self.left_panel = FloatLayout(size_hint=(0.5, 1))
        self.camera_image = Image(size_hint=(1, 1), pos_hint={'x': 0, 'y': 0})
        self.left_panel.add_widget(self.camera_image)
        self.add_widget(self.left_panel)

        # --- RIGHT SIDE: Game Telemetry & Control Output (50% Width) ---
        self.right_panel = FloatLayout(size_hint=(0.5, 1))
        
        self.instructions_label = Label(
            text="[ CRAZYGAMES CONTROLLER ]\n\n"
                 "1. Open 'Racing Limits' on CrazyGames in your browser.\n"
                 "2. Click inside the browser game window so it receives keys.\n"
                 "3. Use hand gestures to drive:\n\n"
                 "   • Closed Fists = Accelerate (W)\n"
                 "   • Open Hands = Brake (S)\n"
                 "   • Turn Steering Wheel Left/Right = Steer (A / D)\n",
            pos_hint={'x': 0.05, 'y': 0.4},
            size_hint=(0.9, 0.55),
            font_size='16sp',
            halign='left',
            valign='top'
        )
        self.instructions_label.bind(size=self.instructions_label.setter('text_size'))
        self.right_panel.add_widget(self.instructions_label)

        self.telemetry_label = Label(
            text="Status: READY\nPedal: NONE\nSteering: CENTER\nKeys Sent: NONE",
            pos_hint={'x': 0.05, 'y': 0.05},
            size_hint=(0.9, 0.3),
            font_size='18sp',
            bold=True,
            halign='left'
        )
        self.telemetry_label.bind(size=self.telemetry_label.setter('text_size'))
        self.right_panel.add_widget(self.telemetry_label)

        self.add_widget(self.right_panel)

        # Tracking Engine & Camera
        self.tracker = HandGestureController()
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 30 FPS Main Execution Loop
        Clock.schedule_interval(self.game_loop, 1.0 / 30.0)

    def set_key_state(self, key_char, should_press):
        """Helper to manage key presses cleanly."""
        if should_press:
            if key_char not in self.pressed_keys:
                keyboard.press(key_char)
                self.pressed_keys.add(key_char)
        else:
            if key_char in self.pressed_keys:
                keyboard.release(key_char)
                self.pressed_keys.remove(key_char)

    def game_loop(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        hand_data = self.tracker.process_frame(frame)

        is_accelerating = False
        is_braking = False
        steering_angle = 0.0

        if len(hand_data) == 2:
            hand_data = sorted(hand_data, key=lambda d: d["center"][0])
            h1, h2 = hand_data[0], hand_data[1]

            # Fists = Accelerate (W) | Open Hands = Brake (S)
            if h1["is_fist"] and h2["is_fist"]:
                is_accelerating = True
            elif not h1["is_fist"] and not h2["is_fist"]:
                is_braking = True

            # Steering Angle Calculation
            p1, p2 = h1["center"], h2["center"]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if dx != 0:
                steering_angle = math.degrees(math.atan2(-dy, dx))

            # Visual Feedback on Video Stream
            center_x = (p1[0] + p2[0]) // 2
            center_y = (p1[1] + p2[1]) // 2
            radius = int(math.hypot(dx, dy) / 2)

            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 0), 2)
            cv2.circle(frame, p1, 12, (0, 0, 255) if h1["is_fist"] else (0, 255, 0), -1)
            cv2.circle(frame, p2, 12, (0, 0, 255) if h2["is_fist"] else (0, 255, 0), -1)
            cv2.line(frame, p1, p2, (0, 255, 255), 3)

        elif len(hand_data) == 1:
            h1 = hand_data[0]
            if h1["is_fist"]:
                is_accelerating = True
            else:
                is_braking = True
            cv2.circle(frame, h1["center"], 12, (0, 0, 255) if h1["is_fist"] else (0, 255, 0), -1)

        # Send Key Events to System
        self.set_key_state('w', is_accelerating)
        self.set_key_state('s', is_braking)

        steer_text = "CENTER"
        steer_left = steering_angle < -12
        steer_right = steering_angle > 12

        self.set_key_state('a', steer_left)
        self.set_key_state('d', steer_right)

        if steer_left:
            steer_text = f"LEFT ({int(steering_angle)}°)"
        elif steer_right:
            steer_text = f"RIGHT ({int(steering_angle)}°)"

        # Update Telemetry Display
        pedal_text = "ACCELERATE (W)" if is_accelerating else ("BRAKE (S)" if is_braking else "COAST")
        active_keys_str = ", ".join(sorted(list(self.pressed_keys))).upper() if self.pressed_keys else "NONE"

        self.telemetry_label.text = (
            f"Pedal State: {pedal_text}\n"
            f"Steering: {steer_text}\n"
            f"Keys Outputting: [{active_keys_str}]"
        )

        # Render Camera Frame to Left Panel
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(w, h), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_image.texture = texture

    def on_stop(self):
        # Clean up any stuck key inputs on app close
        for k in list(self.pressed_keys):
            keyboard.release(k)
        if self.capture.isOpened():
            self.capture.release()


class CrazyGamesControllerApp(App):
    def build(self):
        return SplitScreenSteeringAppLayout()

    def on_stop(self):
        self.root.on_stop()


if __name__ == '__main__':
    CrazyGamesControllerApp().run()