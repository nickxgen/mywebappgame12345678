import cv2
import math
import mediapipe as mp
from pynput.keyboard import Controller

# ==================== TUNING CONFIGURATION ====================
STEERING_DEADZONE = 12.0    # Angle deadzone for immediate response
ACCEL_Y_RATIO = 0.38        # Top 38% of screen triggers 'W'
BRAKE_Y_RATIO = 0.68        # Bottom 32% of screen triggers 'S'
# ===============================================================

keyboard = Controller()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,              # Lowers CPU usage for zero lag
    min_detection_confidence=0.5,    # Faster initial detection
    min_tracking_confidence=0.5      # Fast frame-to-frame tracking
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

currently_pressed = set()

def press_key(key):
    if key not in currently_pressed:
        keyboard.press(key)
        currently_pressed.add(key)

def release_key(key):
    if key in currently_pressed:
        keyboard.release(key)
        currently_pressed.remove(key)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirrored view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    left_point = None
    right_point = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            # Use Index Finger Tip (Landmark 8)
            finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(finger_tip.x * w), int(finger_tip.y * h)

            if label == 'Left':
                left_point = (cx, cy)
            elif label == 'Right':
                right_point = (cx, cy)

            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

    if left_point and right_point:
        lx, ly = left_point
        rx, ry = right_point

        center_x = (lx + rx) // 2
        center_y = (ly + ry) // 2

        cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 3)
        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)

        angle = math.degrees(math.atan2(ry - ly, rx - lx))

        # --- CORRECTED STEERING DIRECTION FOR MIRRORED VIEW ---
        if angle > STEERING_DEADZONE:
            # Tilting left side down -> Turn RIGHT (D)
            cv2.putText(frame, "STEER: RIGHT (D)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            press_key('d')
            release_key('a')
        elif angle < -STEERING_DEADZONE:
            # Tilting right side down -> Turn LEFT (A)
            cv2.putText(frame, "STEER: LEFT (A)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            press_key('a')
            release_key('d')
        else:
            cv2.putText(frame, "STEER: CENTER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            release_key('a')
            release_key('d')

        # Throttle / Brake Triggers
        if center_y < h * ACCEL_Y_RATIO:
            cv2.putText(frame, "THROTTLE: ACCEL (W)", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            press_key('w')
            release_key('s')
        elif center_y > h * BRAKE_Y_RATIO:
            cv2.putText(frame, "THROTTLE: BRAKE (S)", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            press_key('s')
            release_key('w')
        else:
            release_key('w')
            release_key('s')

    else:
        for k in ['a', 'd', 'w', 's']:
            release_key(k)
        cv2.putText(frame, "Show both hands", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("High Precision Steering", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for k in ['a', 'd', 'w', 's']:
    release_key(k)

cap.release()
cv2.destroyAllWindows()