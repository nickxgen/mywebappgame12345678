import cv2
import math
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller

# ==================== TUNING CONFIGURATION ====================
STEERING_DEADZONE = 8.0      # Deadzone in degrees for straight driving
SMOOTHING_FACTOR = 0.30      # EMA weight for smooth steering (0.1 = smooth, 1.0 = instant)

ACCEL_THRESHOLD = 0.40       # Y-ratio upper bound (Top 40% = Accelerate)
BRAKE_THRESHOLD = 0.65       # Y-ratio lower bound (Bottom 35% = Brake)

# Beauty Filter Toggles
ENABLE_BEAUTY_FILTER = True
# ===============================================================

keyboard = Controller()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

currently_pressed = set()
smoothed_angle = 0.0
smoothed_center_y = 0.0

def apply_beauty_filter(img):
    """Smooths skin while maintaining edge detail and boosting warmth/contrast."""
    # 1. Bilateral filter for skin smoothing without blurring edges
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. Adjust contrast and brightness (1.1x contrast, +10 brightness)
    enhanced = cv2.convertScaleAbs(smooth, alpha=1.1, beta=10)
    
    # 3. Soft warm color grade
    b, g, r = cv2.split(enhanced)
    r = cv2.add(r, 8)  # Boost red channel slightly for warm skin tones
    b = cv2.subtract(b, 5) # Reduce harsh blue tint
    
    merged = cv2.merge([b, g, r])
    
    # 4. Subtle blend between smooth image and original for natural skin texture
    return cv2.addWeighted(merged, 0.75, img, 0.25, 0)

def press_key(key):
    if key not in currently_pressed:
        keyboard.press(key)
        currently_pressed.add(key)

def release_key(key):
    if key in currently_pressed:
        keyboard.release(key)
        currently_pressed.remove(key)

def release_all():
    for k in ['a', 'd', 'w', 's']:
        release_key(k)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirrored display
    frame = cv2.flip(frame, 1)
    
    # Apply camera beautification filter
    if ENABLE_BEAUTY_FILTER:
        frame = apply_beauty_filter(frame)
        
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    palms = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw Hand Finger Joints
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

            # Extract Palm Center
            mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            palm_x = int(((mcp.x + wrist.x) / 2) * w)
            palm_y = int(((mcp.y + wrist.y) / 2) * h)
            
            palms.append((palm_x, palm_y))
            cv2.circle(frame, (palm_x, palm_y), 8, (0, 0, 255), -1)

    # Wheel steering logic
    if len(palms) == 2:
        palms.sort(key=lambda pt: pt[0])
        p_left, p_right = palms[0], palms[1]

        lx, ly = p_left
        rx, ry = p_right

        center_x = (lx + rx) // 2
        raw_center_y = (ly + ry) // 2
        
        wheel_radius = int(math.hypot(rx - lx, ry - ly) / 2)
        raw_angle = math.degrees(math.atan2(ry - ly, rx - lx))

        smoothed_angle = (SMOOTHING_FACTOR * raw_angle) + ((1 - SMOOTHING_FACTOR) * smoothed_angle)
        smoothed_center_y = (SMOOTHING_FACTOR * raw_center_y) + ((1 - SMOOTHING_FACTOR) * smoothed_center_y)
        wheel_center = (center_x, int(smoothed_center_y))

        # Steering Wheel Overlay
        cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 4)
        if wheel_radius > 20:
            cv2.circle(frame, wheel_center, wheel_radius, (255, 165, 0), 3)
        cv2.circle(frame, wheel_center, 10, (0, 0, 255), -1)

        # Steering Controls
        if smoothed_angle > STEERING_DEADZONE:
            press_key('d')
            release_key('a')
            steer_status = f"RIGHT ({int(smoothed_angle)}deg)"
            steer_color = (0, 255, 0)
        elif smoothed_angle < -STEERING_DEADZONE:
            press_key('a')
            release_key('d')
            steer_status = f"LEFT ({int(abs(smoothed_angle))}deg)"
            steer_color = (0, 255, 0)
        else:
            release_key('a')
            release_key('d')
            steer_status = "CENTER"
            steer_color = (255, 255, 255)

        # Throttle Controls
        if smoothed_center_y < h * ACCEL_THRESHOLD:
            press_key('w')
            release_key('s')
            throttle_status = "ACCELERATE (W)"
            throttle_color = (0, 255, 0)
        elif smoothed_center_y > h * BRAKE_THRESHOLD:
            press_key('s')
            release_key('w')
            throttle_status = "BRAKE (S)"
            throttle_color = (0, 0, 255)
        else:
            release_key('w')
            release_key('s')
            throttle_status = "COAST"
            throttle_color = (200, 200, 200)

        # UI Text
        cv2.putText(frame, f"STEER: {steer_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, steer_color, 2)
        cv2.putText(frame, f"DRIVE: {throttle_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, throttle_color, 2)

        cv2.line(frame, (0, int(h * ACCEL_THRESHOLD)), (w, int(h * ACCEL_THRESHOLD)), (0, 255, 0), 1)
        cv2.line(frame, (0, int(h * BRAKE_THRESHOLD)), (w, int(h * BRAKE_THRESHOLD)), (0, 0, 255), 1)

    else:
        release_all()
        cv2.putText(frame, "HOLD BOTH PALMS UP TO DRIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Glamour CV Steering Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_all()
cap.release()
cv2.destroyAllWindows()