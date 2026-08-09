import cv2
import math
import sys
import numpy as np
import pygame
import mediapipe as mp

# --- CONFIGURATION & CONSTANTS ---
# Full screen game layout
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30

# Webcam Overlay (Top-Right)
CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_POS_X = SCREEN_WIDTH - CAM_WIDTH - 20
CAM_POS_Y = 20

# Colors
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
YELLOW = (255, 215, 0)
BLUE = (50, 120, 220)
GREEN = (50, 200, 50)
BLACK = (0, 0, 0)
RED = (200, 50, 50)

# Game Parameters
ROAD_WIDTH = 400
ROAD_LEFT = (SCREEN_WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH
CAR_WIDTH = 45
CAR_HEIGHT = 80
CAR_SPEED = 10
STEERING_DEADZONE = 10.0 # Slightly more sensitive

# --- FILTERS ---

def apply_beauty_filter(img):
    """Applies skin smoothing and color enhancement."""
    # Reduced bilateral filter strength to prevent 'blurry' look
    smoothed = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 10)
    v = cv2.add(v, 5)
    enhanced_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

def apply_dark_noir_filter(img):
    """Applies a high-intensity dark, high-contrast noir filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha = 1.6 # Higher contrast
    beta = -40  # Darker
    dark_high_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    rows, cols = dark_high_contrast.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask / mask.max()
    vignette = (dark_high_contrast * mask).astype(np.uint8)
    return cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)

def apply_deep_dark_filter(img):
    """Applies a 'filter full' deep dark moody look."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, 0.4).astype(np.uint8) # Darker
    s = cv2.multiply(s, 1.6).astype(np.uint8) # More 'Full' color saturation
    dark_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(dark_hsv, cv2.COLOR_HSV2BGR)

def apply_midnight_filter(img):
    """An extreme 'Midnight' filter - high intensity black and dark."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 80, 255, cv2.THRESH_TOZERO)
    midnight = cv2.convertScaleAbs(thresholded, alpha=2.2, beta=-80)
    return cv2.cvtColor(midnight, cv2.COLOR_GRAY2BGR)

# --- UTILITIES ---

def is_fist(hand_landmarks):
    """Improved fist detection using multiple finger joints."""
    # Tips vs PIP joints (middle joint)
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    
    closed_fingers = 0
    for tip, pip in zip(tips, pips):
        # In a fist, the tip is lower (higher y-coordinate) than the PIP joint
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            closed_fingers += 1
            
    # Thumb check
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    # For thumb, we check horizontal distance relative to palm center for better accuracy
    if abs(thumb_tip.x - thumb_ip.x) < 0.05:
        closed_fingers += 1

    return closed_fingers >= 4

# --- INITIALIZATION ---

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("YouTuber Edition - Virtual Steering")

# MediaPipe Setup - Optimized for detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1, # 1 is better for real-time, use 0 if laggy
    min_detection_confidence=0.5, # Lowered for easier detection
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Game State
car_x = ROAD_LEFT + (ROAD_WIDTH // 2) - (CAR_WIDTH // 2)
car_y = SCREEN_HEIGHT - 120
score = 0
line_y = 0
road_speed = 7
current_filter = 3 

clock = pygame.time.Clock()
font_main = pygame.font.SysFont("Arial", 32, bold=True)
font_sub = pygame.font.SysFont("Arial", 20, bold=True)

# --- MAIN LOOP ---

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: current_filter = 1
            if event.key == pygame.K_2: current_filter = 2
            if event.key == pygame.K_3: current_filter = 3
            if event.key == pygame.K_4: current_filter = 4
            if event.key == pygame.K_0: current_filter = 0
            if event.key == pygame.K_ESCAPE: running = False

    # 1. CAMERA & HAND TRACKING
    ret, frame = cap.read()
    steer_state = "CENTER"
    accel_state = "DECEL"
    
    frame_surface = None
    if ret:
        frame = cv2.flip(frame, 1)
        # Resize for the YouTuber overlay
        cam_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        # Apply Selected Filter
        if current_filter == 1: cam_frame = apply_beauty_filter(cam_frame)
        elif current_filter == 2: cam_frame = apply_dark_noir_filter(cam_frame)
        elif current_filter == 3: cam_frame = apply_deep_dark_filter(cam_frame)
        elif current_filter == 4: cam_frame = apply_midnight_filter(cam_frame)

        # Hand Tracking (Process on original frame for better resolution, then draw on cam_frame)
        rgb_frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_wrist = None
        right_wrist = None
        fist_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                # Draw skeleton on the small overlay
                mp_draw.draw_landmarks(cam_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist for steering
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                cx, cy = int(wrist.x * CAM_WIDTH), int(wrist.y * CAM_HEIGHT)

                if label == 'Left': left_wrist = (cx, cy)
                elif label == 'Right': right_wrist = (cx, cy)
                if is_fist(hand_landmarks): fist_detected = True

        # Steering Logic
        if left_wrist and right_wrist:
            lx, ly = left_wrist
            rx, ry = right_wrist
            center_x, center_y = (lx + rx) // 2, (ly + ry) // 2
            angle = math.degrees(math.atan2(ry - ly, rx - lx))

            if angle > STEERING_DEADZONE: steer_state = "RIGHT"
            elif angle < -STEERING_DEADZONE: steer_state = "LEFT"

            wheel_radius = int(math.hypot(rx - lx, ry - ly) / 2)
            if wheel_radius > 15:
                cv2.circle(cam_frame, (center_x, center_y), wheel_radius, YELLOW, 3)
                cv2.line(cam_frame, (lx, ly), (rx, ry), (0, 255, 255), 3)

        # Status Overlay on Webcam
        if fist_detected:
            accel_state = "ACCEL"
            cv2.putText(cam_frame, "ACCEL", (10, CAM_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(cam_frame, "IDLE", (10, CAM_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(cam_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")

    # 2. GAME LOGIC
    if steer_state == "LEFT": car_x -= CAR_SPEED
    elif steer_state == "RIGHT": car_x += CAR_SPEED

    road_speed = 18 if accel_state == "ACCEL" else 5
    car_x = max(ROAD_LEFT + 10, min(car_x, ROAD_RIGHT - CAR_WIDTH - 10))
    line_y = (line_y + road_speed) % 60
    score += int(road_speed / 2)

    # 3. RENDERING
    # Draw Background
    screen.fill(GREEN)
    
    # Draw Road
    pygame.draw.rect(screen, GRAY, (ROAD_LEFT, 0, ROAD_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, (ROAD_LEFT - 15, 0, 15, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, (ROAD_RIGHT, 0, 15, SCREEN_HEIGHT))

    # Road Lines
    for i in range(-60, SCREEN_HEIGHT, 60):
        pygame.draw.rect(screen, YELLOW, (SCREEN_WIDTH // 2 - 5, i + line_y, 10, 30))

    # Draw Car
    pygame.draw.rect(screen, BLACK, (car_x-3, car_y-3, CAR_WIDTH+6, CAR_HEIGHT+6), border_radius=10) # Shadow
    pygame.draw.rect(screen, BLUE, (car_x, car_y, CAR_WIDTH, CAR_HEIGHT), border_radius=8)
    pygame.draw.rect(screen, (100, 200, 255), (car_x+5, car_y+12, CAR_WIDTH-10, 20), border_radius=4) # Windshield
    # Tail lights
    pygame.draw.rect(screen, RED, (car_x+5, car_y+CAR_HEIGHT-10, 10, 5))
    pygame.draw.rect(screen, RED, (car_x+CAR_WIDTH-15, car_y+CAR_HEIGHT-10, 10, 5))

    # Draw Webcam Overlay (YouTuber Style)
    if frame_surface:
        # Border for webcam
        pygame.draw.rect(screen, WHITE, (CAM_POS_X - 4, CAM_POS_Y - 4, CAM_WIDTH + 8, CAM_HEIGHT + 8), border_radius=10)
        screen.blit(frame_surface, (CAM_POS_X, CAM_POS_Y))
        
        # Filter Name
        filter_names = ["Original", "Beauty", "Dark Noir", "Deep Dark", "Midnight"]
        f_text = font_sub.render(f"Filter: {filter_names[current_filter]}", True, WHITE)
        screen.blit(f_text, (CAM_POS_X, CAM_POS_Y + CAM_HEIGHT + 10))

    # HUD
    score_panel = pygame.Surface((250, 120), pygame.SRCALPHA)
    score_panel.fill((0, 0, 0, 180))
    screen.blit(score_panel, (30, 30))
    
    score_surf = font_main.render(f"SCORE: {score}", True, WHITE)
    speed_surf = font_sub.render(f"SPEED: {road_speed * 10} km/h", True, YELLOW)
    screen.blit(score_surf, (50, 50))
    screen.blit(speed_surf, (50, 95))

    pygame.display.flip()
    clock.tick(FPS)

cap.release()
pygame.quit()
sys.exit()
