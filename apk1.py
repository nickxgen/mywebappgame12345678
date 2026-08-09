import cv2
import math
import sys
import numpy as np
import pygame
import mediapipe as mp

# --- CONFIGURATION & CONSTANTS ---
VIEW_WIDTH = 640
VIEW_HEIGHT = 640
SCREEN_WIDTH = VIEW_WIDTH * 2
SCREEN_HEIGHT = VIEW_HEIGHT
FPS = 30

# Colors
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
YELLOW = (255, 215, 0)
BLUE = (50, 120, 220)
GREEN = (50, 200, 50)
BLACK = (0, 0, 0)
RED = (200, 50, 50)

# Game Parameters
ROAD_WIDTH = 300
ROAD_LEFT = VIEW_WIDTH + (VIEW_WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH
CAR_WIDTH = 35
CAR_HEIGHT = 60
CAR_SPEED = 8
STEERING_DEADZONE = 12.0

# --- FILTERS ---

def apply_beauty_filter(img):
    """Applies skin smoothing and color enhancement."""
    smoothed = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 15)
    v = cv2.add(v, 10)
    enhanced_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

def apply_dark_noir_filter(img):
    """Applies a high-intensity dark, high-contrast noir filter."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast and decrease brightness
    # formula: output = alpha * input + beta
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = -50  # Brightness control (0-100)
    dark_high_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Add a vignette effect
    rows, cols = dark_high_contrast.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask / mask.max()
    
    vignette = (dark_high_contrast * mask).astype(np.uint8)
    
    # Convert back to BGR for consistency
    return cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)

def apply_deep_dark_filter(img):
    """Applies a 'filter full' deep dark moody look with low exposure."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, 0.35).astype(np.uint8) # Darker
    s = cv2.multiply(s, 1.5).astype(np.uint8)  # More 'Full' color saturation
    dark_hsv = cv2.merge((h, s, v))
    dark_bgr = cv2.cvtColor(dark_hsv, cv2.COLOR_HSV2BGR)
    return cv2.GaussianBlur(dark_bgr, (3, 3), 0)

def apply_midnight_filter(img):
    """An extreme 'Midnight' filter - high intensity black and dark."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Only keep the brightest parts, everything else becomes black
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)
    # High contrast on what remains
    midnight = cv2.convertScaleAbs(thresholded, alpha=2.0, beta=-100)
    return cv2.cvtColor(midnight, cv2.COLOR_GRAY2BGR)

# --- UTILITIES ---

def is_fist(hand_landmarks):
    """Detects if a hand is closed into a fist."""
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
    closed_fingers = sum(1 for tip, pip in zip(tips, pips) 
                        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y)
    return closed_fingers >= 3

# --- INITIALIZATION ---

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pro Virtual Steering - Dark Mode")

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    # Fallback to a black frame if camera fails
    dummy_frame = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)

# Game State
car_x = ROAD_LEFT + (ROAD_WIDTH // 2) - (CAR_WIDTH // 2)
car_y = SCREEN_HEIGHT - 100
score = 0
line_y = 0
road_speed = 7
current_filter = 3 # Default to Deep Dark as requested. 0: None, 1: Beauty, 2: Dark Noir, 3: Deep Dark, 4: Midnight

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24, bold=True)

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
    
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (VIEW_WIDTH, VIEW_HEIGHT))

        # Apply Selected Filter
        if current_filter == 1:
            frame = apply_beauty_filter(frame)
        elif current_filter == 2:
            frame = apply_dark_noir_filter(frame)
        elif current_filter == 3:
            frame = apply_deep_dark_filter(frame)
        elif current_filter == 4:
            frame = apply_midnight_filter(frame)

        # Hand Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_wrist = None
        right_wrist = None
        fist_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                cx, cy = int(wrist.x * VIEW_WIDTH), int(wrist.y * VIEW_HEIGHT)

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
            if wheel_radius > 20:
                cv2.circle(frame, (center_x, center_y), wheel_radius, YELLOW, 4)
                cv2.circle(frame, (center_x, center_y), 12, (0, 0, 255), -1)
                cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 4)

        # UI Overlays on Camera
        filter_names = ["Original", "Beauty", "Dark Noir", "Deep Dark", "Midnight"]
        cv2.putText(frame, f"Filter: {filter_names[current_filter]} (Press 0-4)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        if fist_detected:
            accel_state = "ACCEL"
            cv2.putText(frame, "ACCELERATING", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "IDLE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), (VIEW_WIDTH, VIEW_HEIGHT), "RGB")
    else:
        frame_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        frame_surface.fill(BLACK)

    # 2. GAME LOGIC
    if steer_state == "LEFT": car_x -= CAR_SPEED
    elif steer_state == "RIGHT": car_x += CAR_SPEED

    road_speed = 15 if accel_state == "ACCEL" else 4
    
    # Boundary Check
    car_x = max(ROAD_LEFT + 5, min(car_x, ROAD_RIGHT - CAR_WIDTH - 5))
    
    line_y = (line_y + road_speed) % 40
    score += int(road_speed / 2)

    # 3. RENDERING
    screen.blit(frame_surface, (0, 0))

    # Game World
    pygame.draw.rect(screen, GREEN, (VIEW_WIDTH, 0, VIEW_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, GRAY, (ROAD_LEFT, 0, ROAD_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, (ROAD_LEFT - 10, 0, 10, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, (ROAD_RIGHT, 0, 10, SCREEN_HEIGHT))

    # Road Lines
    for i in range(-40, SCREEN_HEIGHT, 40):
        pygame.draw.rect(screen, YELLOW, (VIEW_WIDTH + (VIEW_WIDTH // 2) - 5, i + line_y, 10, 20))

    # Draw Car (More stylish)
    pygame.draw.rect(screen, BLACK, (car_x-2, car_y-2, CAR_WIDTH+4, CAR_HEIGHT+4), border_radius=8) # Shadow
    pygame.draw.rect(screen, BLUE, (car_x, car_y, CAR_WIDTH, CAR_HEIGHT), border_radius=6)
    pygame.draw.rect(screen, WHITE, (car_x+5, car_y+10, CAR_WIDTH-10, 15), border_radius=2) # Windshield

    # HUD
    panel = pygame.Surface((200, 100), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    screen.blit(panel, (VIEW_WIDTH + 10, 10))
    
    score_surf = font.render(f"SCORE: {score}", True, WHITE)
    speed_surf = font.render(f"SPEED: {road_speed * 10} km/h", True, YELLOW)
    screen.blit(score_surf, (VIEW_WIDTH + 25, 25))
    screen.blit(speed_surf, (VIEW_WIDTH + 25, 60))

    pygame.display.flip()
    clock.tick(FPS)

cap.release()
pygame.quit()
sys.exit()
