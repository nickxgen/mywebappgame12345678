import cv2
import math
import sys
import numpy as np
import pygame
import mediapipe as mp

# --- CONFIGURATION & CONSTANTS ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30

# Webcam Overlay (Bottom-Right)
CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_POS_X = SCREEN_WIDTH - CAM_WIDTH - 20
CAM_POS_Y = SCREEN_HEIGHT - CAM_HEIGHT - 20

# Colors
GRAY = (40, 40, 40)
DARK_GRAY = (25, 25, 25)
WHITE = (255, 255, 255)
YELLOW = (255, 215, 0)
BLUE = (50, 120, 220)
GREEN = (34, 139, 34)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
DARK_RED = (139, 0, 0)

# Gear limits
GEAR_LIMITS = {1: 6, 2: 13, 3: 22}

# --- IMAGE FILTERS ---

def apply_beauty_filter(img):
    smoothed = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 10)
    v = cv2.add(v, 5)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def apply_dark_noir_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = cv2.convertScaleAbs(gray, alpha=1.6, beta=-40)
    rows, cols = dark.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    mask = (kernel_y * kernel_x.T)
    mask = mask / mask.max()
    vignette = (dark * mask).astype(np.uint8)
    return cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)

def apply_deep_dark_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, 0.4).astype(np.uint8)
    s = cv2.multiply(s, 1.6).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def apply_midnight_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 80, 255, cv2.THRESH_TOZERO)
    midnight = cv2.convertScaleAbs(thresholded, alpha=2.2, beta=-80)
    return cv2.cvtColor(midnight, cv2.COLOR_GRAY2BGR)

# --- GESTURE RECOGNITION ---

def is_hand_open(landmarks):
    """Detects open hand by comparing fingertip distances from wrist."""
    wrist = landmarks.landmark[0]
    tips = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky tips
    bases = [5, 9, 13, 17]   # Corresponding MCP joints
    
    open_fingers = 0
    for tip, base in zip(tips, bases):
        tip_lm = landmarks.landmark[tip]
        base_lm = landmarks.landmark[base]
        tip_dist = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
        base_dist = math.hypot(base_lm.x - wrist.x, base_lm.y - wrist.y)
        if tip_dist > base_dist * 1.3:
            open_fingers += 1
    return open_fingers >= 3

# --- INITIALIZATION ---

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Supercar - Dual WASD & Gesture Controls with Gearbox")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Fonts & Telemetry
font_main = pygame.font.SysFont("Arial", 28, bold=True)
font_sub = pygame.font.SysFont("Arial", 18, bold=True)
clock = pygame.time.Clock()

# State Variables
current_filter = 0
current_gear = 1
current_speed = 0.0
target_speed = 0.0
score = 0
car_x = SCREEN_WIDTH // 2
steer_angle = 0.0

# Gesture State & Debounce
last_right_open = False
last_left_open = False
right_shift_cooldown = 0
left_shift_cooldown = 0

# 3D Environment Setup
trees = []
for z in range(50, 600, 40):
    trees.append([-180, z])
    trees.append([180, z + 20])

stripe_z_offsets = [z for z in range(0, 500, 50)]

# --- DRAWING HELPER FUNCTIONS ---

def draw_gearbox_ui(surface, gear, x=20, y=260, width=120, height=180):
    """Draws an interactive manual gear shift pattern UI on the left side."""
    # UI Container Box
    ui_bg = pygame.Surface((width, height), pygame.SRCALPHA)
    ui_bg.fill((0, 0, 0, 190))
    surface.blit(ui_bg, (x, y))
    pygame.draw.rect(surface, BLUE, (x, y, width, height), width=2, border_radius=6)

    # Title
    lbl = font_sub.render("GEAR BOX", True, WHITE)
    surface.blit(lbl, (x + (width - lbl.get_width()) // 2, y + 10))

    # Gear Gates Layout (x, y relative to UI box)
    gear_positions = {
        1: (x + 35, y + 50),
        2: (x + 85, y + 50),
        3: (x + 60, y + 130)
    }

    center_h = y + 90
    
    # Draw Shift Track Lines
    pygame.draw.line(surface, WHITE, (gear_positions[1][0], gear_positions[1][1]), (gear_positions[1][0], center_h), 4)
    pygame.draw.line(surface, WHITE, (gear_positions[2][0], gear_positions[2][1]), (gear_positions[2][0], center_h), 4)
    pygame.draw.line(surface, WHITE, (gear_positions[1][0], center_h), (gear_positions[2][0], center_h), 4)
    pygame.draw.line(surface, WHITE, (gear_positions[3][0], center_h), (gear_positions[3][0], gear_positions[3][1]), 4)

    # Draw Gear Knobs
    for g, pos in gear_positions.items():
        is_active = (g == gear)
        color = YELLOW if is_active else GRAY
        border_col = WHITE if is_active else DARK_GRAY
        
        pygame.draw.circle(surface, color, pos, 14)
        pygame.draw.circle(surface, border_col, pos, 14, 2)
        
        txt_col = BLACK if is_active else WHITE
        g_txt = font_sub.render(str(g), True, txt_col)
        surface.blit(g_txt, (pos[0] - g_txt.get_width() // 2, pos[1] - g_txt.get_height() // 2))

    # Current Knob Position Indicator
    active_pos = gear_positions[gear]
    pygame.draw.circle(surface, RED, active_pos, 5)


# --- MAIN LOOP ---

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: current_gear = 1
            elif event.key == pygame.K_2: current_gear = 2
            elif event.key == pygame.K_3: current_gear = 3
            elif event.key == pygame.K_4: current_filter = 4
            elif event.key == pygame.K_0: current_filter = 0
            elif event.key == pygame.K_r: 
                car_x = SCREEN_WIDTH // 2
                steer_angle = 0
                current_speed = 0
            elif event.key == pygame.K_ESCAPE: running = False

    # 1. CAMERA & GESTURE PROCESSING
    ret, frame = cap.read()
    frame_surface = None
    right_hand_state = "--"
    left_hand_state = "--"
    hand_detected = False
    hand_steer_target = 0.0
    gesture_accel = False
    gesture_brake = False

    if right_shift_cooldown > 0: right_shift_cooldown -= 1
    if left_shift_cooldown > 0: left_shift_cooldown -= 1

    if ret:
        frame = cv2.flip(frame, 1)
        cam_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        # Apply Selected Filter
        if current_filter == 1: cam_frame = apply_beauty_filter(cam_frame)
        elif current_filter == 2: cam_frame = apply_dark_noir_filter(cam_frame)
        elif current_filter == 3: cam_frame = apply_deep_dark_filter(cam_frame)
        elif current_filter == 4: cam_frame = apply_midnight_filter(cam_frame)

        # MediaPipe Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        found_right = False
        found_left = False
        right_open = False
        left_open = False

        left_wrist_pt = None
        right_wrist_pt = None

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_detected = True
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label # 'Left' or 'Right'
                
                # Draw Landmarks
                mp_draw.draw_landmarks(cam_frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                is_open = is_hand_open(landmarks)
                wrist = landmarks.landmark[0]
                cx, cy = int(wrist.x * CAM_WIDTH), int(wrist.y * CAM_HEIGHT)

                if label == 'Right':
                    found_right = True
                    right_open = is_open
                    right_wrist_pt = (cx, cy)
                    right_hand_state = "OPEN (Brake)" if right_open else "CLOSED (Accel)"

                    if right_open: gesture_brake = True
                    else: gesture_accel = True

                    # Shift UP: OPEN -> CLOSED
                    if last_right_open and not right_open and right_shift_cooldown == 0:
                        if current_gear < 3:
                            current_gear += 1
                            right_shift_cooldown = 10
                    last_right_open = right_open

                elif label == 'Left':
                    found_left = True
                    left_open = is_open
                    left_wrist_pt = (cx, cy)
                    left_hand_state = "OPEN" if left_open else "CLOSED"

                    # Shift DOWN: OPEN -> CLOSED
                    if last_left_open and not left_open and left_shift_cooldown == 0:
                        if current_gear > 1:
                            current_gear -= 1
                            left_shift_cooldown = 10
                    last_left_open = left_open

        # Steering Calculation via Gestures
        if left_wrist_pt and right_wrist_pt:
            lx, ly = left_wrist_pt
            rx, ry = right_wrist_pt
            angle_rad = math.atan2(ry - ly, rx - lx)
            hand_steer_target = np.clip(math.degrees(angle_rad) * 1.5, -45, 45)
            center_pt = ((lx + rx) // 2, (ly + ry) // 2)
            cv2.circle(cam_frame, center_pt, 30, (0, 255, 255), 2)
            cv2.line(cam_frame, (lx, ly), (rx, ry), (255, 0, 255), 2)
        elif right_wrist_pt:
            raw_offset = (right_wrist_pt[0] / CAM_WIDTH) - 0.5
            hand_steer_target = np.clip(raw_offset * 90, -45, 45)

        cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(cam_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")

    # 2. INPUT COMBINATION (KEYBOARD WASD + GESTURES)
    keys = pygame.key.get_pressed()
    
    kb_accel = keys[pygame.K_w]
    kb_brake = keys[pygame.K_s]
    kb_left = keys[pygame.K_a]
    kb_right = keys[pygame.K_d]

    # Combine Keyboard Steering with Gesture Steering
    kb_steer_target = 0.0
    if kb_left: kb_steer_target -= 35.0
    if kb_right: kb_steer_target += 35.0

    target_steer = kb_steer_target if kb_steer_target != 0.0 else hand_steer_target

    # Determine Acceleration / Braking
    if kb_accel or gesture_accel:
        target_speed = float(GEAR_LIMITS[current_gear])
    elif kb_brake or gesture_brake:
        target_speed = 0.0
    else:
        target_speed = 1.0  # Idle Coasting

    # 3. VEHICLE PHYSICS
    if current_speed < target_speed:
        current_speed += 0.20
    elif current_speed > target_speed:
        current_speed -= 0.40
    current_speed = max(0.0, current_speed)

    steer_angle += (target_steer - steer_angle) * 0.15
    car_x += (steer_angle * 0.15) * (max(current_speed, 3.0) / 8.0)
    car_x = np.clip(car_x, 380, SCREEN_WIDTH - 380)

    score += int(current_speed / 4)

    # 4. 3D PERSPECTIVE RENDERING
    screen.fill((26, 43, 60))

    horizon_y = 250
    bottom_y = SCREEN_HEIGHT
    road_top_w = 120
    road_bot_w = 600

    vp_x = SCREEN_WIDTH // 2
    road_poly = [
        (vp_x - road_top_w // 2, horizon_y),
        (vp_x + road_top_w // 2, horizon_y),
        (vp_x + road_bot_w // 2, bottom_y),
        (vp_x - road_bot_w // 2, bottom_y)
    ]
    pygame.draw.polygon(screen, GREEN, [(0, horizon_y), (SCREEN_WIDTH, horizon_y), (SCREEN_WIDTH, bottom_y), (0, bottom_y)])
    pygame.draw.polygon(screen, GRAY, road_poly)

    # Move and Draw Road Stripes
    move_step = current_speed * 1.2
    for i in range(len(stripe_z_offsets)):
        stripe_z_offsets[i] = (stripe_z_offsets[i] + move_step) % 500
        z = stripe_z_offsets[i]
        scale = z / 500.0
        sy = int(horizon_y + scale * (bottom_y - horizon_y))
        sw = int(4 + scale * 16)
        sh = int(2 + scale * 12)
        pygame.draw.rect(screen, YELLOW, (vp_x - sw // 2, sy, sw, sh))

    # Environment Trees
    for tree in trees:
        tree[1] = (tree[1] + move_step) % 550
        scale = max(0.05, tree[1] / 550.0)
        tx = int(vp_x + tree[0] * scale)
        ty = int(horizon_y + scale * (bottom_y - horizon_y))
        tw = int(20 * scale)
        th = int(40 * scale)
        if 0 <= ty <= SCREEN_HEIGHT:
            pygame.draw.rect(screen, (92, 51, 23), (tx - tw//4, ty, tw//2, th//2))
            pygame.draw.polygon(screen, (34, 139, 34), [(tx - tw, ty), (tx + tw, ty), (tx, ty - th)])

    # Draw Supercar
    car_y = SCREEN_HEIGHT - 130
    car_w, car_h = 90, 110
    
    pygame.draw.rect(screen, BLACK, (car_x - car_w//2 - 4, car_y - 4, car_w + 8, car_h + 8), border_radius=12)
    pygame.draw.rect(screen, RED, (car_x - car_w//2, car_y, car_w, car_h), border_radius=10)
    
    # Cabin Glass & Roof Perspective
    pygame.draw.polygon(screen, (30, 30, 30), [
        (car_x - 30, car_y + 20), (car_x + 30, car_y + 20),
        (car_x + 22, car_y + 55), (car_x - 22, car_y + 55)
    ])
    pygame.draw.rect(screen, DARK_RED, (car_x - car_w//2 + 8, car_y + car_h - 12, 20, 8))
    pygame.draw.rect(screen, DARK_RED, (car_x + car_w//2 - 28, car_y + car_h - 12, 20, 8))

    # Draw Webcam Overlay
    if frame_surface:
        pygame.draw.rect(screen, RED, (CAM_POS_X - 3, CAM_POS_Y - 3, CAM_WIDTH + 6, CAM_HEIGHT + 6), border_radius=8)
        screen.blit(frame_surface, (CAM_POS_X, CAM_POS_Y))

    # 5. UI OVERLAYS
    # Main Dashboard Telemetry
    hud_bg = pygame.Surface((300, 220), pygame.SRCALPHA)
    hud_bg.fill((0, 0, 0, 190))
    screen.blit(hud_bg, (20, 20))

    filter_names = ["Original", "Beauty", "Dark Noir", "Deep Dark", "Midnight"]
    hud_lines = [
        (f"SCORE: {score}", WHITE),
        (f"SPEED: {int(current_speed * 10)} km/h", YELLOW),
        (f"GEAR: {current_gear} / 3", (0, 255, 0)),
        (f"STEER: {int(steer_angle)}°", WHITE),
        (f"R-HAND: {right_hand_state}", RED if "OPEN" in right_hand_state else (0, 255, 0)),
        (f"L-HAND: {left_hand_state}", RED if "OPEN" in left_hand_state else WHITE),
        (f"FILTER [0-4]: {filter_names[current_filter]}", CYAN)
    ]

    for idx, (text, color) in enumerate(hud_lines):
        txt_surf = font_sub.render(text, True, color)
        screen.blit(txt_surf, (35, 30 + idx * 26))

    # Render Side Gearbox UI
    draw_gearbox_ui(screen, current_gear, x=20, y=260)

    pygame.display.flip()
    clock.tick(FPS)

cap.release()
pygame.quit()
sys.exit()