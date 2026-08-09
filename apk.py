import cv2
import math
import sys
import random
import numpy as np
import pygame
import mediapipe as mp

# --- CONFIGURATION ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30

# Webcam Overlay (Top-Right)
CAM_WIDTH = 280
CAM_HEIGHT = 210
CAM_POS_X = SCREEN_WIDTH - CAM_WIDTH - 20
CAM_POS_Y = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (80, 80, 80)
DARK_GRAY = (40, 40, 40)
YELLOW = (255, 215, 0)
RED = (220, 20, 60)
SKY_BLUE = (135, 206, 235)
DASH_BOARD = (30, 30, 35)

# Game Physics
MIN_SPEED = 40.0
MAX_SPEED = 111.0
NPC_SPEED = 80.0
ACCEL_RATE = 1.2
DECEL_RATE = 0.8
STEER_SENSITIVITY = 15.0

# --- PSEUDO-3D UTILS ---
HORIZON_Y = SCREEN_HEIGHT // 2

def get_3d_pos(x_offset, z):
    """
    x_offset: -1 to 1 (left to right of road)
    z: distance from player (0 is horizon, 1 is player)
    """
    scale = z
    width_at_z = 100 + (SCREEN_WIDTH * 0.8) * scale
    x = (SCREEN_WIDTH // 2) + (x_offset * width_at_z / 2)
    y = HORIZON_Y + (SCREEN_HEIGHT - HORIZON_Y) * scale
    return int(x), int(y), scale

# --- FILTERS ---
def apply_dark_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, 0.45).astype(np.uint8)
    s = cv2.multiply(s, 1.4).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

# --- HAND TRACKING ---
def is_fist(hand_landmarks):
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    closed = sum(1 for t, p in zip(tips, pips) if hand_landmarks.landmark[t].y > hand_landmarks.landmark[p].y)
    return closed >= 3

class NPC:
    def __init__(self):
        self.reset()
        self.z = random.uniform(0.1, 0.5) # Start far away

    def reset(self):
        self.lane_offset = random.choice([-0.7, -0.25, 0.25, 0.7])
        self.z = 0.0
        self.color = random.choice([(200, 0, 0), (0, 150, 0), (0, 0, 180), (200, 200, 0)])

    def update(self, player_speed):
        # Relative movement
        # If player is faster than 80, NPC comes closer (z increases)
        # If player is slower than 80, NPC moves away (z decreases)
        rel_speed = (player_speed - NPC_SPEED) / 1000.0
        self.z += rel_speed
        
        if self.z > 1.0: # Passed player
            self.reset()
        elif self.z < -0.2: # Too far ahead
            self.reset()

    def draw(self, surface):
        if 0 < self.z < 1:
            x, y, scale = get_3d_pos(self.lane_offset, self.z)
            w = int(120 * scale)
            h = int(80 * scale)
            # Draw NPC Car
            pygame.draw.rect(surface, BLACK, (x - w//2 - 2, y - h - 2, w + 4, h + 4), border_radius=4)
            pygame.draw.rect(surface, self.color, (x - w//2, y - h, w, h), border_radius=3)
            # Tail lights
            pygame.draw.rect(surface, RED, (x - w//2 + 5, y - 15, 15, 8))
            pygame.draw.rect(surface, RED, (x + w//2 - 20, y - 15, 15, 8))

# --- INITIALIZATION ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pro Racing - Cockpit View")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
clock = pygame.time.Clock()
font_speed = pygame.font.SysFont("Impact", 64)
font_ui = pygame.font.SysFont("Arial", 24, bold=True)

# Game State
player_speed = MIN_SPEED
road_offset = 0.0 # Steering offset
score = 0
distance = 0.0
npcs = [NPC() for _ in range(3)]
wheel_angle = 0

# --- MAIN LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 1. CAMERA & HAND TRACKING
    ret, frame = cap.read()
    accel_active = False
    steer_val = 0
    
    cam_surface = None
    if ret:
        frame = cv2.flip(frame, 1)
        cam_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        cam_frame = apply_dark_filter(cam_frame)

        rgb_frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_wrist, right_wrist = None, None
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                mp_draw.draw_landmarks(cam_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                cx, cy = int(wrist.x * CAM_WIDTH), int(wrist.y * CAM_HEIGHT)
                if label == 'Left': left_wrist = (cx, cy)
                elif label == 'Right': right_wrist = (cx, cy)
                if is_fist(hand_landmarks): accel_active = True

        if left_wrist and right_wrist:
            angle = math.degrees(math.atan2(right_wrist[1] - left_wrist[1], right_wrist[0] - left_wrist[0]))
            steer_val = -angle / 45.0 # Normalized steering
            wheel_angle = -angle
        else:
            wheel_angle *= 0.8 # Return to center

        cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.image.frombuffer(cam_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")

    # 2. GAME LOGIC
    if accel_active:
        player_speed += ACCEL_RATE
    else:
        player_speed -= DECEL_RATE
    
    player_speed = max(MIN_SPEED, min(player_speed, MAX_SPEED))
    road_offset += steer_val * (player_speed / 100.0) * 0.1
    road_offset = max(-0.8, min(0.8, road_offset))
    
    distance += player_speed / 3600.0
    score += int(player_speed / 20)

    # Update NPCs and check collision
    for npc in npcs:
        npc.update(player_speed)
        # Collision detection (if NPC is close and in player's lane)
        if 0.85 < npc.z < 0.95:
            if abs(npc.lane_offset - road_offset) < 0.3:
                player_speed = MIN_SPEED # SLOW DOWN ON CRASH
                npc.z += 0.1 # Push NPC away

    # 3. RENDERING
    # Draw Sky
    pygame.draw.rect(screen, SKY_BLUE, (0, 0, SCREEN_WIDTH, HORIZON_Y))
    # Draw Road (Pseudo-3D)
    points = [
        get_3d_pos(-1.5 - road_offset, 0), # Top Left
        get_3d_pos(1.5 - road_offset, 0),  # Top Right
        get_3d_pos(1.5 - road_offset, 1),  # Bottom Right
        get_3d_pos(-1.5 - road_offset, 1)  # Bottom Left
    ]
    pygame.draw.polygon(screen, GRAY, [(p[0], p[1]) for p in points])
    
    # Road Lines
    for z in [i/10.0 for i in range(11)]:
        z_mod = (z + (distance * 10)) % 1.0
        p1 = get_3d_pos(-0.05 - road_offset, z_mod)
        p2 = get_3d_pos(0.05 - road_offset, z_mod)
        p3 = get_3d_pos(0.05 - road_offset, min(1.0, z_mod + 0.05))
        p4 = get_3d_pos(-0.05 - road_offset, min(1.0, z_mod + 0.05))
        pygame.draw.polygon(screen, WHITE, [(p[0], p[1]) for p in [p1, p2, p3, p4]])

    # Draw NPCs
    for npc in npcs: npc.draw(screen)

    # --- COCKPIT OVERLAY ---
    # Dashboard
    pygame.draw.rect(screen, DASH_BOARD, (0, SCREEN_HEIGHT - 220, SCREEN_WIDTH, 220))
    pygame.draw.polygon(screen, DASH_BOARD, [(0, SCREEN_HEIGHT-220), (SCREEN_WIDTH, SCREEN_HEIGHT-220), 
                                             (SCREEN_WIDTH, SCREEN_HEIGHT-300), (0, SCREEN_HEIGHT-300)])
    
    # Steering Wheel (Dynamic)
    wheel_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
    pygame.draw.circle(screen, DARK_GRAY, wheel_center, 160, 30)
    # Spokes
    for a in [0, 120, 240]:
        rad = math.radians(a + wheel_angle)
        end_x = wheel_center[0] + math.cos(rad) * 140
        end_y = wheel_center[1] + math.sin(rad) * 140
        pygame.draw.line(screen, DARK_GRAY, wheel_center, (end_x, end_y), 25)
    pygame.draw.circle(screen, BLACK, wheel_center, 40) # Center hub

    # Gauges
    pygame.draw.circle(screen, BLACK, (SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT - 100), 70)
    pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT - 100), 70, 2)
    # Speed Needle
    speed_angle = math.radians(135 + (player_speed / MAX_SPEED) * 270)
    nx = SCREEN_WIDTH//2 - 250 + math.cos(speed_angle) * 60
    ny = SCREEN_HEIGHT - 100 + math.sin(speed_angle) * 60
    pygame.draw.line(screen, RED, (SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT - 100), (nx, ny), 4)

    # HUD Text (Racing Limits Style)
    speed_surf = font_speed.render(f"{int(player_speed):03d}", True, YELLOW)
    kmh_surf = font_ui.render("KMH", True, WHITE)
    screen.blit(speed_surf, (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 120))
    screen.blit(kmh_surf, (SCREEN_WIDTH - 110, SCREEN_HEIGHT - 85))
    
    dist_surf = font_ui.render(f"{distance:.2f} KM", True, GREEN)
    screen.blit(dist_surf, (40, 40))

    # Webcam
    if cam_surface:
        pygame.draw.rect(screen, WHITE, (CAM_POS_X-2, CAM_POS_Y-2, CAM_WIDTH+4, CAM_HEIGHT+4), border_radius=10)
        screen.blit(cam_surface, (CAM_POS_X, CAM_POS_Y))

    pygame.display.flip()
    clock.tick(FPS)

cap.release()
pygame.quit()
sys.exit()