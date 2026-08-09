import cv2
import math
import sys
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe as mp

# --- CONFIGURATION & CONSTANTS ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30

CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_POS_X = SCREEN_WIDTH - CAM_WIDTH - 20
CAM_POS_Y = 20

# Gear Config (Limits & Acceleration Rates)
GEARS = ["R", "N", "1", "2", "3"]
GEAR_LIMITS = {"R": -8.0, "N": 0.0, "1": 6.0, "2": 14.0, "3": 25.0}
ACCEL_RATES = {"R": -0.10, "N": 0.0, "1": 0.20, "2": 0.15, "3": 0.10}

# --- FILTERS ---

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

# --- GESTURES ---

def is_hand_open(landmarks):
    wrist = landmarks.landmark[0]
    tips = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]
    open_fingers = 0
    for tip, base in zip(tips, bases):
        tip_lm = landmarks.landmark[tip]
        base_lm = landmarks.landmark[base]
        tip_dist = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
        base_dist = math.hypot(base_lm.x - wrist.x, base_lm.y - wrist.y)
        if tip_dist > base_dist * 1.3:
            open_fingers += 1
    return open_fingers >= 3

# --- OPENGL TEXTURE HELPERS ---

def surface_to_texture(surface):
    """Converts Pygame surface or Raw Image to an OpenGL Texture ID."""
    rgb_data = pygame.image.tostring(surface, "RGB", True)
    width, height = surface.get_rect().size

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_data)
    return texture_id, width, height

def draw_textured_quad(texture_id, x, y, width, height):
    """Renders a 2D OpenGL textured quad on screen coordinates."""
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glColor3f(1.0, 1.0, 1.0)
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x, y)
    glTexCoord2f(1, 1); glVertex2f(x + width, y)
    glTexCoord2f(1, 0); glVertex2f(x + width, y + height)
    glTexCoord2f(0, 0); glVertex2f(x, y + height)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glDeleteTextures(1, [texture_id])

# --- 3D OPENGL GEOMETRY ---

def draw_cube(size_x, size_y, size_z, color):
    x, y, z = size_x / 2.0, size_y / 2.0, size_z / 2.0
    vertices = [
        [x, y, -z], [x, -y, -z], [-x, -y, -z], [-x, y, -z],
        [x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z]
    ]
    surfaces = [
        (0,1,2,3), (4,5,6,7), (0,4,7,3),
        (1,5,6,2), (0,1,5,4), (3,2,6,7)
    ]
    glColor3f(*color)
    glBegin(GL_QUADS)
    for surface in surfaces:
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_car():
    # Lower Chassis
    glPushMatrix()
    glTranslatef(0, 0.4, 0)
    draw_cube(1.8, 0.5, 4.0, (0.8, 0.1, 0.1))
    glPopMatrix()

    # Roof Cabin
    glPushMatrix()
    glTranslatef(0, 0.85, -0.2)
    draw_cube(1.4, 0.45, 2.0, (0.2, 0.2, 0.2))
    glPopMatrix()

    # Headlights
    glPushMatrix()
    glTranslatef(0.6, 0.4, -2.0)
    draw_cube(0.3, 0.15, 0.1, (1.0, 1.0, 0.8))
    glTranslatef(-1.2, 0, 0)
    draw_cube(0.3, 0.15, 0.1, (1.0, 1.0, 0.8))
    glPopMatrix()

def draw_world(z_offset):
    # Ground
    glColor3f(0.15, 0.45, 0.15)
    glBegin(GL_QUADS)
    glVertex3f(-100, 0, -200)
    glVertex3f(100, 0, -200)
    glVertex3f(100, 0, 100)
    glVertex3f(-100, 0, 100)
    glEnd()

    # Asphalt Road
    glColor3f(0.2, 0.2, 0.2)
    glBegin(GL_QUADS)
    glVertex3f(-6, 0.01, -200)
    glVertex3f(6, 0.01, -200)
    glVertex3f(6, 0.01, 100)
    glVertex3f(-6, 0.01, 100)
    glEnd()

    # Road Stripes
    glColor3f(1.0, 0.85, 0.0)
    for z in range(-200, 100, 12):
        sz = z + (z_offset % 12)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.02, sz)
        glVertex3f(0.15, 0.02, sz)
        glVertex3f(0.15, 0.02, sz + 5)
        glVertex3f(-0.15, 0.02, sz + 5)
        glEnd()

    # Trees
    for z in range(-180, 80, 20):
        for side in [-9, 9]:
            glPushMatrix()
            glTranslatef(side, 0, z)
            draw_cube(0.4, 2.0, 0.4, (0.4, 0.2, 0.1))
            glTranslatef(0, 1.8, 0)
            draw_cube(1.8, 1.8, 1.8, (0.1, 0.5, 0.1))
            glPopMatrix()

# --- INITIALIZATION ---

pygame.init()
pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Supercar - OpenGL Fixed Render Loop")

glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
gluPerspective(60, (SCREEN_WIDTH / SCREEN_HEIGHT), 0.1, 300.0)

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

font_sub = pygame.font.SysFont("Arial", 18, bold=True)
clock = pygame.time.Clock()

# Physics & Game State
car_x = 0.0
current_speed = 0.0
gear_idx = 2  # 1st Gear
steer_angle = 0.0
world_z = 0.0
current_filter = 0
cam_orbit_angle = 0.0

last_right_open, last_left_open = False, False
right_shift_cooldown, left_shift_cooldown = 0, 0

# --- MAIN LOOP ---

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: current_filter = 1
            elif event.key == pygame.K_2: current_filter = 2
            elif event.key == pygame.K_3: current_filter = 3
            elif event.key == pygame.K_4: current_filter = 4
            elif event.key == pygame.K_0: current_filter = 0
            elif event.key == pygame.K_ESCAPE: running = False

    # 1. CAMERA & GESTURE PROCESSING
    ret, frame = cap.read()
    cam_surface = None
    right_open, left_open = False, False
    found_right, found_left = False, False
    hand_steer_target = 0.0

    if right_shift_cooldown > 0: right_shift_cooldown -= 1
    if left_shift_cooldown > 0: left_shift_cooldown -= 1

    if ret:
        frame = cv2.flip(frame, 1)
        cam_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        if current_filter == 1: cam_frame = apply_beauty_filter(cam_frame)
        elif current_filter == 2: cam_frame = apply_dark_noir_filter(cam_frame)
        elif current_filter == 3: cam_frame = apply_deep_dark_filter(cam_frame)
        elif current_filter == 4: cam_frame = apply_midnight_filter(cam_frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_wrist_pt, right_wrist_pt = None, None

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                mp_draw.draw_landmarks(cam_frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                is_open = is_hand_open(landmarks)
                wrist = landmarks.landmark[0]
                cx, cy = int(wrist.x * CAM_WIDTH), int(wrist.y * CAM_HEIGHT)

                if label == 'Right':
                    found_right, right_open = True, is_open
                    right_wrist_pt = (cx, cy)
                    if last_right_open and not right_open and right_shift_cooldown == 0:
                        if gear_idx < len(GEARS) - 1:
                            gear_idx += 1
                            right_shift_cooldown = 10
                    last_right_open = right_open

                elif label == 'Left':
                    found_left, left_open = True, is_open
                    left_wrist_pt = (cx, cy)
                    if last_left_open and not left_open and left_shift_cooldown == 0:
                        if gear_idx > 0:
                            gear_idx -= 1
                            left_shift_cooldown = 10
                    last_left_open = left_open

        if left_wrist_pt and right_wrist_pt:
            lx, ly = left_wrist_pt
            rx, ry = right_wrist_pt
            angle_rad = math.atan2(ry - ly, rx - lx)
            hand_steer_target = np.clip(math.degrees(angle_rad) * 1.5, -45, 45)

        cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.image.frombuffer(cam_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")

    # 2. PHYSICS & CONTROLS
    current_gear_str = GEARS[gear_idx]
    max_limit = GEAR_LIMITS[current_gear_str]
    accel_rate = ACCEL_RATES[current_gear_str]

    if right_open and left_open:
        if current_speed > 0: current_speed = max(0.0, current_speed - 0.4)
        elif current_speed < 0: current_speed = min(0.0, current_speed + 0.4)
    elif (found_right and not right_open) and (found_left and not left_open):
        if current_gear_str == "R":
            if current_speed > max_limit: current_speed += accel_rate
        elif current_gear_str != "N":
            if current_speed < max_limit: current_speed += accel_rate
    else:
        if current_speed > 0: current_speed = max(0.0, current_speed - 0.1)
        elif current_speed < 0: current_speed = min(0.0, current_speed + 0.1)

    steer_angle += (hand_steer_target - steer_angle) * 0.1
    car_x += (steer_angle * 0.003) * (abs(current_speed) / 5.0)
    car_x = np.clip(car_x, -5.0, 5.0)

    world_z += current_speed * 0.2

    # 3. CAMERA CINEMATICS
    target_orbit = 90.0 if current_gear_str == "R" else 0.0
    cam_orbit_angle += (target_orbit - cam_orbit_angle) * 0.08

    cam_rad = math.radians(cam_orbit_angle)
    cam_dist = 9.0
    cam_x = car_x + cam_dist * math.sin(cam_rad)
    cam_z = 5.0 + cam_dist * (1.0 - math.cos(cam_rad))
    cam_y = 3.5

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(cam_x, cam_y, cam_z, car_x, 0.8, -3.0, 0, 1, 0)

    # 4. RENDER 3D SCENE
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_world(world_z)

    glPushMatrix()
    glTranslatef(car_x, 0, 0)
    glRotatef(-steer_angle * 0.4, 0, 1, 0)
    draw_car()
    glPopMatrix()

    # 5. RENDER 2D OVERLAYS (HUD & WEBCAM)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)

    # Draw Webcam Texture Overlay
    if cam_surface:
        tex_id, tw, th = surface_to_texture(cam_surface)
        draw_textured_quad(tex_id, CAM_POS_X, CAM_POS_Y, tw, th)

    # Render Telemetry Text
    hud_surface = pygame.Surface((320, 200), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 180))

    r_state = "OPEN (Brake)" if right_open else "CLOSED (Accel)"
    l_state = "OPEN (Brake)" if left_open else "CLOSED (Accel)"

    hud_lines = [
        (f"SPEED: {int(abs(current_speed) * 10)} km/h", (255, 215, 0)),
        (f"GEAR: [ {current_gear_str} ]", (0, 255, 0) if current_gear_str != "R" else (255, 50, 50)),
        (f"CAM ANGLE: {int(cam_orbit_angle)} deg", (0, 255, 255)),
        (f"STEER: {int(steer_angle)} deg", (255, 255, 255)),
        (f"RIGHT HAND: {r_state}", (255, 80, 80) if right_open else (80, 255, 80)),
        (f"LEFT HAND: {l_state}", (255, 80, 80) if left_open else (80, 255, 80)),
        (f"FILTER [0-4]: {current_filter}", (255, 255, 255))
    ]

    for idx, (text, color) in enumerate(hud_lines):
        txt_surf = font_sub.render(text, True, color)
        hud_surface.blit(txt_surf, (15, 10 + idx * 24))

    hud_tex, hw, hh = surface_to_texture(hud_surface)
    draw_textured_quad(hud_tex, 20, 20, hw, hh)

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    pygame.display.flip()
    clock.tick(FPS)

cap.release()
pygame.quit()
sys.exit()