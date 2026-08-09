import sys
import random
import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize Pygame
pygame.init()

# Canvas Setup
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CV Flappy Bird - Motion Control")

clock = pygame.time.Clock()

# Fonts
font_large = pygame.font.SysFont("Trebuchet MS", 36, bold=True)
font_small = pygame.font.SysFont("Trebuchet MS", 20)

# Colors
COLOR_SKY = (120, 200, 255)
COLOR_PIPE = (30, 200, 50)
COLOR_BIRD = (255, 215, 0)
COLOR_TEXT = (255, 255, 255)

# --- MediaPipe Hands Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Game Variables
bird_x = 180
bird_y = SCREEN_HEIGHT // 2
bird_radius = 20

pipes = []
pipe_width = 70
pipe_gap = 180
pipe_speed = 4
pipe_timer = 0

score = 0
high_score = 0
game_over = False

def create_pipe():
    gap_y = random.randint(120, SCREEN_HEIGHT - 120 - pipe_gap)
    return {
        "x": SCREEN_WIDTH + 10,
        "gap_top": gap_y,
        "gap_bottom": gap_y + pipe_gap,
        "passed": False
    }

def reset_game():
    global bird_y, pipes, score, game_over
    bird_y = SCREEN_HEIGHT // 2
    pipes = [create_pipe()]
    score = 0
    game_over = False

reset_game()

# --- Main Engine Loop ---
running = True
while running:
    dt = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game_over:
                reset_game()
            elif event.key == pygame.K_ESCAPE:
                running = False

    # 1. OpenCV WebCam & MediaPipe Processing
    ret, frame = cap.read()
    cam_surface = None

    if ret:
        frame = cv2.flip(frame, 1)  # Mirror feed
        h_cam, w_cam, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract Wrist & Middle MCP to calculate palm center
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                # Normalized Palm Y position (0.0 at top, 1.0 at bottom)
                palm_y_norm = (wrist.y + mcp.y) / 2.0

                # Directly map hand altitude to bird height
                if not game_over:
                    target_y = int(palm_y_norm * SCREEN_HEIGHT)
                    # Smooth interpolation (EMA) to reduce tracking jitter
                    bird_y += (target_y - bird_y) * 0.25

        # Convert OpenCV Frame to Pygame Surface for Picture-in-Picture
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        py_img = pygame.image.frombuffer(frame_rgb.tobytes(), (w_cam, h_cam), "RGB")
        cam_surface = pygame.transform.scale(py_img, (180, 135))

    # 2. Game Mechanics Update
    if not game_over:
        # Spawn Pipes
        pipe_timer += 1
        if pipe_timer >= 90:
            pipes.append(create_pipe())
            pipe_timer = 0

        # Update Pipes
        bird_rect = pygame.Rect(bird_x - bird_radius, bird_y - bird_radius, bird_radius * 2, bird_radius * 2)

        for p in pipes[:]:
            p["x"] -= pipe_speed

            # Score Increment
            if not p["passed"] and p["x"] < bird_x:
                score += 1
                p["passed"] = True

            # Collision Bounds
            top_pipe = pygame.Rect(p["x"], 0, pipe_width, p["gap_top"])
            bottom_pipe = pygame.Rect(p["x"], p["gap_bottom"], pipe_width, SCREEN_HEIGHT - p["gap_bottom"])

            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                game_over = True

            # Remove Off-Screen Pipes
            if p["x"] < -pipe_width:
                pipes.remove(p)

        # Boundary Collisions (Screen Top/Bottom)
        if bird_y - bird_radius <= 0 or bird_y + bird_radius >= SCREEN_HEIGHT:
            game_over = True

        if score > high_score:
            high_score = score

    # 3. Render Graphics
    screen.fill(COLOR_SKY)

    # Draw Pipes
    for p in pipes:
        pygame.draw.rect(screen, COLOR_PIPE, (p["x"], 0, pipe_width, p["gap_top"]))
        pygame.draw.rect(screen, COLOR_PIPE, (p["x"], p["gap_bottom"], pipe_width, SCREEN_HEIGHT - p["gap_bottom"]))

    # Draw Bird
    pygame.draw.circle(screen, COLOR_BIRD, (int(bird_x), int(bird_y)), bird_radius)
    pygame.draw.circle(screen, (255, 255, 255), (int(bird_x + 8), int(bird_y - 6)), 5)  # Eye
    pygame.draw.circle(screen, (0, 0, 0), (int(bird_x + 10), int(bird_y - 6)), 2)      # Pupil
    pygame.draw.polygon(screen, (255, 100, 0), [(bird_x + 15, bird_y), (bird_x + 26, bird_y + 4), (bird_x + 15, bird_y + 8)]) # Beak

    # Render Camera Feed Overlay (Top Right)
    if cam_surface:
        pip_x, pip_y = SCREEN_WIDTH - 200, 15
        pygame.draw.rect(screen, (255, 255, 255), (pip_x - 3, pip_y - 3, 186, 141), 2)
        screen.blit(cam_surface, (pip_x, pip_y))

    # Render HUD
    score_surf = font_large.render(f"Score: {score}", True, COLOR_TEXT)
    high_surf = font_small.render(f"Best: {high_score}", True, COLOR_TEXT)
    screen.blit(score_surf, (20, 20))
    screen.blit(high_surf, (20, 65))

    # Game Over Screen Overlay
    if game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        go_text = font_large.render("GAME OVER", True, (255, 60, 60))
        reset_text = font_small.render("Lower/Raise hand & Press 'R' to Restart", True, COLOR_TEXT)

        screen.blit(go_text, (SCREEN_WIDTH // 2 - go_text.get_width() // 2, SCREEN_HEIGHT // 2 - 40))
        screen.blit(reset_text, (SCREEN_WIDTH // 2 - reset_text.get_width() // 2, SCREEN_HEIGHT // 2 + 20))

    pygame.display.flip()

cap.release()
pygame.quit()
sys.exit()