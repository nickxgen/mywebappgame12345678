import pygame
import sys
import random
import time

# Initialize Pygame
pygame.init()

# Canvas Setup
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("VirtuDrive - Steering Tester (Auto-Restart Enabled)")

# Color Palette
COLOR_GRASS = (34, 139, 34)
COLOR_ROAD = (45, 45, 45)
COLOR_BORDER = (240, 240, 240)
COLOR_LANE = (255, 215, 0)
COLOR_PLAYER = (30, 144, 255)
COLOR_TRAFFIC = (220, 20, 60)
COLOR_TEXT = (255, 255, 255)

# Road Dimensions
ROAD_WIDTH = 460
ROAD_LEFT = (SCREEN_WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH

# Player Settings
car_w, car_h = 44, 80

# Physics Setup
MAX_ROAD_SPEED = 18.0
MIN_ROAD_SPEED = 2.0
ACCEL_RATE = 0.25
DECEL_RATE = 0.35
FRICTION = 0.90

# Traffic Class
class TrafficCar:
    def __init__(self):
        self.w = 44
        self.h = 80
        self.reset()

    def reset(self):
        self.x = random.randint(ROAD_LEFT + 15, ROAD_RIGHT - self.w - 15)
        self.y = random.randint(-600, -100)
        self.speed = random.uniform(2.0, 5.0)

    def update(self, player_road_speed):
        self.y += (player_road_speed - self.speed)
        if self.y > SCREEN_HEIGHT + 100:
            self.reset()

    def draw(self, surface):
        rect = pygame.Rect(self.x, self.y, self.w, self.h)
        pygame.draw.rect(surface, COLOR_TRAFFIC, rect, border_radius=8)

clock = pygame.time.Clock()
font_large = pygame.font.SysFont("Trebuchet MS", 28, bold=True)
font_title = pygame.font.SysFont("Trebuchet MS", 48, bold=True)
font_small = pygame.font.SysFont("Trebuchet MS", 20)

def reset_game():
    """Resets all game parameters to start fresh."""
    global car_x, car_y, vx, vy_road, score, lives, traffic_fleet, game_over, line_y
    car_x = SCREEN_WIDTH // 2 - car_w // 2
    car_y = SCREEN_HEIGHT - 130
    vx = 0.0
    vy_road = 6.0
    score = 0
    lives = 3
    line_y = 0.0
    game_over = False
    traffic_fleet = [TrafficCar() for _ in range(3)]

reset_game()

# --- Main Game Loop ---
running = True
restart_timer_start = 0

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if not game_over:
        # --- GAMEPLAY LOGIC ---
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            vx -= 0.8
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            vx += 0.8

        vx *= FRICTION
        car_x += vx

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            vy_road = min(vy_road + ACCEL_RATE, MAX_ROAD_SPEED)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            vy_road = max(vy_road - DECEL_RATE, MIN_ROAD_SPEED)
        else:
            if vy_road > 8.0:
                vy_road -= 0.1
            elif vy_road < 8.0:
                vy_road += 0.1

        # Boundaries
        if car_x < ROAD_LEFT + 8:
            car_x = ROAD_LEFT + 8
            vx = 0
        elif car_x > ROAD_RIGHT - car_w - 8:
            car_x = ROAD_RIGHT - car_w - 8
            vx = 0

        line_y = (line_y + vy_road) % 60
        score += int(vy_road)

        player_rect = pygame.Rect(car_x, car_y, car_w, car_h)

        # Update Traffic
        for t_car in traffic_fleet:
            t_car.update(vy_road)
            t_rect = pygame.Rect(t_car.x, t_car.y, t_car.w, t_car.h)

            if player_rect.colliderect(t_rect):
                lives -= 1
                t_car.reset()
                if lives <= 0:
                    game_over = True
                    restart_timer_start = time.time()

    else:
        # --- AUTO-RESTART COUNTDOWN ---
        elapsed = time.time() - restart_timer_start
        if elapsed >= 3.0:  # Restarts after 3 seconds
            reset_game()

    # --- RENDERING ---
    screen.fill(COLOR_GRASS)

    # Road Surface
    pygame.draw.rect(screen, COLOR_ROAD, (ROAD_LEFT, 0, ROAD_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, COLOR_BORDER, (ROAD_LEFT - 12, 0, 12, SCREEN_HEIGHT))
    pygame.draw.rect(screen, COLOR_BORDER, (ROAD_RIGHT, 0, 12, SCREEN_HEIGHT))

    lane_x_1 = ROAD_LEFT + ROAD_WIDTH // 3
    lane_x_2 = ROAD_LEFT + (2 * ROAD_WIDTH) // 3

    for y in range(int(-60 + line_y), SCREEN_HEIGHT, 60):
        pygame.draw.rect(screen, COLOR_LANE, (lane_x_1 - 4, y, 8, 30))
        pygame.draw.rect(screen, COLOR_LANE, (lane_x_2 - 4, y, 8, 30))

    # Traffic
    for t_car in traffic_fleet:
        t_car.draw(screen)

    # Player Car
    pygame.draw.rect(screen, COLOR_PLAYER, (car_x, car_y, car_w, car_h), border_radius=8)

    # HUD
    speed_kmh = int(vy_road * 12)
    score_surf = font_large.render(f"SCORE: {score}", True, COLOR_TEXT)
    speed_surf = font_large.render(f"SPEED: {speed_kmh} km/h", True, COLOR_TEXT)
    lives_surf = font_small.render(f"LIVES: {'|' * max(0, lives)}", True, (255, 80, 80))

    pygame.draw.rect(screen, (0, 0, 0, 180), (10, 10, 240, 110), border_radius=6)
    screen.blit(score_surf, (20, 20))
    screen.blit(speed_surf, (20, 55))
    screen.blit(lives_surf, (20, 90))

    # Game Over Overlay Screen
    if game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        remaining = max(1, 3 - int(time.time() - restart_timer_start))
        
        go_title = font_title.render("CRASHED! GAME OVER", True, (255, 50, 50))
        go_score = font_large.render(f"Final Score: {score}", True, COLOR_TEXT)
        go_restart = font_large.render(f"Auto-restarting in {remaining}...", True, COLOR_LANE)

        screen.blit(go_title, (SCREEN_WIDTH // 2 - go_title.get_width() // 2, 220))
        screen.blit(go_score, (SCREEN_WIDTH // 2 - go_score.get_width() // 2, 300))
        screen.blit(go_restart, (SCREEN_WIDTH // 2 - go_restart.get_width() // 2, 360))

    pygame.display.flip()

pygame.quit()
sys.exit()