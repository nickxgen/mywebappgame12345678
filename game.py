import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Virtual Steering - 2D Car Game")

# Colors
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
YELLOW = (255, 215, 0)
RED = (220, 50, 50)
BLUE = (50, 120, 220)
GREEN = (50, 200, 50)

# Road settings
ROAD_WIDTH = 400
ROAD_LEFT = (SCREEN_WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH

# Player Car settings
car_width = 40
car_height = 70
car_x = SCREEN_WIDTH // 2 - car_width // 2
car_y = SCREEN_HEIGHT - 120
car_speed = 6

# Game variables
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)
large_font = pygame.font.SysFont("Arial", 48)

score = 0
game_over = False

# Road line animation
line_y = 0
road_speed = 7

# Obstacles setup
obstacles = []

def spawn_obstacle():
    obs_w = 40
    obs_h = 70
    obs_x = random.randint(ROAD_LEFT + 10, ROAD_RIGHT - obs_w - 10)
    obs_y = -obs_h
    return [obs_x, obs_y, obs_w, obs_h]

# Spawn initial obstacle
obstacles.append(spawn_obstacle())

# --- Game Loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # Press R to restart after Game Over
        if event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_r:
                car_x = SCREEN_WIDTH // 2 - car_width // 2
                obstacles = [spawn_obstacle()]
                score = 0
                game_over = False

    if not game_over:
        # Read keys (works with physical keyboard OR pynput WASD triggers)
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            car_x -= car_speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            car_x += car_speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            road_speed = 12  # Accelerate road speed
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            road_speed = 3   # Brake/slow down road
        else:
            road_speed = 7   # Normal speed

        # Keep car inside road boundaries
        if car_x < ROAD_LEFT + 5:
            car_x = ROAD_LEFT + 5
        if car_x > ROAD_RIGHT - car_width - 5:
            car_x = ROAD_RIGHT - car_width - 5

        # Move road center lines
        line_y += road_speed
        if line_y >= 40:
            line_y = 0

        # Update obstacles
        for obs in obstacles:
            obs[1] += road_speed

        # Spawn new obstacle
        if obstacles[-1][1] > 200:
            obstacles.append(spawn_obstacle())

        # Remove off-screen obstacles
        if obstacles[0][1] > SCREEN_HEIGHT:
            obstacles.pop(0)
            score += 10

        # Collision detection
        player_rect = pygame.Rect(car_x, car_y, car_width, car_height)
        for obs in obstacles:
            obs_rect = pygame.Rect(obs[0], obs[1], obs[2], obs[3])
            if player_rect.colliderect(obs_rect):
                game_over = True

    # --- Drawing ---
    screen.fill(GREEN)  # Grass background

    # Draw Road
    pygame.draw.rect(screen, GRAY, (ROAD_LEFT, 0, ROAD_WIDTH, SCREEN_HEIGHT))
    # Draw Road Borders
    pygame.draw.rect(screen, WHITE, (ROAD_LEFT - 10, 0, 10, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, (ROAD_RIGHT, 0, 10, SCREEN_HEIGHT))

    # Draw Center Dashed Line
    for i in range(-40, SCREEN_HEIGHT, 40):
        pygame.draw.rect(screen, YELLOW, (SCREEN_WIDTH // 2 - 5, i + line_y, 10, 20))

    # Draw Obstacles (Red Cars)
    for obs in obstacles:
        pygame.draw.rect(screen, RED, (obs[0], obs[1], obs[2], obs[3]), border_radius=6)

    # Draw Player Car (Blue Car)
    pygame.draw.rect(screen, BLUE, (car_x, car_y, car_width, car_height), border_radius=6)

    # Draw Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (20, 20))

    # Game Over Overlay
    if game_over:
        over_text = large_font.render("GAME OVER", True, RED)
        restart_text = font.render("Press 'R' to Restart", True, WHITE)
        screen.blit(over_text, (SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 - 40))
        screen.blit(restart_text, (SCREEN_WIDTH // 2 - 90, SCREEN_HEIGHT // 2 + 20))

    pygame.display.flip()
    clock.tick(60)