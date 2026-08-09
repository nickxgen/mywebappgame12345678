import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Virtual Steering - Testing Track (No Obstacles)")

# Colors
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
YELLOW = (255, 215, 0)
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

score = 0

# Road line animation
line_y = 0
road_speed = 7

# --- Game Loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Read keys (works with physical keyboard OR pynput WASD triggers)
    keys = pygame.key.get_pressed()

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        car_x -= car_speed
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        car_x += car_speed
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        road_speed = 14  # Accelerate road speed
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        road_speed = 3   # Slow down road
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

    # Accumulate score continuously as you drive
    score += int(road_speed / 2)

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

    # Draw Player Car (Blue Car)
    pygame.draw.rect(screen, BLUE, (car_x, car_y, car_width, car_height), border_radius=6)

    # Draw Live Data Overlay
    score_text = font.render(f"Distance Score: {score}", True, WHITE)
    speed_text = font.render(f"Speed: {road_speed * 10} km/h", True, WHITE)
    screen.blit(score_text, (20, 20))
    screen.blit(speed_text, (20, 50))

    pygame.display.flip()
    clock.tick(60)