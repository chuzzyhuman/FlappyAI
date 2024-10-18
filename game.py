import pygame as pg
import numpy as np

WIDTH, HEIGHT = 600, 600
FPS, GROUND_SPEED = 60, 7
PLAYER_SIZE, GROUND_SIZE = 40, 100
PIPE_WIDTH, PIPE_EDGE, GAP_HEIGHT, GAP_MIN, PIPE_DIST = 5, 0, 100, 75, 75
PSEUDO_RANDOM = True
HEIGHT_RAND_RANGE, DIST_RAND_RANGE = 1, 1
HEIGHT_RANGE = (50, HEIGHT - GROUND_SIZE - 250)
EYES_OPEN = False

BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("FlappyAI")
font = pg.font.Font("font.ttf", 24)
clock = pg.time.Clock()

def pseudo_random(num, min_val, max_val):
    a = 1664525
    c = 1013904223
    m = 2**32
    
    num = (a * num + c) % m
    scaled_num = min_val + (num % (max_val - min_val + 1))
    
    return scaled_num

class Player:
    def __init__(self):
        self.x = 50
        self.y = 200
        self.width = PLAYER_SIZE
        self.height = PLAYER_SIZE
        self.yspeed = 0
        self.dead = False
        
    def reset(self):
        self.x = 50
        self.y = 200
        self.yspeed = 0
        self.dead = False
        
    def jump(self):
        if not self.dead:
            self.yspeed = -8
        
    def move(self, pipe):
        self.y += self.yspeed
        if self.yspeed < 15:
            self.yspeed += 0.5
        if self.y >= HEIGHT - GROUND_SIZE - PLAYER_SIZE:
            self.y = HEIGHT - GROUND_SIZE - PLAYER_SIZE
            self.yspeed = 0
            self.dead = True
            self.x -= GROUND_SPEED
        if self.y < 0:
            self.y = 0
            self.yspeed = 0
            self.dead = True
        if self.x + self.width > pipe.x and self.x < pipe.x + pipe.width:
            if self.y < pipe.y or self.y + self.height > pipe.y + pipe.height:
                self.dead = True
        
    def draw(self):
        pg.draw.rect(screen, WHITE, (self.x - 2, self.y - 2, self.width + 4, self.height + 4))
        pg.draw.rect(screen, BLUE if self.dead else RED, (self.x, self.y, self.width, self.height))
        if EYES_OPEN:
            if self.dead:
                pg.draw.line(screen, BLACK, (self.x + 15, self.y + 10), (self.x + 23, self.y + 18), 5)
                pg.draw.line(screen, BLACK, (self.x + 23, self.y + 10), (self.x + 15, self.y + 18), 5)
                pg.draw.line(screen, BLACK, (self.x + 28, self.y + 10), (self.x + 36, self.y + 18), 5)
                pg.draw.line(screen, BLACK, (self.x + 36, self.y + 10), (self.x + 28, self.y + 18), 5)
            else:
                pg.draw.line(screen, BLACK, (self.x + 32, self.y + 10), (self.x + 32, self.y + 20), 6)
                pg.draw.line(screen, BLACK, (self.x + 19, self.y + 10), (self.x + 19, self.y + 20), 6)
                
class Pipe:
    def __init__(self, x):
        self.x = x
        self.y = 100
        self.width = PIPE_WIDTH
        self.height = GAP_HEIGHT
        
    def move(self):
        self.x -= GROUND_SPEED
        
    def draw(self):
        draw_pipe(self)

def draw_pipe(pipe):
    edge = PIPE_EDGE
    pg.draw.rect(screen, WHITE, (pipe.x + edge - 2, 0, pipe.width - edge*2 + 4, pipe.y - 20))
    pg.draw.rect(screen, WHITE, (pipe.x + edge - 2, pipe.y + pipe.height, pipe.width - edge*2 + 4, HEIGHT - (pipe.y + pipe.height) + 20))
    pg.draw.rect(screen, WHITE, (pipe.x - 2, pipe.y - 22, pipe.width + 4, 24))
    pg.draw.rect(screen, WHITE, (pipe.x - 2, pipe.y + pipe.height - 2, pipe.width + 4, 24))
    pg.draw.rect(screen, (100, 100, 100), (pipe.x + edge, 0, pipe.width - edge*2, pipe.y - 20))
    pg.draw.rect(screen, (100, 100, 100), (pipe.x + edge, pipe.y + pipe.height, pipe.width - edge*2, HEIGHT - (pipe.y + pipe.height) + 20))
    pg.draw.rect(screen, (100, 100, 100), (pipe.x, pipe.y - 20, pipe.width, 20))
    pg.draw.rect(screen, (100, 100, 100), (pipe.x, pipe.y + pipe.height, pipe.width, 20))

player = Player()
pipes = [Pipe(WIDTH)]
pipe_idx = 0

time, pipe_time, run, pause, spece_pressed = 0, 0, True, False, False
speed = [0.1, 1, 100]

while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_p:
                pause = not pause
            if event.key == pg.K_e:
                EYES_OPEN = not EYES_OPEN
            if event.key == pg.K_SPACE:
                spece_pressed = True
        if event.type == pg.KEYUP:
            if event.key == pg.K_SPACE:
                spece_pressed = False
    
    if spece_pressed:
        player.jump()

    if pause:
        pg.draw.rect(screen, WHITE, (10, 10, 10, 30))
        pg.draw.rect(screen, WHITE, (30, 10, 10, 30))
        pg.display.update()
        continue
    
    time += 1
    pipe_time += 1
    
    screen.fill(BLACK)
    for pipe in pipes:
        pipe.move()
        pipe.draw()
    player.move(pipes[pipe_idx])
    player.draw()
    pg.draw.rect(screen, WHITE, (0, HEIGHT - GROUND_SIZE - 2, WIDTH, 100))
    pg.draw.rect(screen, (100, 100, 100), (0, HEIGHT - GROUND_SIZE, WIDTH, 100))
    pg.draw.rect(screen, BLACK, (WIDTH, 0, WIDTH, HEIGHT))
    pg.draw.rect(screen, WHITE, (WIDTH, 0, 3, HEIGHT))
    
    if pipes[pipe_idx].x + PIPE_WIDTH < 0:
        pipes.pop(pipe_idx)
    if pipes[pipe_idx].x + PIPE_WIDTH < 50:
        pipe_idx += 1
    if pipe_time == PIPE_DIST:
        pipes.append(Pipe(WIDTH))
        pipes[-1].height = max(pipes[-2].height-1, GAP_MIN)
        if PSEUDO_RANDOM:
            pipes[-1].y = pseudo_random(pipes[-2].y, HEIGHT_RANGE[0], HEIGHT_RANGE[1]) + np.random.randint(-HEIGHT_RAND_RANGE, HEIGHT_RAND_RANGE+1)
            pipe_time = np.random.randint(-DIST_RAND_RANGE, DIST_RAND_RANGE+1)
        else:
            pipes[-1].y = np.random.randint(HEIGHT_RANGE[0], HEIGHT_RANGE[1]) + np.random.randint(-HEIGHT_RAND_RANGE, HEIGHT_RAND_RANGE+1)
            pipe_time = np.random.randint(-DIST_RAND_RANGE, DIST_RAND_RANGE+1)
    
    pg.display.update()
    
    if player.dead:
        player.reset()
        pipes = [Pipe(WIDTH)]
        pipe_idx = 0
        time = 0
        pipe_time = 0
    
    clock.tick(FPS)
