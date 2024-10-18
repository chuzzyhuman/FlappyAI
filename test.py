import pygame as pg
import numpy as np

WIDTH, HEIGHT = 600, 600
SCREEN_WIDTH = 1200
FPS, GROUND_SPEED = 60, 7
PLAYER_SIZE, GROUND_SIZE = 40, 100
PLAYER_X, PLAYER_Y = 50, 200
MAX_YSPEED = 15
PIPE_WIDTH, PIPE_EDGE, GAP_MAX, GAP_MIN, PIPE_DIST = 5, 0, 100, 75, 70
PSEUDO_RANDOM = True
HEIGHT_RAND_RANGE, DIST_RAND_RANGE = 1, 1
HEIGHT_RANGE = (50, HEIGHT - GROUND_SIZE - 250)
EYES_OPEN = False
SHOW_TEXT, GRAPH_LOG, GRAPH_NUM = True, False, 0
SAVE_MODE = False

BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)
COLOR_LIST = [(i, 255, 0) for i in range(0, 256, 51)] + [(255, i, 0) for i in range(255, -1, -51)] + [(i, 0, 0) for i in range(255, -1, -51)] + [(0, 0, 0) for i in range(100)]

INPUTS, OUTPUTS = 4, 1
POPULATION = 10000

MAX_WEIGHT = 5
DELTA_THRESHOLD = 0.5
ACTIVATION_MODE = 2
ACTIVATION_THRESHOLD = [0, 0.5, 0, 0.5]

pg.init()
screen = pg.display.set_mode((SCREEN_WIDTH, HEIGHT))
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

def squash(x, n):
    if n == 0:
        return x
    if n == 1:
        return 1/(1 + np.exp(-x))
    if n == 2:
        return np.tanh(x)
    if n == 3:
        return max(0, x)
    
def log(x, n):
    if n:
        return np.log10(1+x)
    return x

class Genome:
    def __init__(self):
        self.weight1 = np.zeros((3, 5))
        self.weight2 = np.zeros((3, 4))
        self.weight3 = np.zeros((1, 4))
        self.value1 = np.array([0, 0, 0, 0, 1], dtype=np.float64)
        self.value2 = np.array([0, 0, 0, 1], dtype=np.float64)
        self.value3 = np.array([0, 0, 0, 1], dtype=np.float64)
        self.value4 = np.array([0], dtype=np.float64)
        self.fitness = 0
        self.avg_fitness = 0
        self.score = 0
        self.avg_score = 0
        self.species = 0
        self.reload()

    def clone(self):
        g = Genome()
        g.weight1 = self.weight1.copy()
        g.weight2 = self.weight2.copy()
        g.weight3 = self.weight3.copy()
        g.fitness = self.fitness
        g.avg_fitness = self.avg_fitness
        g.score = self.score
        g.avg_score = self.avg_score
        g.species = self.species
        g.reload()
        return g

    def reload(self):
        self.value1 = np.array([0, 0, 0, 0, 1], dtype=np.float64)
        self.value2 = np.array([0, 0, 0, 1], dtype=np.float64)
        self.value3 = np.array([0, 0, 0, 1], dtype=np.float64)
        self.value4 = np.array([0], dtype=np.float64)

    def distance(self, other):
        weight_diff = 0
        for i in range(3):
            for j in range(5):
                weight_diff += abs(self.weight1[i, j] - other.weight1[i, j]) ** 2
        for i in range(3):
            for j in range(4):
                weight_diff += abs(self.weight2[i, j] - other.weight2[i, j]) ** 2
        for i in range(1):
            for j in range(4):
                weight_diff += abs(self.weight3[i, j] - other.weight3[i, j]) ** 2
        return weight_diff / (3*5 + 3*4 + 1*4)

    def feed_forward(self, inputs):
        self.value1[:4] = inputs
        self.value2 = squash(np.dot(self.weight1, self.value1), ACTIVATION_MODE)
        self.value2 = np.append(self.value2, 1)
        self.value3 = squash(np.dot(self.weight2, self.value2), ACTIVATION_MODE)
        self.value3 = np.append(self.value3, 1)
        self.value4 = squash(np.dot(self.weight3, self.value3), ACTIVATION_MODE)
        return self.value4

    def change_weight(self):
        r = np.random.rand()
        if r < 1/6:
            self.weight1[np.random.randint(3), np.random.randint(5)] = np.random.randn()
            self.weight1 = np.clip(self.weight1, -MAX_WEIGHT, MAX_WEIGHT)
        elif r < 2/6:
            self.weight2[np.random.randint(3), np.random.randint(4)] = np.random.randn()
            self.weight2 = np.clip(self.weight2, -MAX_WEIGHT, MAX_WEIGHT)
        elif r < 3/6:
            self.weight3[np.random.randint(1), np.random.randint(4)] = np.random.randn()
            self.weight3 = np.clip(self.weight3, -MAX_WEIGHT, MAX_WEIGHT)
        elif r < 4/6:
            self.weight1[np.random.randint(3), np.random.randint(5)] += np.random.randn()
            self.weight1 = np.clip(self.weight1, -MAX_WEIGHT, MAX_WEIGHT)
        elif r < 5/6:
            self.weight2[np.random.randint(3), np.random.randint(4)] += np.random.randn()
            self.weight2 = np.clip(self.weight2, -MAX_WEIGHT, MAX_WEIGHT)
        else:
            self.weight3[np.random.randint(1), np.random.randint(4)] += np.random.randn()
            self.weight3 = np.clip(self.weight3, -MAX_WEIGHT, MAX_WEIGHT)

    def mutate(self):
        self.change_weight()

    def crossover(self, other):
        child = self.clone()
        for i in range(3):
            for j in range(5):
                if np.random.rand() < 0.5:
                    child.weight1[i, j] = other.weight1[i, j]
        for i in range(3):
            for j in range(4):
                if np.random.rand() < 0.5:
                    child.weight2[i, j] = other.weight2[i, j]
        for i in range(1):
            for j in range(4):
                if np.random.rand() < 0.5:
                    child.weight3[i, j] = other.weight3[i, j]
        return child
    
class Player:
    def __init__(self, genome=None):
        self.x = PLAYER_X
        self.y = PLAYER_Y
        self.yspeed = 0
        self.dead = False
        self.input = []
        self.genome = genome if genome else Genome()
        
    def reset(self):
        self.x = PLAYER_X
        self.y = PLAYER_Y
        self.yspeed = 0
        self.dead = False
        self.genome.fitness = 0
        self.genome.score = 0
        
    def feed_forward(self):
        outputs = self.genome.feed_forward(self.input)
        if OUTPUTS == 1:
            if outputs[0] > ACTIVATION_THRESHOLD[ACTIVATION_MODE]:
                self.jump()
        elif OUTPUTS == 2:
            if outputs[0] >= outputs[1]:
                self.jump()
        
    def jump(self):
        if not self.dead:
            self.yspeed = -8
        
    def move(self, pipe):
        self.y += self.yspeed
        self.input = [(pipe.x - self.x - PLAYER_SIZE) / WIDTH,
                      #(pipe.y - self.y) / PLAYER_SIZE,
                      #(pipe.y - self.y + pipe.height - PLAYER_SIZE) / PLAYER_SIZE,
                      (pipe.y - self.y + (pipe.height - PLAYER_SIZE) / 2) / PLAYER_SIZE,
                      (HEIGHT - GROUND_SIZE - self.y - PLAYER_SIZE) / HEIGHT, 
                      self.yspeed / MAX_YSPEED]
        if self.yspeed < MAX_YSPEED:
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
        if self.x + PLAYER_SIZE > pipe.x and self.x < pipe.x + pipe.width:
            if self.y < pipe.y or self.y + PLAYER_SIZE > pipe.y + pipe.height:
                self.dead = True
        if self.dead:
            death_score[gen-1][score] = death_score[gen-1].get(score, 0) + 1
        else:
            self.genome.score += 0.01
        
    def draw(self):
        pg.draw.rect(screen, WHITE, (self.x - 2, self.y - 2, PLAYER_SIZE + 4, PLAYER_SIZE + 4))
        pg.draw.rect(screen, BLUE if self.dead else COLOR_LIST[self.genome.species], (self.x, self.y, PLAYER_SIZE, PLAYER_SIZE))
        if EYES_OPEN:
            if self.dead:
                pg.draw.line(screen, BLACK, (self.x + 15, self.y + 10), (self.x + 23, self.y + 18), 5)
                pg.draw.line(screen, BLACK, (self.x + 23, self.y + 10), (self.x + 15, self.y + 18), 5)
                pg.draw.line(screen, BLACK, (self.x + 28, self.y + 10), (self.x + 36, self.y + 18), 5)
                pg.draw.line(screen, BLACK, (self.x + 36, self.y + 10), (self.x + 28, self.y + 18), 5)
            else:
                pg.draw.line(screen, BLACK, (self.x + 32, self.y + 10), (self.x + 32, self.y + 20), 6)
                pg.draw.line(screen, BLACK, (self.x + 19, self.y + 10), (self.x + 19, self.y + 20), 6)
        if SHOW_TEXT:
            text = font.render(f"{self.genome.avg_fitness:.2f}", True, WHITE)
            screen.blit(text, (self.x, self.y - 25))
                
class Pipe:
    def __init__(self, x):
        self.x = x
        self.y = HEIGHT_RANGE[0] + 20
        self.width = PIPE_WIDTH
        self.height = GAP_MAX
        
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

def speciate(population):
    species = []
    for genome in population:
        for s in species:
            if genome.distance(s[0]) < DELTA_THRESHOLD:
                s.append(genome)
                break
        else:
            if len(species) < 50:
                species.append([genome])
            else:
                species.sort(key=lambda x: max([g.avg_fitness for g in x]), reverse=True)
                species[-1].append(genome)
    species.sort(key=lambda x: max([g.avg_fitness for g in x]), reverse=True)
    for s in species:
        s.sort(key=lambda x: x.avg_fitness, reverse=True)
    for i, s in enumerate(species):
        for g in s:
            g.species = i
    return species

def reproduce(population):
    global species
    population.sort(key=lambda x: x.genome.avg_fitness, reverse=True)
    species = speciate([player.genome for player in population])
    new_population = [population[i].genome for i in range(POPULATION//100)]
    for i in range(len(species)//2+1):
        n = 0
        if i <= 5:
            n = 50
        elif i <= 10:
            n = 20
        elif i <= 15:
            n = 5
        s = species[i]
        for _ in range(POPULATION//20+n):
            j = min(int(abs(np.random.randn())*2), 5, len(s)-1)
            child = s[j].clone()
            for _ in range(int(abs(np.random.randn())*2)+1):
                child.mutate()
            child.avg_fitness = s[j].avg_fitness
            child.avg_score = s[j].avg_score
            new_population.append(child)
    for s in species:
        for i in range(min(len(s), 30)):
            parent1 = np.random.choice(s[:min(len(s)//4+1, 5)])
            parent2 = np.random.choice(s[:min(len(s)//4+1, 5)])
            child = parent1.crossover(parent2)
            if parent1.avg_fitness > parent2.avg_fitness:
                child.avg_fitness = parent1.avg_fitness
                child.avg_score = parent1.avg_score
            else:
                child.avg_fitness = parent2.avg_fitness
                child.avg_score = parent2.avg_score
            new_population.append(child)
    for p in sorted(population, key=lambda x: x.genome.fitness, reverse=True):
        new_population.append(p.genome.clone())
    return [Player(genome) for genome in new_population[:POPULATION]]

def draw_stats():
    text = font.render("Slow" if speed[speed_idx] == 0.1 else "" if speed[speed_idx] == 1 else "Fast", True, WHITE)
    screen.blit(text, (WIDTH - 55, 10))
    text = font.render(f"Generation: {gen}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 10))
    text = font.render(f"Score:", True, WHITE)
    screen.blit(text, (WIDTH + 20, 40))
    text = font.render(f"{(score/100):.2f}", True, WHITE if max(best_score + best_avg_score) >= score/100 else GREEN)
    screen.blit(text, (WIDTH + 85, 40))
    text = font.render(f"Alive: {len(population)}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 70))
    text = font.render(f"Species: {[len(s) for s in species]}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 100))
    
    top, bottom = 140, 300
    w, h, g1, g2 = 90, 340, 80, 45
    min_value, max_value = -1, 1
    best = max(population + dead_population, key=lambda x: x.genome.score).genome
    
    if SHOW_TEXT:
        for i, value in enumerate(max(population + dead_population, key=lambda x: x.genome.score).input):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 20, h - 12 + i*g2))
        text = font.render(f"{max(population + dead_population, key=lambda x: x.genome.score).genome.value4[0]:.2f}", True, WHITE)
        screen.blit(text, (WIDTH + 100 + 3*g1 + 20, h - 12))
    
    #draw weight1
    for i in range(3):
        for j in range(5):
            pos1 = (WIDTH + w + 0*g1, h + j*g2)
            pos2 = (WIDTH + w + 1*g1, h + i*g2)
            pg.draw.line(screen, RED if best.weight1[i, j] > 0 else BLUE if best.weight1[i, j] < 0 else (50, 50, 50), pos1, pos2, abs(int(best.weight1[i, j])) + 2)
    #draw weight2
    for i in range(3):
        for j in range(4):
            pos1 = (WIDTH + w + 1*g1, h + j*g2)
            pos2 = (WIDTH + w + 2*g1, h + i*g2)
            pg.draw.line(screen, RED if best.weight2[i, j] > 0 else BLUE if best.weight2[i, j] < 0 else (50, 50, 50), pos1, pos2, abs(int(best.weight2[i, j])) + 2)
    #draw weight3
    for i in range(1):
        for j in range(4):
            pos1 = (WIDTH + w + 2*g1, h + j*g2)
            pos2 = (WIDTH + w + 3*g1, h + i*g2)
            pg.draw.line(screen, RED if best.weight3[i, j] > 0 else BLUE if best.weight3[i, j] < 0 else (50, 50, 50), pos1, pos2, abs(int(best.weight3[i, j])) + 2)

    #draw node values
    for i, value in enumerate(best.value1):
        b = int(255 * (value - min_value) / (max_value - min_value))
        b = max(0, min(255, b))
        pg.draw.circle(screen, (b, b, b), (WIDTH + w + 0*g1, h + i*g2), 20)
        pg.draw.circle(screen, WHITE, (WIDTH + w + 0*g1, h + i*g2), 20, 3)
    for i, value in enumerate(best.value2):
        b = int(255 * (value - min_value) / (max_value - min_value))
        b = max(0, min(255, b))
        pg.draw.circle(screen, (b, b, b), (WIDTH + w + 1*g1, h + i*g2), 20)
        pg.draw.circle(screen, WHITE, (WIDTH + w + 1*g1, h + i*g2), 20, 3)
    for i, value in enumerate(best.value3):
        b = int(255 * (value - min_value) / (max_value - min_value))
        b = max(0, min(255, b))
        pg.draw.circle(screen, (b, b, b), (WIDTH + w + 2*g1, h + i*g2), 20)
        pg.draw.circle(screen, WHITE, (WIDTH + w + 2*g1, h + i*g2), 20, 3)
    for i, value in enumerate(best.value4):
        b = int(255 * (value - min_value) / (max_value - min_value))
        b = max(0, min(255, b))
        pg.draw.circle(screen, (b, b, b), (WIDTH + w + 3*g1, h + i*g2), 20)
        pg.draw.circle(screen, WHITE, (WIDTH + w + 3*g1, h + i*g2), 20, 3)

    pg.draw.line(screen, WHITE, (WIDTH + 20, bottom), (SCREEN_WIDTH - 20, bottom), 5)
    pg.draw.line(screen, WHITE, (WIDTH + 20, top), (WIDTH + 20, bottom), 5)
    
    if GRAPH_NUM == 0:
        max_score = max(best_score + best_avg_score)
        for i in range(len(best_score) - 1):
            pg.draw.line(screen, RED, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_score) - 1), bottom - log(best_score[i], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_score) - 1), bottom - log(best_score[i + 1], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), 5)
        for i in range(len(best_avg_score) - 1):
            pg.draw.line(screen, BLUE, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_score) - 1), bottom - log(best_avg_score[i], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_score) - 1), bottom - log(best_avg_score[i + 1], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), 5)
        text = font.render(f"{max_score:.2f}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 135))
    elif GRAPH_NUM == 1:
        max_fitness = max(best_fitness + best_avg_fitness)
        for i in range(len(best_fitness) - 1):
            pg.draw.line(screen, (255, 255, 0), (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_fitness) - 1), bottom - log(best_fitness[i], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_fitness) - 1), bottom - log(best_fitness[i + 1], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), 5)
        for i in range(len(best_avg_fitness) - 1):
            pg.draw.line(screen, (255, 0, 255), (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_fitness) - 1), bottom - log(best_avg_fitness[i], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_fitness) - 1), bottom - log(best_avg_fitness[i + 1], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), 5)
        text = font.render(f"{max_fitness:.2f}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 135))
    elif GRAPH_NUM == 2:
        for i in range(max(0, gen - len(COLOR_LIST) + 101), gen):
            max_score = 0
            if len(death_score[i]) > 0:
                max_score = max(death_score[i])
            if i == gen - 1:
                max_score = max(max_score, score)
            color = COLOR_LIST[min(gen - i - 1, len(COLOR_LIST)-1)]
            total = 0
            prev_point = (WIDTH + 20, top)
            for k, v in death_score[i].items():
                total += v
                pg.draw.line(screen, color, prev_point, (WIDTH + 20 + (k-1)*(SCREEN_WIDTH-WIDTH-40)/max_score, prev_point[1]), 5)
                pg.draw.line(screen, color, (WIDTH + 20 + (k-1)*(SCREEN_WIDTH-WIDTH-40)/max_score, prev_point[1]), (WIDTH + 20 + k*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(POPULATION - total, GRAPH_LOG)*(bottom - top)/log(POPULATION, GRAPH_LOG)), 5)
                prev_point = (WIDTH + 20 + k*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(POPULATION - total, GRAPH_LOG)*(bottom - top)/log(POPULATION, GRAPH_LOG))
            pg.draw.line(screen, color, prev_point, (SCREEN_WIDTH - 21, prev_point[1]), 5)
            pg.draw.line(screen, color, (SCREEN_WIDTH - 21, prev_point[1]), (SCREEN_WIDTH - 20, bottom), 5)
        text = font.render(f"{POPULATION}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 135))

    if GRAPH_LOG:
        text = font.render("LOG", True, GREEN)
        screen.blit(text, (WIDTH + 30, 160))
        
species = []
population = reproduce([Player() for _ in range(POPULATION)])
dead_population = []
pipes = [Pipe(WIDTH)]
pipe_idx = 0

score, pipe_time, gen, run, pause, speed_idx = 0, 0, 1, True, False, 2
speed = [0.1, 1, 100]
best_score = [0]
best_avg_score = [0]
best_fitness = [0]
best_avg_fitness = [0]
death_score = [{} for _ in range(10000)]

"""
g1 = Genome()
for _ in range(500):
    g1.mutate()
g2 = g1.clone()
#plot the distance between the two genomes after every mutation
dist = []
for _ in range(500):
    g1.mutate()
    dist.append(g1.distance(g2))
plt.plot(dist)
plt.show()
"""

while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                pause = not pause
            if event.key == pg.K_LEFT:
                speed_idx = max(0, speed_idx - 1)
            if event.key == pg.K_RIGHT:
                speed_idx = min(2, speed_idx + 1)
            if event.key == pg.K_l:
                GRAPH_LOG = not GRAPH_LOG
            if event.key == pg.K_e:
                EYES_OPEN = not EYES_OPEN
            if event.key == pg.K_h:
                SHOW_TEXT = not SHOW_TEXT
            if event.key == pg.K_g:
                GRAPH_NUM = (GRAPH_NUM + 1) % 3
            if event.key == pg.K_s:
                SAVE_MODE = not SAVE_MODE
    
    if pause:
        pg.draw.rect(screen, WHITE, (10, 10, 10, 30))
        pg.draw.rect(screen, WHITE, (30, 10, 10, 30))
        pg.display.update()
        continue
    
    score += 1
    pipe_time += 1
    
    screen.fill(BLACK)
    for pipe in pipes:
        pipe.move()
        pipe.draw()
    """
    for player in dead_population:
        player.move(pipes[pipe_idx])
        player.draw()
    """
    for player in population:
        player.move(pipes[pipe_idx])
        player.draw()
    pg.draw.rect(screen, WHITE, (0, HEIGHT - GROUND_SIZE - 2, WIDTH, 100))
    pg.draw.rect(screen, (100, 100, 100), (0, HEIGHT - GROUND_SIZE, WIDTH, 100))
    pg.draw.rect(screen, BLACK, (WIDTH, 0, SCREEN_WIDTH - WIDTH, HEIGHT))
    pg.draw.rect(screen, WHITE, (WIDTH, 0, 3, HEIGHT))
    
    if pipes[pipe_idx].x + PIPE_WIDTH < 0:
        pipes.pop(pipe_idx)
    if pipes[pipe_idx].x + PIPE_WIDTH < PLAYER_X:
        pipe_idx += 1
        for player in population:
            if -2 < player.yspeed < 4:
                player.genome.fitness += 0.05
            else:
                player.genome.fitness -= 0.2
    if pipe_time == PIPE_DIST:
        pipes.append(Pipe(WIDTH))
        pipes[-1].height = GAP_MIN + (pipes[-2].height - GAP_MIN) * 0.95
        if PSEUDO_RANDOM:
            pipes[-1].y = pseudo_random(pipes[-2].y, HEIGHT_RANGE[0], HEIGHT_RANGE[1]) + np.random.randint(-HEIGHT_RAND_RANGE, HEIGHT_RAND_RANGE+1)
            pipe_time = np.random.randint(-DIST_RAND_RANGE, DIST_RAND_RANGE+1)
        else:
            pipes[-1].y = np.random.randint(HEIGHT_RANGE[0], HEIGHT_RANGE[1]) + np.random.randint(-HEIGHT_RAND_RANGE, HEIGHT_RAND_RANGE+1)
            pipe_time = np.random.randint(-DIST_RAND_RANGE, DIST_RAND_RANGE+1)
    
    i = 0
    while i < len(population):
        if population[i].dead:
            dead_population.append(population.pop(i))
        else:
            population[i].feed_forward()
            i += 1
    
    draw_stats()
    pg.display.update()
    
    if len(population) == 0:
        best = max(dead_population, key=lambda x: x.genome.score).genome
        score = 0
        pipe_time = 0
        gen += 1
        for player in dead_population:
            player.genome.fitness += np.log(player.genome.score + 1)
            player.genome.avg_fitness = (player.genome.avg_fitness*3 + player.genome.fitness)/4
            player.genome.avg_score = (player.genome.avg_score*3 + player.genome.score)/4
        best_score.append(max(player.genome.score for player in dead_population))
        best_avg_score.append(max(player.genome.avg_score for player in dead_population))
        best_fitness.append(max(player.genome.fitness for player in dead_population))
        best_avg_fitness.append(max(player.genome.avg_fitness for player in dead_population))
        population = reproduce(dead_population)
        population.sort(key=lambda x: x.genome.fitness)
        dead_population = []

        for player in population:
            player.reset()
        
        pipes = [Pipe(WIDTH)]
        pipe_idx = 0

    if speed[speed_idx] != 100:
        clock.tick(FPS * speed[speed_idx])
