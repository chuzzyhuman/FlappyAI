import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 600, 600
SCREEN_WIDTH = 1200
FPS, GROUND_SPEED = 60, 7
PLAYER_SIZE, GROUND_SIZE = 40, 100
PIPE_WIDTH, PIPE_EDGE, PIPE_HEIGHT, GAP_MIN, PIPE_DIST = 5, 0, 100, 75, 70
HEIGHT_RANGE = (50, HEIGHT - GROUND_SIZE - 250)
EYES_OPEN = False
SHOW_TEXT, GRAPH_LOG = True, True

BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)
SPECIES_COLOR = [(i, 255, 0) for i in range(0, 256, 51)] + [(255, i, 0) for i in range(255, -1, -51)] + [(i, 0, 0) for i in range(255, -1, -51)] + [(0, 0, 0) for i in range(50)]

INPUTS, OUTPUTS = 4, 1
NODE_ID, INNOVATION = INPUTS + OUTPUTS, 0
POPULATION = 5000

c1, c2, c3 = 1, 1, 0.4
MAX_WEIGHT, MAX_BIAS = 5, 5
DELTA_THRESHOLD = 0.4
DEL_NODE, ADD_NODE = 0.02, 0.02
DEL_LINK, ADD_LINK = 0.05, 0.3
MUTATE_PROB = 0.7
ACTIVATION_MODE = 2
ACTIVATION_THRESHOLD = [0, 0.5, 0.5, 0.5]
MAX_LAYER = 3

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
    if n == 4:
        return np.log10(1+x)

class Node:
    def __init__(self, id, bias=0):
        self.id = id
        self.bias = bias
    
    def clone(self):
        n = Node(self.id, self.bias)
        return n
        
class Link:
    def __init__(self, in_id, out_id, weight=0, enabled=True, innov=-1):
        self.in_id = in_id
        self.out_id = out_id
        self.weight = weight
        self.enabled = enabled
        self.innov = innov
        
    def clone(self):
        l = Link(self.in_id, self.out_id, self.weight, self.enabled, self.innov)
        return l

class Genome:
    def __init__(self):
        self.nodes = np.array([Node(i) for i in range(INPUTS + OUTPUTS)])
        self.links = np.array([], dtype=object)
        self.fitness = 0
        self.avg_fitness = 0
        self.species = 0
        self.reload()

    def clone(self):
        g = Genome()
        g.nodes = np.array([n.clone() for n in self.nodes])
        g.links = np.array([c.clone() for c in self.links])
        g.fitness = self.fitness
        g.avg_fitness = self.avg_fitness
        g.species = self.species
        g.reload()
        return g

    def reload(self):
        for node in self.nodes[INPUTS + OUTPUTS:]:
            if not np.any([link.in_id == node.id and link.enabled for link in self.links]) or not np.any([link.out_id == node.id and link.enabled for link in self.links]):
                self.nodes = self.nodes[self.nodes != node]
        self.links = np.array([link for link in self.links if link.in_id in [n.id for n in self.nodes] and link.out_id in [n.id for n in self.nodes]])
        
        self.id_to_index = {n.id: i for i, n in enumerate(self.nodes)}
        self.layer = np.zeros(len(self.nodes))
        self.layer[:INPUTS] = 0
        self.layer[INPUTS:] = 1
        self.order = []
        self.list_order = []
        
        queue = [[n.id, []] for n in self.nodes]
        while queue:
            first = queue.pop(0)
            n, prev = first[0], first[1]
            for link in self.links:
                if link.in_id == n and link.enabled:
                    if link.out_id in prev:
                        link.enabled = False
                    else:
                        self.layer[self.id_to_index[link.out_id]] = max(
                            self.layer[self.id_to_index[link.out_id]], 
                            self.layer[self.id_to_index[n]] + 1
                        )
                        queue.append([link.out_id, prev + [n]])

        for i in range(INPUTS, INPUTS + OUTPUTS):
            self.layer[i] = max(self.layer[INPUTS+OUTPUTS:]) + 1 if self.layer[INPUTS+OUTPUTS:].size > 0 else 1
        self.max_layer = int(max(self.layer))
        self.layer_dict = {i: [n.id for n in self.nodes if self.layer[self.id_to_index[n.id]] == i] for i in range(self.max_layer + 1)}
        
        for i in range(self.max_layer + 1):
            self.order += self.layer_dict[i]

        for n in self.order:
            for link in self.links:
                if link.in_id == n and link.enabled:
                    self.list_order.append(link)

    def distance(self, other):
        excess, disjoint, weight, same = 0, 0, 0, 1
        i, j = 0, 0
        while i < len(self.links) and j < len(other.links):
            if self.links[i].innov == other.links[j].innov:
                weight += abs(self.links[i].weight - other.links[j].weight)
                same += 1
                i += 1
                j += 1
            elif self.links[i].innov < other.links[j].innov:
                disjoint += 1
                i += 1
            else:
                disjoint += 1
                j += 1
        excess += len(self.links) - i + len(other.links) - j
        n = min(max(len(self.links), len(other.links), 1), 30)
        return (c1 * excess + c2 * disjoint) / n + c3 * weight / same

    def feed_forward(self, inputs):
        self.value = np.zeros(len(self.nodes))
        self.value[:INPUTS] = inputs
        cnt = 0
        """
        for node in self.order:
            idx = self.id_to_index[node]
            self.value[idx] = squash(self.value[idx] + self.nodes[idx].bias, ACTIVATION_MODE)
            while(cnt < len(self.list_order) and self.list_order[cnt].in_id == node):
                link = self.list_order[cnt]
                self.value[self.id_to_index[link.out_id]] += self.value[idx] * link.weight
                cnt += 1
        """
        while(cnt < len(self.list_order)):
            if cnt == 0 or self.list_order[cnt].in_id != self.list_order[cnt - 1].in_id:
                idx = self.id_to_index[self.list_order[cnt].in_id]
                self.value[idx] = squash(self.value[idx] + self.nodes[idx].bias, ACTIVATION_MODE)
            link = self.list_order[cnt]
            self.value[self.id_to_index[link.out_id]] += self.value[self.id_to_index[link.in_id]] * link.weight
            cnt += 1
        return self.value[INPUTS:INPUTS+OUTPUTS]

    def add_node(self, link):
        global NODE_ID, INNOVATION
        link.enabled = False
        new_node = Node(NODE_ID)
        self.nodes = np.append(self.nodes, new_node)
        if ACTIVATION_MODE == 2:
            self.links = np.append(self.links, [Link(link.in_id, NODE_ID, link.weight / MAX_BIAS, True, INNOVATION), 
                                                Link(NODE_ID, link.out_id, MAX_BIAS, True, INNOVATION + 1)])
        INNOVATION += 2
        NODE_ID += 1

    def add_link(self):
        global INNOVATION
        in_node = np.random.choice([n.id for n in np.concatenate((self.nodes[:INPUTS], self.nodes[INPUTS + OUTPUTS:]))])
        out_node = np.random.choice([n.id for n in self.nodes[INPUTS:]])
        if self.layer[self.id_to_index[in_node]] > self.layer[self.id_to_index[out_node]]:
            in_node, out_node = out_node, in_node
        elif in_node == out_node:
            return
        for link in self.links:
            if link.in_id == in_node and link.out_id == out_node:
                link.enabled = True
                link.weight = 0
                return
        self.links = np.append(self.links, Link(in_node, out_node, np.random.uniform(-0.001, 0.001), True, INNOVATION))
        INNOVATION += 1

    def delete_node(self):
        node = np.random.choice([n.id for n in self.nodes[INPUTS + OUTPUTS:]])
        self.nodes = self.nodes[self.nodes != node]
        self.links = self.links[np.logical_not(np.isin([link.in_id for link in self.links], node)) & 
                                np.logical_not(np.isin([link.out_id for link in self.links], node))]

    def delete_link(self):
        if len(self.links) > 0:
            link = np.random.choice(self.links)
            link.enabled = False

    def change_weight(self):
        if len(self.links) > 0:
            link = np.random.choice(self.links)
            link.weight += np.random.randn() / 10
            link.weight = np.clip(link.weight, -MAX_WEIGHT, MAX_WEIGHT)

    def change_bias(self):
        if len(self.nodes) > INPUTS:
            node = np.random.choice(self.nodes[INPUTS:])
            node.bias += np.random.randn() / 10
            node.bias = np.clip(node.bias, -MAX_BIAS, MAX_BIAS)

    def mutate_node(self):
        r = np.random.rand()
        if r < DEL_NODE and len(self.nodes) > INPUTS + OUTPUTS:
            self.delete_node()
        elif r < DEL_NODE + ADD_NODE and len([link for link in self.links if link.enabled]) > 0:
            self.add_node(np.random.choice([link for link in self.links if link.enabled]))
        elif len(self.nodes) > INPUTS + OUTPUTS:
            self.change_bias()

    def mutate_link(self):
        r = np.random.rand()
        len_links = len([link for link in self.links if link.enabled])
        if r < DEL_LINK and len(self.links) > 0:
            self.delete_link()
        elif r < DEL_NODE + ADD_LINK:
            self.add_link()
        elif len(self.links) > 0:
            self.change_weight()

    def mutate(self):
        r = np.random.rand()
        if r < MUTATE_PROB:
            self.mutate_link()
        else:
            self.mutate_node()
        self.reload()

    def crossover(self, other):
        child = Genome()
        child.nodes = np.array([])
        i, j = 0, 0
        while i < len(self.nodes) and j < len(other.nodes):
            if self.nodes[i].id == other.nodes[j].id:
                child.nodes = np.append(child.nodes, self.nodes[i].clone())
                i += 1
                j += 1
            elif self.nodes[i].id < other.nodes[j].id:
                child.nodes = np.append(child.nodes, self.nodes[i].clone())
                i += 1
            else:
                child.nodes = np.append(child.nodes, other.nodes[j].clone())
                j += 1
        while i < len(self.nodes):
            child.nodes = np.append(child.nodes, self.nodes[i].clone())
            i += 1
        while j < len(other.nodes):
            child.nodes = np.append(child.nodes, other.nodes[j].clone())
            j += 1

        i, j = 0, 0
        while i < len(self.links) and j < len(other.links):
            if self.links[i].innov == other.links[j].innov:
                if np.random.rand() < 0.5:
                    child.links = np.append(child.links, self.links[i].clone())
                else:
                    child.links = np.append(child.links, other.links[j].clone())
                i += 1
                j += 1
            elif self.links[i].innov < other.links[j].innov:
                if np.random.rand() < 0.9:
                    child.links = np.append(child.links, self.links[i].clone())
                i += 1
            else:
                if np.random.rand() < 0.9:
                    child.links = np.append(child.links, other.links[j].clone())
                j += 1
        while i < len(self.links):
            child.links = np.append(child.links, self.links[i].clone())
            i += 1
        while j < len(other.links):
            child.links = np.append(child.links, other.links[j].clone())
            j += 1

        child.reload()
        return child

class Player:
    def __init__(self, genome=None):
        self.x = 50
        self.y = 200
        self.width = PLAYER_SIZE
        self.height = PLAYER_SIZE
        self.yspeed = 0
        self.dead = False
        self.input = []
        self.genome = genome if genome else Genome()
        
    def reset(self):
        self.x = 50
        self.y = 200
        self.yspeed = 0
        self.dead = False
        self.genome.fitness = 0
        
    def feed_forward(self):
        outputs = self.genome.feed_forward(self.input)
        if outputs[0] > ACTIVATION_THRESHOLD[ACTIVATION_MODE]:
            self.jump()
        
    def jump(self):
        if not self.dead:
            self.yspeed = -8
        
    def move(self, pipe):
        self.y += self.yspeed
        self.input = [(pipe.x - self.x - PLAYER_SIZE) / WIDTH,
                      #(pipe.y - self.y) / PLAYER_SIZE,
                      #(pipe.y - self.y + pipe.height - self.height) / PLAYER_SIZE,
                      (pipe.y - self.y + (pipe.height - self.height) / 2) / (PLAYER_SIZE*2),
                      (HEIGHT - GROUND_SIZE - self.y) / HEIGHT, 
                      self.yspeed / 10]
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
        if not self.dead:
            self.genome.fitness += 0.01
        
    def draw(self):
        pg.draw.rect(screen, WHITE, (self.x - 2, self.y - 2, self.width + 4, self.height + 4))
        pg.draw.rect(screen, BLUE if self.dead else SPECIES_COLOR[self.genome.species], (self.x, self.y, self.width, self.height))
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
        self.y = 100
        self.width = PIPE_WIDTH
        self.height = PIPE_HEIGHT
        
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
    for i in range(len(species)//4+1):
        n = 0
        if i <= 5:
            n = 50
        elif i <= 10:
            n = 20
        elif i <= 15:
            n = 5
        s = species[i]
        for _ in range(POPULATION//20+n):
            j = min(int(abs(np.random.randn())*3), 10, len(s)-1)
            child = s[j].clone()
            for _ in range(np.random.randint(5)):
                child.mutate()
            if child.max_layer <= MAX_LAYER and max([len(l) for l in child.layer_dict.values()]) <= 5:
                child.avg_fitness = s[j].avg_fitness
                new_population.append(child)
    for s in species:
        for i in range(min(len(s), 30)):
            parent1 = np.random.choice(s[:min(len(s)//4+1, 10)])
            parent2 = np.random.choice(s[:min(len(s)//4+1, 10)])
            child = parent1.crossover(parent2)
            if child.max_layer <= MAX_LAYER and max([len(l) for l in child.layer_dict.values()]) <= 5:
                child.avg_fitness = max(parent1.avg_fitness, parent2.avg_fitness)
                new_population.append(child)
    for p in sorted(population, key=lambda x: x.genome.fitness, reverse=True):
        new_population.append(p.genome.clone())
    return [Player(genome) for genome in new_population[:POPULATION]]

def draw_stats():
    text = font.render("Slow" if speed[speed_idx] == 0.1 else "" if speed[speed_idx] == 1 else "Fast", True, WHITE)
    screen.blit(text, (WIDTH - 55, 10))
    text = font.render(f"Generation: {gen}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 10))
    text = font.render(f"Time: {(time/100):.2f}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 40))
    text = font.render(f"Alive: {len(population)}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 70))
    text = font.render(f"Species: {[len(s) for s in species]}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 100))
    
    top, bottom = 140, 300
    w, h, g1, g2 = 90, 340, 80, 45
    min_value, max_value = -1, 1
    best = max(population + dead_population, key=lambda x: x.genome.fitness).genome
    
    if SHOW_TEXT:
        for i, value in enumerate(max(population + dead_population, key=lambda x: x.genome.fitness).input):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 20, h - 12 + i*g2))
        for i, value in enumerate(max(population + dead_population, key=lambda x: x.genome.fitness).genome.value[INPUTS:INPUTS+OUTPUTS]):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 100 + best.max_layer*g1 + 20, h - 12 + i*g2))
    for i in range(best.max_layer + 1):
        for node in best.layer_dict[i]:
            for link in best.links:
                if link.in_id == node and link.enabled:
                    pos1 = (WIDTH + w + i*g1, h + best.layer_dict[i].index(node)*g2)
                    pos2 = (WIDTH + w + best.layer[best.id_to_index[link.out_id]]*g1, h + best.layer_dict[best.layer[best.id_to_index[link.out_id]]].index(link.out_id)*g2)
                    pg.draw.line(screen, RED if link.weight > 0 else BLUE, pos1, pos2, abs(int(link.weight)) + 2)
        for node in best.layer_dict[i]:
            b = int(255 * (best.value[best.id_to_index[node]] - min_value) / (max_value - min_value))
            b = max(0, min(255, b))
            pg.draw.circle(screen, (b, b, b) if best.nodes[best.id_to_index[node]].bias == 0 else ((b, 0, 0) if best.nodes[best.id_to_index[node]].bias > 0 else (0, 0, b)), (WIDTH + w + i*g1, h + best.layer_dict[i].index(node)*g2), 20)
            pg.draw.circle(screen, WHITE, (WIDTH + w + i*g1, h + best.layer_dict[i].index(node)*g2), 20, 3)
            if SHOW_TEXT:
                text = font.render(str(node), True, WHITE)
                screen.blit(text, (WIDTH + w + i*g1 - 4.7 - 5.2*int(np.log10(node if node != 0 else 1)), h + best.layer_dict[i].index(node)*g2 - 12))
    pg.draw.line(screen, WHITE, (WIDTH + 20, bottom), (SCREEN_WIDTH - 20, bottom), 5)
    pg.draw.line(screen, WHITE, (WIDTH + 20, top), (WIDTH + 20, bottom), 5)
    max_time = max(best_time + best_avg_time)
    text = font.render(f"{max_time:.2f}", True, WHITE)
    screen.blit(text, (WIDTH + 30, 135))
    
    for i in range(len(best_time) - 1):
        pg.draw.line(screen, RED, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_time) - 1), bottom - squash(best_time[i], GRAPH_LOG*4)*(bottom - top)/squash(max_time, GRAPH_LOG*4)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_time) - 1), bottom - squash(best_time[i + 1], GRAPH_LOG*4)*(bottom - top)/squash(max_time, GRAPH_LOG*4)), 5)
    for i in range(len(best_avg_time) - 1):
        pg.draw.line(screen, BLUE, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_time) - 1), bottom - squash(best_avg_time[i], GRAPH_LOG*4)*(bottom - top)/squash(max_time, GRAPH_LOG*4)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_time) - 1), bottom - squash(best_avg_time[i + 1], GRAPH_LOG*4)*(bottom - top)/squash(max_time, GRAPH_LOG*4)), 5)
    
    if GRAPH_LOG:
        text = font.render("LOG", True, GREEN)
        screen.blit(text, (WIDTH + 30, 160))
    
    """
    for i, node in enumerate(best.nodes):
        print(f"Node {i}: {node.id}, {node.bias}")
    for i, link in enumerate(best.links):
        print(f"Link {i}: {link.in_id} -> {link.out_id}, {link.weight}, {link.enabled}, {link.innov}")
    print()
    """
    
species = []
population = reproduce([Player() for _ in range(POPULATION)])
dead_population = []
pipes = [Pipe(WIDTH)]
pipe_idx = 0

time, pipe_time, gen, run, pause, speed_idx = 0, 0, 1, True, False, 2
speed = [0.1, 1, 100]
best_time = [0]
best_avg_time = [0]

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
            if event.key == pg.K_g:
                GRAPH_LOG = not GRAPH_LOG
            if event.key == pg.K_e:
                EYES_OPEN = not EYES_OPEN
            if event.key == pg.K_s:
                SHOW_TEXT = not SHOW_TEXT
    
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
    if pipes[pipe_idx].x + PIPE_WIDTH < 50:
        pipe_idx += 1
    if pipe_time == PIPE_DIST:
        pipes.append(Pipe(WIDTH))
        pipes[-1].height = max(pipes[-2].height-1, GAP_MIN)
        pipes[-1].y = pseudo_random(pipes[-2].y, HEIGHT_RANGE[0], HEIGHT_RANGE[1]) + np.random.randint(-5, 5)
        PIPE_DIST = np.random.randint(70, 80)
        #PIPE_DIST = 70 + np.random.randint(-5, 5)
        pipe_time = 0
    
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
        best = max(dead_population, key=lambda x: x.genome.fitness).genome
        for i, node in enumerate(best.nodes):
            print(f"Node {i}: {node.id}, {node.bias}")
        for i, link in enumerate(best.links):
            print(f"Link {i}: {link.in_id} -> {link.out_id}, {link.weight}, {link.enabled}, {link.innov}")
        print()
        print([len(s) for s in species])
        print()
        time = 0
        pipe_time = 0
        gen += 1
        for player in dead_population:
            player.genome.avg_fitness = (player.genome.avg_fitness*3 + player.genome.fitness)/4
        best_time.append(max(player.genome.fitness for player in dead_population))
        best_avg_time.append(max(player.genome.avg_fitness for player in dead_population))
        population = reproduce(dead_population)
        dead_population = []
        for player in population:
            player.reset()
            player.genome.reload()
        pipes = [Pipe(WIDTH)]
        pipe_idx = 0

        #node_ids is n.id of all the population
        node_ids = [n.id for n in np.concatenate(([player.genome.nodes for player in population]))]
        for i in range(1, 10000000):
            if i not in node_ids:
                NODE_ID = i
                break
        innovs = [l.innov for l in np.concatenate(([player.genome.links for player in population]))]
        for i in range(1, 10000000):
            if i not in innovs:
                INNOVATION = i
                break
    
    if speed[speed_idx] != 100:
        clock.tick(FPS * speed[speed_idx])
