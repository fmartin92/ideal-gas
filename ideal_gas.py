import numpy as np
import copy
import functools
import matplotlib.pyplot as plt
from matplotlib import animation
import pygame

TIME_EPSILON = 0.01
COLLISION_THRESHOLD = 0.01
EDGE_LENGTH = 1
NUMBER_OF_PARTICLES = 50
SECONDS = 5
FRAMES_PER_SECOND = 30
FRAMES = SECONDS * FRAMES_PER_SECOND
DIMENSION = 2
CANVAS_SIZE = 500


class Particle:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel

    def bounce_off_border(self):
        bounce_sign = [1 if abs(self.pos[i] + TIME_EPSILON * self.vel[i])
                       < EDGE_LENGTH else -1 for i in range(DIMENSION)]
        vel = [bounce_sign[i] * self.vel[i] for i in range(DIMENSION)]
        pos = [self.pos[i] + TIME_EPSILON * vel[i] for i in range(DIMENSION)]
        return Particle(pos, vel)

    def is_out_of_bounds(self):
        return functools.reduce(lambda x, y: x or y, [abs(self.pos[i]) > EDGE_LENGTH for i in range(DIMENSION)])

    def next_step(self):
        naive_update = Particle(
            [self.pos[i] + TIME_EPSILON * self.vel[i] for i in range(DIMENSION)], self.vel)

        if naive_update.is_out_of_bounds():
            return self.bounce_off_border()
        else:
            return naive_update


def random_particle():
    pos = list(np.random.uniform(-EDGE_LENGTH, EDGE_LENGTH, DIMENSION))
    vel = list(np.random.normal(size=DIMENSION))
    return Particle(pos, vel)


def inner_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def norm(x):
    return np.sqrt(inner_product(x, x))


def distance(x, y):
    return norm([x[i] - y[i] for i in range(len(x))])


def collide(particles):
    collisions = []
    parts = copy.deepcopy(particles)
    for i1 in range(len(particles)):
        for i2 in range(i1 + 1, len(particles)):
            if distance(parts[i1].pos, parts[i2].pos) < COLLISION_THRESHOLD:
                collisions.append(parts[i1].pos)
                vel_diff = [parts[i1].vel[i] - parts[i2].vel[i]
                            for i in range(DIMENSION)]
                pos_diff = [parts[i1].pos[i] - parts[i2].pos[i]
                            for i in range(DIMENSION)]
                q = inner_product(vel_diff, pos_diff) / norm(pos_diff)**2
                parts[i1] = Particle(
                    parts[i1].pos, [parts[i1].vel[i] - q * pos_diff[i] for i in range(DIMENSION)])
                parts[i2] = Particle(
                    parts[i2].pos, [parts[i2].vel[i] + q * pos_diff[i] for i in range(DIMENSION)])
    return (parts, collisions)


def next_step(particles):
    collided_particles = collide(particles)[0]
    return [particle.next_step() for particle in collided_particles]


def evolve_system(particles):
    history = [particles]
    for _ in range(FRAMES):
        history.append(next_step(history[-1]))
    return history


def generate_animation(particles):
    '''this needs DIMENSION = 2 in order to work'''
    fig = plt.figure()
    ax = plt.axes(xlim=(-EDGE_LENGTH, EDGE_LENGTH),
                  ylim=(-EDGE_LENGTH, EDGE_LENGTH))
    points, = ax.plot([], [], 'bo', ms=6)
    red_dot, = ax.plot([], [], 'or', ms=6)

    history = evolve_system(particles)

    def init():
        return points, red_dot

    def animate(i):
        x = [p.pos[0] for p in history[i][1:]]
        y = [p.pos[1] for p in history[i][1:]]
        points.set_data(x, y)
        red_dot.set_data(history[i][0].pos[0], history[i][0].pos[1])
        return points, red_dot

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=FRAMES, interval=1000/FRAMES_PER_SECOND, blit=True)
    anim.save('particle.gif', writer='PillowWriter')


def convert_coords(coords):
    return [CANVAS_SIZE*(coord+1)/2 for coord in coords]


def pygame_animate(particles):
    screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
    blue_ball = pygame.image.load('blue_ball.png').convert()
    red_ball = pygame.image.load('red_ball.png').convert()
    background = pygame.image.load('background.png').convert()
    screen.blit(background, (0, 0))
    should_run = True
    while should_run:
        screen.blit(background, (0, 0))
        particles = next_step(particles)
        current_pos = convert_coords(particles[0].pos)
        current_pos = red_ball.get_rect(center=current_pos)
        screen.blit(red_ball, (current_pos.left, current_pos.top))
        for p in particles[1:]:
            current_pos = convert_coords(p.pos)
            current_pos = blue_ball.get_rect(center=current_pos)
            screen.blit(blue_ball, (current_pos.left, current_pos.top))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_run = False
    pygame.quit()


def pygame_animate_collisions(particles):
    screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
    blue_ball = pygame.image.load('blue_ball.png').convert()
    background = pygame.image.load('background.png').convert()
    screen.blit(background, (0, 0))
    should_run = True
    while should_run:
        [particles, collisions] = collide(particles)
        particles = [particle.next_step() for particle in particles]
        for p in collisions:
            current_pos = convert_coords(p)
            current_pos = blue_ball.get_rect(center=current_pos)
            screen.blit(blue_ball, (current_pos.left, current_pos.top))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_run = False
    pygame.quit()
