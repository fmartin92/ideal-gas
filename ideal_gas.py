import numpy as np
import copy
import functools
import matplotlib.pyplot as plt
from matplotlib import animation
import pygame

HALF_EDGE_LENGTH = 1
PARTICLE_RADIUS = 2 * HALF_EDGE_LENGTH / 100
FRAMES_PER_SECOND = 60
TIME_EPSILON = 1 / FRAMES_PER_SECOND
NUMBER_OF_PARTICLES = 60
SECONDS = 5
FRAMES = SECONDS * FRAMES_PER_SECOND
DIMENSION = 2
CANVAS_SIZE = 500
PARTICLE_RADIUS_IN_PIXELS = int(CANVAS_SIZE / 100)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def inner_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def norm(x):
    return np.sqrt(inner_product(x, x))


def distance(x, y):
    return norm([x[i] - y[i] for i in range(len(x))])


class Particle:
    def __init__(self, pos, vel, out_of_bounds_fn, bounce_fn):
        self.pos = pos
        self.vel = vel
        self.out_of_bounds_fn = out_of_bounds_fn
        self.bounce_fn = bounce_fn

    def is_out_of_bounds(self):
        return self.out_of_bounds_fn(self)

    def bounce_off_border(self):
        return self.bounce_fn(self)

    def __repr__(self):
        def stringify(vector):
            return ', '.join([str(coord) for coord in vector])
        return f'Particle at ({stringify(self.pos)}) with velocity vector ({stringify(self.vel)})'

    def next_step(self):
        naive_update = Particle(
            [self.pos[i] + TIME_EPSILON * self.vel[i] for i in range(DIMENSION)], self.vel, self.out_of_bounds_fn, self.bounce_fn)

        if naive_update.is_out_of_bounds():
            return self.bounce_off_border()
        else:
            return naive_update


class Collision:
    def __init__(self, particle1, particle2, idx1, idx2):
        self.particle1 = particle1
        self.particle2 = particle2
        self.idx1 = idx1
        self.idx2 = idx2

    def __eq__(self, other):
        return self.idx1 == other.idx1 and self.idx2 == other.idx2

    def resolve_collision(self):
        vel_diff = [self.particle1.vel[i] - self.particle2.vel[i]
                    for i in range(DIMENSION)]
        pos_diff = [self.particle1.pos[i] - self.particle2.pos[i]
                    for i in range(DIMENSION)]
        q = inner_product(vel_diff, pos_diff) / norm(pos_diff)**2
        new_particle1 = Particle(
            self.particle1.pos, [self.particle1.vel[i] - q * pos_diff[i] for i in range(DIMENSION)], self.particle1.out_of_bounds_fn, self.particle1.bounce_fn)
        new_particle2 = Particle(
            self.particle2.pos, [self.particle2.vel[i] + q * pos_diff[i] for i in range(DIMENSION)], self.particle1.out_of_bounds_fn, self.particle1.bounce_fn)
        return (new_particle1, new_particle2)


class Particle_Ensemble:
    def __init__(self, particles, prev_collisions=[]):
        self.particles = particles
        self.prev_collisions = prev_collisions

    def calculate_collisions(self):
        collisions = []
        for i1 in range(len(self.particles)):
            for i2 in range(i1 + 1, len(self.particles)):
                if distance(self.particles[i1].pos, self.particles[i2].pos) < PARTICLE_RADIUS:
                    collisions.append(
                        Collision(self.particles[i1], self.particles[i2], i1, i2))
        return collisions

    def next_step(self):
        new_particles = copy.deepcopy(self.particles)
        collisions = self.calculate_collisions()
        for collision in collisions:
            if collision not in self.prev_collisions:
                (new_particles[collision.idx1], new_particles[collision.idx2]
                 ) = collision.resolve_collision()
        for i in range(len(new_particles)):
            new_particles[i] = new_particles[i].next_step()
        return Particle_Ensemble(new_particles, collisions)


def random_rectangle_ensemble():
    def bounce_off_border(particle):
        bounce_sign = [1 if abs(particle.pos[i] + TIME_EPSILON * particle.vel[i])
                       < HALF_EDGE_LENGTH else -1 for i in range(DIMENSION)]
        vel = [bounce_sign[i] * particle.vel[i] for i in range(DIMENSION)]
        pos = [particle.pos[i] + TIME_EPSILON * vel[i]
               for i in range(DIMENSION)]
        return Particle(pos, vel, particle.out_of_bounds_fn, particle.bounce_fn)

    def is_out_of_bounds(particle):
        return functools.reduce(lambda x, y: x or y, [abs(particle.pos[i]) > HALF_EDGE_LENGTH for i in range(DIMENSION)])

    def random_particle():
        pos = list(np.random.uniform(-HALF_EDGE_LENGTH,
                                     HALF_EDGE_LENGTH, DIMENSION))
        vel = list(np.random.normal(size=DIMENSION))
        return Particle(pos, vel, is_out_of_bounds, bounce_off_border)

    return Particle_Ensemble([random_particle() for _ in range(NUMBER_OF_PARTICLES)])


def convert_coords(coords):
    return [int(CANVAS_SIZE*(coord+1)/2) for coord in coords]


def pygame_animate(particle_ensemble):
    screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
    background = pygame.image.load('background.png').convert()
    screen.blit(background, (0, 0))
    should_run = True
    while should_run:
        screen.blit(background, (0, 0))
        particle_ensemble = particle_ensemble.next_step()
        red_ball_pos = convert_coords(particle_ensemble.particles[0].pos)
        pygame.draw.circle(screen, RED,
                           red_ball_pos, PARTICLE_RADIUS_IN_PIXELS)
        for p in particle_ensemble.particles[1:]:
            current_pos = convert_coords(p.pos)
            pygame.draw.circle(screen, BLUE,
                               current_pos, PARTICLE_RADIUS_IN_PIXELS)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_run = False
    pygame.quit()


# we may want to get rid of this altogether v
def evolve_system(particles):
    history = [particles]
    for _ in range(FRAMES):
        history.append(next_step(history[-1]))
    return history


def generate_animation(particles):
    '''this needs DIMENSION = 2 in order to work'''
    fig = plt.figure()
    ax = plt.axes(xlim=(-HALF_EDGE_LENGTH, HALF_EDGE_LENGTH),
                  ylim=(-HALF_EDGE_LENGTH, HALF_EDGE_LENGTH))
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


ens = random_rectangle_ensemble()
pygame_animate(ens)
