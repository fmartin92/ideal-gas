import numpy as np
import copy
import functools
import matplotlib.pyplot as plt
from matplotlib import animation
import pygame
from pygame import gfxdraw

HALF_EDGE_LENGTH = 1
CANVAS_SIZE = 400
PARTICLE_RADIUS = 2 * HALF_EDGE_LENGTH / 100
PARTICLE_RADIUS_IN_PIXELS = int(CANVAS_SIZE / 100)
FRAMES_PER_SECOND = 60
TIME_EPSILON = 1 / FRAMES_PER_SECOND
DIMENSION = 2


RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)


def convert_coords(coords):
    return [int(CANVAS_SIZE*(coord+1)/2) for coord in coords]


def inner_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def norm(x):
    return np.sqrt(inner_product(x, x))


def distance(x, y):
    return norm([x[i] - y[i] for i in range(len(x))])


class Particle:
    def __init__(self, pos, vel, receptacle):
        self.pos = pos
        self.vel = vel
        self.receptacle = receptacle

    def is_out_of_bounds(self):
        return not self.receptacle.contains(self)

    def bounce_off_border(self):
        (new_pos, new_vel) = self.receptacle.bounce(self)
        return Particle(new_pos, new_vel, self.receptacle)

    def __repr__(self):
        def stringify(vector):
            return ', '.join([str(coord) for coord in vector])
        return f'Particle at ({stringify(self.pos)}) with velocity vector ({stringify(self.vel)})'

    def next_step(self):
        naive_update = Particle(
            [self.pos[i] + TIME_EPSILON * self.vel[i] for i in range(DIMENSION)], self.vel, self.receptacle)

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
            self.particle1.pos, [self.particle1.vel[i] - q * pos_diff[i] for i in range(DIMENSION)], self.particle1.receptacle)
        new_particle2 = Particle(
            self.particle2.pos, [self.particle2.vel[i] + q * pos_diff[i] for i in range(DIMENSION)], self.particle2.receptacle)
        return (new_particle1, new_particle2)


class Particle_Ensemble:
    def __init__(self, particles, receptacle, prev_collisions=[]):
        self.particles = particles
        self.receptacle = receptacle
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
        return Particle_Ensemble(new_particles, self.receptacle, collisions)


class Bounding_Receptacle:
    def __init__(self, contains, bounce, draw_method):
        self.contains = contains
        self.bounce = bounce
        self.draw_method = draw_method

    def random_particle(self):
        candidate_particle = Particle(
            [2 * HALF_EDGE_LENGTH for _ in range(DIMENSION)], [2 * HALF_EDGE_LENGTH for _ in range(DIMENSION)], self)  # always out of bounds
        while candidate_particle.is_out_of_bounds():
            pos = list(np.random.uniform(-HALF_EDGE_LENGTH,
                                         HALF_EDGE_LENGTH, DIMENSION))
            vel = list(np.random.normal(size=DIMENSION))
            candidate_particle = Particle(pos, vel, self)
        return candidate_particle


class System:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def animate(self):
        screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
        should_run = True
        while should_run:
            screen.fill(WHITE)
            self.ensemble.receptacle.draw_method(screen)
            self.ensemble = self.ensemble.next_step()
            (x, y) = convert_coords(self.ensemble.particles[0].pos)
            gfxdraw.filled_circle(screen,
                                  x, y, PARTICLE_RADIUS_IN_PIXELS, RED)
            for p in self.ensemble.particles[1:]:
                (x, y) = convert_coords(p.pos)
                gfxdraw.filled_circle(
                    screen, x, y, PARTICLE_RADIUS_IN_PIXELS, BLUE)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_run = False
        pygame.quit()

    def draw_collisions(self):
        screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
        should_run = True
        screen.fill(WHITE)
        self.ensemble.receptacle.draw_method(screen)
        while should_run:
            collisions = self.ensemble.calculate_collisions()
            self.ensemble = self.ensemble.next_step()
            for p in collisions:
                current_pos = convert_coords(p.particle1.pos)
                pygame.draw.circle(screen, BLUE,
                                   current_pos, PARTICLE_RADIUS_IN_PIXELS)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_run = False
        pygame.quit()


def random_ensemble(number_of_particles, receptacle):
    particles = [receptacle.random_particle()
                 for _ in range(number_of_particles)]
    return Particle_Ensemble(particles, receptacle)


def rectangle_contains(particle):
    return functools.reduce(lambda x, y: x and y, [abs(particle.pos[i]) < HALF_EDGE_LENGTH for i in range(DIMENSION)])


def rectangle_bounce(particle):
    bounce_sign = [1 if abs(particle.pos[i] + TIME_EPSILON * particle.vel[i])
                   < HALF_EDGE_LENGTH else -1 for i in range(DIMENSION)]
    vel = [bounce_sign[i] * particle.vel[i] for i in range(DIMENSION)]
    pos = [particle.pos[i] + TIME_EPSILON * vel[i]
           for i in range(DIMENSION)]
    return (pos, vel)


def rectangle_draw(screen):
    pass


def circle_contains(particle):
    return norm(particle.pos) < 1


def circle_bounce(particle):
    tangent = [-particle.pos[1], particle.pos[0]]
    aux = [2*inner_product(particle.vel, tangent) * coord for coord in tangent]
    vel = [-particle.vel[i] + aux[i] for i in range(DIMENSION)]
    pos = [particle.pos[i] + TIME_EPSILON * vel[i] for i in range(DIMENSION)]
    return (pos, vel)


def circle_draw(screen):
    center = int(CANVAS_SIZE/2)
    gfxdraw.filled_circle(screen, center, center, center, GRAY)


if __name__ == '__main__':
    rectangle = Bounding_Receptacle(
        rectangle_contains, rectangle_bounce, rectangle_draw)
    circle = Bounding_Receptacle(circle_contains, circle_bounce, circle_draw)
    system = System(random_ensemble(50, circle))
    system.animate()
