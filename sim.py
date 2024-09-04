#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Copyright (c) 2024 Egor Vasilenko and others

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import itertools
import math
import numpy as np
from numpy import exp
import random
import matplotlib
from matplotlib import pyplot as plot
import pygame
import os
matplotlib.rcParams['figure.dpi'] = 300
os.environ["PATH"] += os.pathsep + "/usr/local/texlive/2023/bin/universal-darwin"


"""
Obtaining screen parameters.
"""
surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
surface_width, surface_height = pygame.display.get_surface().get_size()
print("surface size", surface_width, "x", surface_height)


"""
Statistics to collect.
"""
statistics = "time_alive" # W_average time_alive


"""
Setting model parameters.
"""
T_sim = 1000
dt = 1
W = 1000
phi = 5 * 2
omega = 10**(-2)
pump_cost = 10**(-2)
h = 10**(-1)#1#5 * 10**(-1)
k_s_1 = 2 * 10**(-4)
k_s_2 = 2 * 10**(-5)
k_food = 10**(-2)
k_drag = 10**(-2)
visc = 2 * 10**(-2)
radius = surface_height / 64
fiber_length = 10 * radius
omega_flow = 10**(-6)
k_flow = 10**(-3)
p_plank = 10**(-4)
D_chan = 0.3
T = 1000
grid_rows = 16
grid_cols = 25
K_d = 0.1

g_syn = 10**3
t_rise, t_fast, t_slow = 3, 0.5, 10
a_decay = 0.5
E_syn = 0
alpha_dec = 3

block_size = surface_height / 16
max_W = 1000
P_max = 1
P_half = 0.5
P_border = 0
nutricity = 1
uptake_rate = 10**(-2)
prod_rate = 10**(-2) / 2
D = 2 * 10**(-2)
uptake_death_rate = 10**(-3)
D_cell = 1
N_arc = 10
N_rad = 3
k_chan = 0.1
theta_rate = 0.2
k_cilia = 1


"""
Distance between two 2D points.
"""
def dist(x_1, y_1, x_2, y_2):
    return ((x_1 - x_2)**2 + (y_1 - y_2)**2)**(1/2)


"""
Angle betwen horizontal axis and (v_x, v_y) vector.
"""
def get_angle(v_x, v_y):
    if v_y >= 0:
        return math.acos(v_x / (v_x**2 + v_y**2)**(1/2))
    else:
        return -math.acos(v_x / (v_x**2 + v_y**2)**(1/2))


"""
Excitatory postsynaptic current.
"""
def I_EPSC(t, V):
    if t < 0: return 0
    return g_syn * (1 - exp(- t / t_rise)) * (a_decay * exp(- t / t_fast) + (1 - a_decay) * exp(- t / t_slow)) * max(E_syn - V, 0)


"""
Draw square grid of bacterial mat density.
"""
def draw_grid(P, surface):
    for i in range(0, grid_rows):
        for j in range(0, grid_cols):
            y = i * block_size
            x = (surface_width - grid_cols * block_size) / 2 + j * block_size
            rect = pygame.Rect(x, y, block_size, block_size)
            pygame.draw.rect(surface, (163 * P[i][j], 177 * P[i][j], 138 * P[i][j]), rect)


"""
Returns center of masses for the cell system.
"""
def get_CoM(cells):
    c_x, c_y = 0, 0
    for cell in cells:
        c_x, c_y = c_x + cell.x, c_y + cell.y
    c_x, c_y = c_x / len(cells), c_y / len(cells)
    return c_x, c_y


"""
Health bar class. Rectangle with position and size.
The maximum number of health points is fixed.
"""
class HealthBar():
  def __init__(self, x, y, w, h, max_hp):
    self.x = x
    self.y = y
    self.w = w
    self.h = h
    self.hp = max_hp
    self.max_hp = max_hp

  def draw(self, surface):
    ratio = self.hp / self.max_hp
    pygame.draw.rect(surface, "#e63946", (self.x, self.y, self.w, self.h))
    pygame.draw.rect(surface, "#ccd5ae", (self.x, self.y, self.w * ratio, self.h))


"""
Cell class.
"""
class Cell:
    def __init__(self, x, y, v_x, v_y, V):
        self.x, self.y = x, y
        self.v_x, self.v_y = v_x, v_y
        self.p_x, self.p_y = 0, 0
        self.V = V
        self.a_x, self.a_y = 0, 0
        self.theta = 0
        self.activated_channels = set()
        self.channel_timers = dict()
        # FUTURE self.A, self.R, self.I = set(), set(), set()
        self.v_cilia = 0


def simulate(N_rad, k_a, k_b, W):
    """
    Preparing the surface.
    """


    # Initiating of the creature's health bar.
    health_bar = HealthBar(surface_width * 700 / 1024, surface_height * 20 / 640, surface_width * 300 / 1024, surface_height * 40 / 640, max_W)


    # Generating a random initial distribution of plankton on the grid.
    P = np.zeros((grid_rows, grid_cols))
    for i in range(grid_rows):
        for j in range(grid_cols):
            P[i][j] = random.random()


    # Initiation of a creature.
    cells = []
    for i in range(N_arc):
        for j in range(N_rad):
            cells.append(Cell(
                surface_width / 2 + (surface_height / 320 + surface_height / 320 * j) * radius * math.cos(2 * math.pi * i / N_arc),
                surface_height / 2 + (surface_height / 320 + surface_height / 320 * j) * radius * math.sin(2 * math.pi * i / N_arc),
                0,
                0,
                0
            ))


    # Setting initial polarizations of the cells.
    c_x, c_y = get_CoM(cells)
    for cell in cells:
        d_c = dist(cell.x, cell.y, c_x, c_y)
        n_x = (cell.x - c_x) / d_c
        n_y = (cell.y - c_y) / d_c
        if not (n_x == n_y == 0):
            cell.p_x = n_x / (n_x**2 + n_y**2)**(1/2)
            cell.p_y = n_y / (n_x**2 + n_y**2)**(1/2)
            cell.theta = get_angle(cell.p_x, cell.p_y)


    # Establishment of connections between adjacent cells.
    fibers = set()
    d, dist_prev = dict(), dict()
    n = N_arc * N_rad
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if dist(cells[i].x, cells[i].y, cells[j].x, cells[j].y) < fiber_length:
                fibers.add((i, j))
                d[(i, j)] = dist(cells[i].x, cells[i].y, cells[j].x, cells[j].y)
                dist_prev[(i, j)] = d[(i, j)]


    """
    Stepwise simulation.
    """


    # Counters.
    t, t_alive, V_sum, W_sum = 0, 0, 0, 0
    # Simulation state.
    active = True
    while active:
        # Quit simulation if user suspended it.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                active = False

        # Timer tick update.
        t += 1
        t_alive += 1
        if t > T:
            t = 0


        # Calculating center of masses.
        c_x, c_y = get_CoM(cells)


        Z = np.zeros((grid_rows, grid_cols))
        uptake = 0

        for i in range(n):
            cell = cells[i]


            # Viscosity forces.
            cell.a_x = -visc * cell.v_x
            cell.a_y = -visc * cell.v_y


            # Chemoattractant gradient.
            j_grid = math.floor(cell.x / block_size)
            i_grid = math.floor((cell.y - (surface_width - grid_cols * block_size) / 2) / block_size)
            if 0 <= i_grid < grid_rows and 0 <= j_grid < grid_cols:
                Z[i_grid][j_grid] += 1
                # Food adsorption.
                uptake += nutricity * uptake_rate * P[i_grid][j_grid]**2 / (P_half**2 + P[i_grid][j_grid]**2)
                c = P[i_grid][j_grid]
            else:
                c = 0
            # Changing membrane receptors binding states.
            n_x, n_y = 0, 0
            cell.V = c / (c + K_d)
            d_c = dist(cell.x, cell.y, c_x, c_y)
            n_x = (cell.x - c_x) / d_c
            n_y = (cell.y - c_y) / d_c
            if not (n_x == n_y == 0):
                cell.p_x = n_x / (n_x**2 + n_y**2)**(1/2) * cell.V * k_food
                cell.p_y = n_y / (n_x**2 + n_y**2)**(1/2) * cell.V * k_food


            # Interactions with other cells.
            S_i = cell.V
            n_adj = 0
            for j in range(n):
                if (i, j) not in fibers: continue
                if j in cell.activated_channels:
                    S_i += k_chan *  I_EPSC(cell.channel_timers[j], - cells[j].V + cell.V)
                    n_adj += 1
            S_i /= (n_adj + 1)
            for j in range(n):
                if (i, j) not in fibers: continue

                x_1, y_1 = cell.x, cell.y
                x_2, y_2 = cells[j].x, cells[j].y
                delta_x, delta_y = x_2 - x_1, y_2 - y_1
                n_x, n_y = delta_x / (delta_x**2 + delta_y**2)**(1/2), delta_y / (delta_x**2 + delta_y**2)**(1/2)


                # Channel transmissions.
                if (cells[j].V > cell.V) and (j not in cell.activated_channels):
                    cell.activated_channels.add(j)
                    cell.channel_timers[j] = 0

                if j in cell.activated_channels:
                    l = dist(x_1, y_1, x_2, y_2) / 2
                    S_ij = k_chan * I_EPSC(cell.channel_timers[j], - cells[j].V + cell.V)
                    B = (k_a * S_ij + 2 * D_cell * k_a * S_i / k_b / l) / (k_b + 2 * D_cell / l)
                    cell.p_x += D_chan * math.cos(cells[j].theta) * B
                    cell.p_y += D_chan * math.sin(cells[j].theta) * B
                    cell.channel_timers[j] += 1
                    if cell.channel_timers[j] == alpha_dec * t_slow:
                        cell.activated_channels.remove(j)
                        del cell.channel_timers[j]


                # Resulting change in the direction of polarization.
                if cell.p_x**2 + cell.p_y**2 > 0:
                    n_p_x = cell.p_x / dist(0, 0, cell.p_x, cell.p_y)
                    n_p_y = cell.p_y / dist(0, 0, cell.p_x, cell.p_y)
                else:
                    n_p_x = math.cos(cell.theta)
                    n_p_y = math.sin(cell.theta)
                theta_fin = get_angle(n_p_x, n_p_y)
                cell.theta += math.sin(theta_fin - cell.theta) * theta_rate
                while cell.theta < 0:
                    cell.theta += 2 * math.pi
                while cell.theta > 2 * math.pi:
                    cell.theta -= 2 * math.pi
                radial_angle = get_angle(n_x, n_y)
                cell.a_x += k_drag * n_x * math.cos(cell.theta - radial_angle)
                cell.a_y += k_drag * n_y * math.cos(cell.theta - radial_angle)
                cell.v_cilia = k_drag * math.cos(cell.theta - radial_angle)


                # Elasticity forces.
                delta_r = dist(x_1, y_1, x_2, y_2) - d[(i, j)]
                dr_dt = dist(x_1, y_1, x_2, y_2) - dist_prev[(i, j)]

                cells[i].a_x += k_s_1 * delta_r * n_x
                cells[i].a_y += k_s_1 * delta_r * n_y

                cells[i].a_x += k_s_2 * dr_dt * n_x
                cells[i].a_y += k_s_2 * dr_dt * n_y


        # Plankton dynamics.
        for i in range(grid_rows):
            for j in range(grid_cols):
                if 1 <= i <= grid_rows - 2:
                    diff_term_y = P[i + 1][j] - 2 * P[i][j] + P[i - 1][j]
                elif i == 0:
                    diff_term_y = P[i + 1][j] - 2 * P[i][j] + P_border
                elif i == grid_rows - 1:
                    diff_term_y = P_border - 2 * P[i][j] + P[i - 1][j]
                if 1 <= j <= grid_cols - 2:
                    diff_term_x = P[i][j + 1] - 2 * P[i][j] + P[i][j - 1]
                elif j == 0:
                    diff_term_x = P[i][j + 1] - 2 * P[i][j] + P_border
                elif j == grid_cols - 1:
                    diff_term_x = P_border - 2 * P[i][j] + P[i][j - 1]
                diff_term = D * (diff_term_y + diff_term_x)
                prod_term = (1 + math.sin(2 * math.pi * t / T - 2 * math.pi * j / grid_cols)) / 2 * prod_rate * P[i][j] * (1 - P[i][j] / P_max)
                uptake_term = uptake_death_rate * Z[i][j] * P[i][j]**2 / (P_half**2 + P[i][j]**2)
                P[i][j] += diff_term + prod_term - uptake_term

        draw_grid(P, surface)


        # Update of creature mechanical and electrochemical states.
        V_sum_prev = V_sum
        V_sum, P_cilia = 0, 0

        for i in range(len(cells)):
            cell = cells[i]
            cell.v_x += cell.a_x * dt
            cell.v_y += cell.a_y * dt
            cell.x += cell.v_x * dt
            cell.y += cell.v_y * dt
            V_sum += cell.V
            P_cilia += k_cilia * cell.v_cilia**2
            for j in range(len(cells)):
                if (i, j) not in fibers: continue
                dist_prev[(i, j)] = dist(cell.x, cell.y, cells[j].x, cells[j].y)


        # Drawing fibers.
        for i in range(n):
            for j in range(n):
                if (i, j) not in fibers: continue
                x_1, y_1, x_2, y_2 = cells[i].x, cells[i].y, cells[j].x, cells[j].y
                pygame.draw.line(
                    surface,
                    "#ffcdb2",
                    (x_1, y_1),
                    (x_2, y_2),
                    width = int(radius / 2)
                )


        # Drawing cells.
        for cell in cells:
            col_1 = tuple(255 * c for c in matplotlib.colors.to_rgb("#90e0ef"))
            col_2 = tuple(255 * c for c in matplotlib.colors.to_rgb("#fb6f92"))
            phase_color = (1 + math.cos(cell.theta)) / 2
            col = (
                (1 - phase_color) * col_1[0] + phase_color * col_2[0],
                (1 - phase_color) * col_1[1] + phase_color * col_2[1],
                (1 - phase_color) * col_1[2] + phase_color * col_2[2]
            )

            pygame.draw.circle(
                surface,
                col,
                (cell.x, cell.y),
                radius
            )


        # Update of creature's remaining energy.
        W += phi * uptake - omega * V_sum - pump_cost * (V_sum - V_sum_prev) - h * N_arc * N_rad - P_cilia
        W = min(max_W, W)
        W_sum += W

        if W < 0 or (statistics == "W_average" and t_alive == T_sim):
            print("k_a", k_a, "k_b", k_b)
            if statistics == "time_alive":
                print("life time", t_alive)
                return t_alive
            elif statistics == "W_average":
                print("W", W_sum / T_sim)
                return W_sum / T_sim

        health_bar.hp = W
        health_bar.draw(surface)


        pygame.display.update()


#setting_list = [(0.001, 1), (0.01, 1), (0.1, 1), (1, 1), (10, 1), (100, 1), (1000, 1)]
par_list, lifetime_list, W_average_list = [], [], []
setting_list = [1, 2, 3, 4, 5]
k_a, k_b = 0.1, 1
for setting in setting_list:
    #k_a, k_b = setting[0], setting[1]
    #par_list.append(math.log10(k_a / k_b))
    print("N_rad =", setting)
    par_list.append(setting)
    if statistics == "time_alive":
        lifetime_list.append(simulate(setting, k_a, k_b, W))
    else:
        W_average_list.append(simulate(setting, k_a, k_b, W))

#plot.xlabel(r"$log_{10}(k_a\ /\ k_b)$")
plot.xlabel(r"$N_r$")
if statistics == "time_alive":
    plot.scatter(par_list, lifetime_list, marker = "s", color = "black", s = 24)
    plot.ylabel(r"$t_{alive}$")
    print(lifetime_list)
elif statistics == "W_average":
    plot.scatter(par_list, W_average_list, marker = "s", color = "black", s = 24)
    plot.ylabel(r"$\langle W \rangle$")
    print(W_average_list)
plot.savefig("figures/act inh.png")
