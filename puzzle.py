import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
from fractions import Fraction
import random
from collections import deque
import sys
import os
import json
import itertools
import copy, pickle, math


from misc import *
from genetic_algorithm.individual import Individual
from settings import settings
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name

with open ("lowTownValues.p", 'rb') as fp:
            lowTownValues = pickle.load(fp)

with open ("functionTownValues.p", 'rb') as fp:
            functionTownValues = pickle.load(fp)

class Block:
    def __init__(self,
                nr: Optional[int] = None,
                value: Optional[float] = None,
                colour: Optional[Tuple[int, int, int]] = [0,0,0],
                x: Optional[float] = None,
                y: Optional[float] = None):
        self.nr = nr
        self.value = value
        self.colour = colour
        self.x = x
        self.y = y

class Group:
    def __init__(self, nr: int,
                value: float,
                density: float,
                colour: Optional[Tuple[int, int, int]] = None,
                blocks: Optional[List[Block]] = None):
        self.nr = nr
        self.value = value
        self.density = density
        if colour:
            self.colour = colour
        else:
            self.colour = [int((1 - value) * 255), 0, int(value * 255)]
        self.blocksLeft = blocks
        self.blocksPlaced = []
        self.averageX = 0
        self.averageY = 0

    def add_block(self, block):
        block.value = self.value
        block.colour = self.colour
        if self.blocksLeft:
            block.nr = len(self.blocksLeft)
            self.blocksLeft.append(block)
        else:
            block.nr = 0
            self.blocksLeft = [block]


class Puzzle(Individual):
    def __init__(self, board_size: Tuple[int, int],
                 nrGroups: int,
                 nrBlocks: int,
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 hidden_layer_architecture: Optional[List[int]] = [20, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 groups: Optional[List[Group]] = None):

        self.board_size = board_size
        self.nrGroups = nrGroups
        self.nrBlocks = nrBlocks

        self.finished = False
        self.settings = settings

        self._fitness = 1000
        self.groups = groups

        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = 14 #@TODO: Add one-hot back in
        self.input_values_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(2)                               # 4 outputs, ['u', 'd', 'l', 'r']
        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation))

        if chromosome:
            self.network.params = chromosome
        else:
            pass

        self.target_pos = (int(self.board_size[0]/2), int(self.board_size[1]/2))

        self.generate_groups()

    def generate_groups(self):
        self.blocks = []
        if not self.groups:
            self.groups = {}
            for i in range(self.nrGroups):
                v = random.randint(0, 1000) / 1000
                c = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
                d = random.randint(1, self.settings["density_max"])
                n = Group(i, v, d, c)
                self.groups[v] = n
            self.nrGroups = len(self.groups)
            for j in range(self.nrBlocks):
                r = random.randint(0, self.nrGroups - 1)
                b = Block()
                i = list(self.groups.keys())[r]
                self.groups[i].add_block(b)
                self.blocks.append(b)

            for k in list(self.groups.keys()):
                if not self.groups[k].blocksLeft:
                    self.groups.pop(k)

        self.nrGroups = len(self.groups)
        self.groupsLeft = len(self.groups)
        random.shuffle(self.blocks)

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self):
        if len(self.blocks):
            self._fitness = 10000
            return True

        distSet = self.settings["distanceSetting"].lower()
        coherencies = []
        distancesMin = []
        distancesAvg = []

        values = list(self.groups.keys())

        for v in values:
            blocks = self.groups[v].blocksPlaced
            nrBlocks = len(blocks)

            density = self.groups[v].density

            minDistance = np.inf
            sumDistance = 0
            sumInD = 0
            innerDistances = []

            if nrBlocks <= lowTownValues[0][-1]:
                sumOptimal = lowTownValues[1][nrBlocks - 1]
            else:
                a,b,c,d = functionTownValues[0]
                sumOptimal = int(a*(nrBlocks**b) + c * (nrBlocks**d))

            if nrBlocks > 1:
                pairs = list(itertools.combinations(range(nrBlocks), 2))
                for p in pairs:
                    a = blocks[p[0]]
                    b = blocks[p[1]]
                    dx = abs(a.x - b.x)/density
                    dy = abs(a.y - b.y)/density
                    if distSet == "manhattan":
                        innerDistances.append(dx + dy)
                        sumInD += dx+dy
                    if distSet == "euclidean":
                        innerDistances.append(math.sqrt(dx**2 + dy**2))
                        sumInD += math.sqrt(dx**2 + dy**2)

                coherencies.append(abs(sumInD - sumOptimal)/sumOptimal)
            else:
                coherencies.append(1)

            for b in blocks:
                d = abs(b.x - self.target_pos[0]) + abs(b.y - self.target_pos[1])
                sumDistance += d
                if d < minDistance:
                    minDistance = d
            distancesMin.append(minDistance)
            distancesAvg.append(sumDistance/nrBlocks)

        maxDist = math.sqrt(self.nrBlocks) + 1
        minDist = 1
        maxVal = max(values)
        minVal = min(values)
        valDiff = maxVal - minVal
        distDiff = maxDist - minDist

        distScores = []
        for i in range(len(values)):
            targetDist = values[i] * distDiff + minDist
            deviance = abs(targetDist - distancesAvg[i])
            distScores.append(deviance/targetDist)

        fitnessScore = 0
        distScore = 0
        for i in range(len(coherencies)):
            coherency = coherencies[i]
            dist = distScores[i]
            #print(dist, coherency)
            fitnessScore += dist + (coherency -1) **2 - 1
            distScore += dist
        self._fitness = fitnessScore / len(coherencies)
        return self._fitness

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def look(self, cell):
        array = self.input_values_as_array

        array[0] = self.target_pos[0] / self.board_size[0]
        array[1] = self.target_pos[1] / self.board_size[1]

        array[2] = self.groupsLeft/self.nrGroups
        array[3] = len(self.blocks) / self.nrBlocks
        array[4] = len(self.groups[cell.value].blocksLeft)/(len(self.groups[cell.value].blocksLeft) + len(self.groups[cell.value].blocksPlaced))

        array[5] = self.groups[cell.value].density / self.board_size[0]
        array[6] = self.groups[cell.value].density / self.board_size[1]

        array[7] = self.groups[cell.value].averageX
        array[8] = self.groups[cell.value].averageY

        array[9] = cell.value

        ul = []
        ur = []
        dl = []
        dr = []
        for b in self.groups[cell.value].blocksPlaced:
            if b.x > array[7]:
                if b.y> array[8]:
                    dr.append(b)
                else:
                    ur.append(b)
            else:
                if b.y > array[8]:
                    dl.append(b)
                else:
                    ul.append(b)

        array[10] = len(ul)
        array[11] = len(ur)
        array[12] = len(dl)
        array[13] = len(dr)

    def update(self):
        if self.finished:
            return False
        else:
            return True

    def fill(self) -> bool:
        if self.finished:
            return False
        b = self.blocks.pop()
        self.placeCell(b)
        if not self.blocks:
            self.finished = True
        return b


    def placeCell(self, c):
        self.look(c)
        self.network.feed_forward(self.input_values_as_array)
        output = self.network.out
        x = output[0][0]
        y = output[1][0]
        g = self.groups[c.value]
        c.x = x * self.board_size[0]
        c.y = y * self.board_size[1]
        if g.averageX > 0:
            g.averageX = (g.averageX * len(g.blocksPlaced) + c.x) / len(g.blocksPlaced)
            g.averageY = (g.averageY * len(g.blocksPlaced) + c.y) / len(g.blocksPlaced)
        else:
            g.averageX = c.x
            g.averageY = c.y

        g.blocksLeft.remove(c)
        g.blocksPlaced.append(c)


def save_puzzle(population_folder: str, individual_name: str, puzzle: Puzzle, settings: Dict[str, Any]) -> None:
    # Make population folder if it doesn't exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Save off settings
    if 'settings.json' not in os.listdir(population_folder):
        f = os.path.join(population_folder, 'settings.json')
        with open(f, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    # Make directory for the individual
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    # Save some constructor information for replay
    # @NOTE: No need to save chromosome since that is saved as .npy
    # @NOTE: No need to save board_size or hidden_layer_architecture
    #        since these are taken from settings
    constructor = {}
    constructor['nrGroups'] = puzzle.nrGroups
    constructor['nrBlocks'] = puzzle.nrBlocks
    #constructor['groups'] = puzzle.initialGroups
    puzzle_constructor_file = os.path.join(individual_dir, 'constructor_params.json')

    # Save
    with open(puzzle_constructor_file, 'w', encoding='utf-8') as out:
        json.dump(constructor, out, sort_keys=True, indent=4)

    L = len(puzzle.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = puzzle.network.params[w_name]
        bias = puzzle.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)

def load_puzzle(population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Puzzle:
    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")

        with open(f, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    elif isinstance(settings, dict):
        settings = settings

    elif isinstance(settings, str):
        filepath = settings
        with open(filepath, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    # Load constructor params for the specific snake
    constructor_params = {}
    puzzle_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(puzzle_constructor_file, 'r', encoding='utf-8') as fp:
        constructor_params = json.load(fp)

    puzzle = Puzzle(settings['board_size'], constructor_params['nrGroups'],
                  constructor_params['nrBlocks'],
                  chromosome=params,
                  hidden_layer_architecture=settings['hidden_network_architecture'],
                  hidden_activation=settings['hidden_layer_activation'],
                  output_activation=settings['output_layer_activation'],
                  #groups = constructor_params['groups']
                  )
    return puzzle
