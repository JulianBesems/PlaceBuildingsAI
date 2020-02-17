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
                colour: Optional[Tuple[int, int, int]] = None,
                blocks: Optional[List[Block]] = None):
        self.nr = nr
        self.value = value
        if colour:
            self.colour = colour
        else:
            self.colour = [int((1 - value) * 255), 0, int(value * 255)]
        self.blocks = blocks

    def add_block(self, block):
        block.value = self.value
        block.colour = self.colour
        if self.blocks:
            block.nr = len(self.blocks)
            self.blocks.append(block)
        else:
            block.nr = 0
            self.blocks = [block]


class Puzzle(Individual):
    def __init__(self, board_size: Tuple[int, int],
                 nrGroups: int,
                 nrBlocks: int,
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 start_pos: Optional[Point] = None,
                 hidden_layer_architecture: Optional[List[int]] = [1123125, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 groups: Optional[List[Group]] = None):

        self.board_size = board_size
        self.nrGroups = nrGroups
        self.nrBlocks = nrBlocks
        self.nrEmptiesS = (self.board_size[0] - 2) * (self.board_size[1] - 2) - (self.nrBlocks + 1)
        self.nrEmpties = (self.board_size[0] - 2) * (self.board_size[1] - 2) - (self.nrBlocks + 1)
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
        num_inputs = 4 + 24 #@TODO: Add one-hot back in
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


        if not start_pos:
            if settings["fillingOrder"].lower() == 'linear':
                x = 0
                y = 1
            else:
                x = int(self.board_size[0]/2)
                y = int(self.board_size[1]/2)
            start_pos = Point(x, y)

        self.target_pos = (int(self.board_size[0]/2), int(self.board_size[1]/2))
        self.start_pos = start_pos

        self.init_puzzle()
        self.generate_groups()
        self.grDictFilled = {}
        for k in list(self.groups.keys()):
            self.grDictFilled[self.groups[k].value] = []

    def init_puzzle(self) -> None:
        filled = []
        self.filled_array = filled
        self.board = np.empty([self.board_size[0], self.board_size[1]], Block)
        self.unfilledCells = []
        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                self.unfilledCells.append(Point(r,c))
        random.shuffle(self.unfilledCells)

    def generate_groups(self):
        if not self.groups:
            self.groups = {}
            for i in range(self.nrGroups):
                v = random.randint(0, 1000) / 1000
                c = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
                n = Group(i, v, c)
                self.groups[v] = n
            self.nrGroups = len(self.groups)
            for j in range(self.nrBlocks):
                r = random.randint(0, self.nrGroups - 1)
                b = Block()
                i = list(self.groups.keys())[r]
                self.groups[i].add_block(b)

            for k in list(self.groups.keys()):
                if not self.groups[k].blocks:
                    self.groups.pop(k)

        self.initialGroups = copy.deepcopy(self.groups)
        self.nrGroups = len(self.groups)
        self.groupsLeft = len(self.groups)
        self.blocksLeft = self.nrBlocks + 0

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitnessBeta(self):
        if len(self.groups):
            self._fitness = self.blocksLeft
            return True

        distSet = self.settings["distanceSetting"].lower()
        coherencies = []
        distances = []
        values = list(self.grDictFilled.keys())
        for v in values:
            blocks = self.grDictFilled[v]
            nrBlocks = len(blocks)
            minDistance = np.inf
            innerDistance = 0
            psi = 0.650245952951
            sumOptimal = psi * (nrBlocks ** (5/2))/2 + 0.115 * (nrBlocks**(3/2))
            for p in blocks:
                if distSet == "manhattan":
                    dist= abs(p.x - self.target_pos.x)  + abs(p.y - self.target_pos.y)
                elif distSet == "euclidean":
                    dist = math.sqrt(abs(p.x - self.target_pos.x)**2  + abs(p.y - self.target_pos.y)**2)
                if dist < minDistance:
                    minDistance = dist
                for b in blocks:
                    innerDistance += self.cityDistance(p,b)
            innerDistance = 0.5 * innerDistance
            coherencies.append(max(innerDistance/sumOptimal, 1))
            distances.append(minDistance)

            """pairs = list(itertools.combinations(range(nrBlocks), 2))
            for p in pairs:
                a = blocks[p[0]]
                b = blocks[p[1]]
                dx = abs(a.x - b.x)
                dy = abs(a.x - b.x)
                if distSet == "manhattan":
                    innerDistances.append(ax + by)
                if distSet == "euclidean":
                    innerDistances.append(math.sqrt(dx**2 + dy**2))"""


        maxDist = max(distances)
        minDist = min(distances)
        maxVal = max(values)
        minVal = min(values)
        valDiff = maxVal - minVal
        distDiff = maxDist - minDist

        distScores = []
        for i in range(len(values)):
            targetDist = (values[i] - minVal)/valDiff * distDiff + minDist
            deviance = abs(targetDist - distances[i])
            distScores.append(deviance/targetDist)

        fitnessScore = 0
        for i in range(len(coherencies)):
            fitness = distScores[i] * coherencies[i] + 5 * (coherencies[i] - 1)
            fitnessScore += fitness
        fitnessScore = fitnessScore / len(distances)
        self._fitness = fitnessScore

    def calculate_fitness(self):
        if len(self.groups):
            self._fitness = self.blocksLeft * 10
            return True

        distSet = self.settings["distanceSetting"].lower()
        coherencies = []
        distancesMin = []
        distancesAvg = []
        values = list(self.grDictFilled.keys())
        for v in values:
            blocks = self.grDictFilled[v]
            nrBlocks = len(blocks)
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
                    dx = abs(a.x - b.x)
                    dy = abs(a.y - b.y)
                    if distSet == "manhattan":
                        innerDistances.append(dx + dy)
                        sumInD += dx+dy
                    if distSet == "euclidean":
                        innerDistances.append(math.sqrt(dx**2 + dy**2))
                        sumInD += math.sqrt(dx**2 + dy**2)

                coherencies.append(sumInD/sumOptimal)
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
            fitnessScore += dist + coherency - 1
            distScore += dist
        self._fitness = fitnessScore / len(coherencies)
        return self._fitness
        print(distScore/len(coherencies))

    def cityDistance(self, p1, p2):
        dx = abs(p1.x - p2.x)
        dy = abs(p1.y - p2.y)
        d = dx + dy
        if dx:
            d += 1/3
        if dy:
            d += 1/3
        return d

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def look(self, pos):
        array = self.input_values_as_array

        array[0] = self.groupsLeft/self.nrGroups
        array[1] = self.blocksLeft / self.nrBlocks
        #array.append(self.board_size[0])
        #array.append(self.board_size[1])

        array[2] = (abs(self.target_pos[0] - pos.x) + abs(self.target_pos[1] - pos.y)) / (math.sqrt(self.nrBlocks) + 1)
        #array.append(pos.x)
        #array.append(pos.y)
        #array[3] = self.blocksLeft
        #array[4] = self.groupsLeft
        array[3] = self.nrEmpties / self.nrEmptiesS


        populated = []
        values = []
        BlocksRemaining = []
        #nrBlocksPlaced = []

        for i in range(pos.x-1, pos.x+2):
            for j in range(pos.y-1, pos.y+2):
                if not (i == pos.x and j == pos.y):
                    try:
                        block = self.board[i][j]
                    except IndexError:
                        block = None

                    if block and block.value:
                        populated.append(1)
                        values.append(block.value)
                        try:
                            g = self.groups[block.value]
                            gs = self.initialGroups[block.value]
                            BlocksRemaining.append(len(g.blocks)/len(gs.blocks))
                        except KeyError:
                            BlocksRemaining.append(0)
                        #nrBlocksPlaced.append(len(self.grDictFilled[block.value]))
                    else:
                        populated.append(0)
                        values.append(0)
                        BlocksRemaining.append(0)
                        #nrBlocksPlaced.append(-1)

        index = 4
        for i in range(len(values)):
            array[index] = populated[i]
            array[index + 8] = values[i]
            array[index + 16] = BlocksRemaining[i]
            #array[index + 16] = nrBlocksPlaced[i]
            index +=1


    def update(self):
        if self.finished:
            return False
        else:
            return True

    def fill(self) -> bool:
        if self.finished:
            return False

        #check start
        if self.filled_array:
            pos = self.filled_array[-1]
        else:
            pos = Point(self.start_pos.x, self.start_pos.y)
            self.board[pos.x][pos.y] = Block(x = pos.x, y = pos.y)
            #return(self.board[pos.x][pos.y])

        if settings["fillingOrder"].lower() == "linear":
            if pos.x < self.board_size[0]-2:
                self.placeCell(Point(pos.x+1, pos.y))
                return(self.board[pos.x+1][pos.y])
            else:
                self.placeCell(Point(1, pos.y + 1))
                return(self.board[1][pos.y+1])

        elif settings["fillingOrder"].lower() == "spiral":
            if not self.filled_array:
                self.placeCell(Point(pos.x-1, pos.y))
                return(self.board[pos.x-1][pos.y])
            elif self.board[pos.x][pos.y+1] and not self.board[pos.x-1][pos.y]:
                self.placeCell(Point(pos.x-1, pos.y))
                return(self.board[pos.x-1][pos.y])
            elif self.board[pos.x+1][pos.y]:
                self.placeCell(Point(pos.x, pos.y+1))
                return(self.board[pos.x][pos.y+1])
            elif self.board[pos.x][pos.y-1]:
                self.placeCell(Point(pos.x+1, pos.y))
                return(self.board[pos.x+1][pos.y])
            elif self.board[pos.x-1][pos.y]:
                self.placeCell(Point(pos.x, pos.y-1))
                return(self.board[pos.x][pos.y-1])
            elif self.board[pos.x][pos.y-1]:
                self.placeCell(Point(pos.x+1, pos.y))
                return(self.board[pos.x+1][pos.y])

        elif settings["fillingOrder"].lower() == "random":
            i = random.randint(0,len(self.unfilledCells)-1)
            pos = self.unfilledCells.pop()
            self.placeCell(pos)
            return(self.board[pos.x][pos.y])


    def placeCell(self, p):
        self.look(p)
        self.network.feed_forward(self.input_values_as_array)
        output = self.network.out
        placeBlock = int(round(output[1][0], 0)) #random.randint(0,2)

        if (not placeBlock) and self.nrEmpties:
            self.filled_array.append(p)
            self.board[p.x][p.y] = Block(x = p.x, y = p.y)
            self.nrEmpties -= 1
        elif p.x == self.target_pos[0] and p.y == self.target_pos[1]:
            self.filled_array.append(p)
        else:
            self.filled_array.append(p)
            #i = random.randint(0, len(self.groups) - 1)
            #i = int(round(output[0][0] * (len(self.groups) - 1), 0))
            v = output[0][0]
            vd = 10
            i = None
            for k in list(self.groups.keys()):
                if abs(v - k) < vd:
                    vd = abs(v - k)
                    i = k
            g = self.groups[i]
            b = g.blocks.pop()
            self.grDictFilled[g.value].append(p)
            self.board[p.x][p.y] = b
            b.x = p.x
            b.y = p.y
            self.blocksLeft -=1
            if not g.blocks or len(g.blocks) == 0:
                self.groups.pop(i)
                self.groupsLeft -=1
            if len(self.groups) == 0:
                self.finished = True

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
    constructor['start_pos'] = puzzle.start_pos.to_dict()
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
