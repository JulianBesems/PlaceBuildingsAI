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
                y: Optional[float] = None,
                size: Optional[Tuple[float,float]] = [1,1],
                kind: Optional[List] = "circle"):
        self.nr = nr
        self.value = value
        self.colour = colour
        self.x = x
        self.y = y
        self.size = size
        self.kind = kind

class Group:
    def __init__(self, nr: int,
                value: float,
                board: Any,
                colour: Optional[Tuple[int, int, int]] = None,
                blocks: Optional[List[Block]] = None,
                point: Optional[Point] = None):
        self.nr = nr
        self.value = value
        self.board = board
        if colour:
            self.colour = colour
        else:
            self.colour = [int((1 - value) * 255), 0, int(value * 255)]
        self.blocksLeft = blocks
        self.blocksPlaced = []
        self.point = point
        self.zones = None
        self.board.cells[self.point.x, self.point.y] = Block(self.nr, self.value, self.colour, self.point.x, self.point.y)

    def getZones(self):
        self.zones = self.board.getZones(self.point, len(self.blocksLeft))

    def add_block(self, block):
        block.nr = self.nr
        block.value = self.value
        block.colour = self.colour
        if self.blocksLeft:
            self.blocksLeft.append(block)
        else:
            self.blocksLeft = [block]

    @property
    def finished(self):
        return len(self.blocksLeft) == 0

class Board:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells = np.empty([self.width, self.height], Block)

    def getWallDist(self, point, area):
        pos = point
        p = area
        if p == (-1,-1) or p == (-1,0):
            mr = pos.x + pos.y
        elif p == (0,-1) or p == (1,-1):
            mr = self.width - pos.x + pos.y
        elif p == (-1,1) or p == (0,1):
            mr = pos.x  + self.height - pos.y
        elif p == (1,1) or p == (1,0):
            mr = self.height + self.width - pos.x - pos.y
        else:
            mr = None
        return mr

    def getZones(self, pos: Point, mr = 1000):
        zones = {}
        for i in range(-1,2):
            for j in range(-1,2):
                if not (i==0 and j==0):
                    zones[(i,j)] = self.getZone(pos,(i,j), mr)
        return zones


    def getZone(self, pos: Point, area: Tuple[int,int], mr2 = 1000):
        pos = pos
        p = area

        mr = self.getWallDist(pos,p)

        maxRange = min(mr, mr2)

        cells = {}
        if abs(p[0] + p[1]) == 1:
            for i in range(maxRange):
                n = int(i/2) + 1
                c = []
                for j in range(n):
                    x,y = (p[0] + (p[0] * (i-j) - p[1]*j), p[1] + (p[0]*j + p[1] * (i-j)))
                    px = pos.x + x
                    py = pos.y + y
                    if -1 < px < self.width and -1 < py < self.height:
                        c.append([(px,py), self.cells[px,py]])
                cells[i+1] = c

        elif p[0] == p[1]:
            for i in range(maxRange):
                n = int(i/2) + 1
                c = []
                for j in range(n):
                    x,y = (p[0] + p[0] * j, p[1] + p[1] * (i - j))
                    px = pos.x + x
                    py = pos.y + y
                    if -1 < px < self.width and -1 < py < self.height:
                        c.append([(px,py), self.cells[px,py]])
                cells[i+2] = c

        else:
            for i in range(maxRange):
                n = int(i/2) + 1
                c = []
                for j in range(n):
                    x,y = (p[0] + p[0] * (i - j), p[1] + p[1] * j)
                    px = pos.x + x
                    py = pos.y + y
                    if -1 < px < self.width and -1 < py < self.height:
                        c.append([(px,py), self.cells[px,py]])
                cells[i+2] = c
        return cells


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

        self.failed = False
        self.settings = settings

        self.progress = 0
        self._fitness = 0
        self.groups = groups
        self.finishedGroups = {}
        self.board = Board(self.board_size[0], self.board_size[1])

        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = 24 #@TODO: Add one-hot back in
        self.input_values_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(8)                               # 4 outputs, ['u', 'd', 'l', 'r']
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
        if not self.groups:
            self.groups = {}
            startPoints = []
            for _ in range(int(self.nrGroups/2)):
                added = False
                while not added:
                    p = (random.randint(int(self.board.width * 0.25),int(self.board.width*0.75)), random.randint(int(self.board.height *0.25),int(self.board.height*0.75)))
                    if not (p in startPoints):
                        startPoints.append(p)
                        added = True

            for _ in range(int(self.nrGroups/2), self.nrGroups):
                added = False
                while not added:
                    p = (random.randint(0,self.board.width-1), random.randint(0,self.board.height-1))
                    if not (p in startPoints or (int(self.board.width * 0.25) < p[0] < int(self.board.width*0.75)) or (int(self.board.height * 0.25) < p[1] < int(self.board.height*0.75))):
                        startPoints.append(p)
                        added = True

            for i in range(self.nrGroups):
                v = random.randint(0, 1000) / 1000
                c = (int(v*255), int(v*255), int(v*255) )
                #c = (random.randint(20, 200), random.randint(20, 200), random.randint(20, 200))
                p = Point(startPoints[i][0], startPoints[i][1])
                n = Group(i, v, self.board, c, point = p)
                self.groups[i] = n
            for j in range(self.nrBlocks):
                r = random.randint(0, self.nrGroups-1)
                b = Block()
                i = list(self.groups.keys())[r]
                self.groups[i].add_block(b)

            for k in list(self.groups.keys()):
                if self.groups[k].finished:
                    self.groups.pop(k)
        for g in self.groups:
            self.groups[g].getZones()
        self.nrGroups = len(self.groups)

    @property
    def fitness(self):
        return self._fitness

    @property
    def finished(self):
        for g in self.groups:
            if not self.groups[g].finished:
                return False
        return True


    def calculate_fitness(self):
        self._fitness = (self.progress/self.nrBlocks) * 100
        return self._fitness

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def fill(self) -> bool:
        if self.finished or self.failed:
            return False
        i = self.progress % len(self.groups)
        k = list(self.groups.keys())[i]
        b = self.placeBlock(self.groups[k].blocksLeft[-1])
        if not b:
            self.failed = True
            return False
        self.progress += 1
        return b

    def placeBlock(self, b):
        empties = self.look(b)
        self.network.feed_forward(self.input_values_as_array)
        output = []
        outval = None
        i=0

        # Make it go for secondary choices
        for x in list(self.network.out):
            output.append([i,x])
            i +=1
        output.sort(key = lambda x: x[1])
        found = False

        while output and not found:
            d = output.pop()
            try:
                c = empties[d[0]]
                outval = d[1]
                found = True
            except KeyError:
                pass

        if not found:
            print(empties)
            return None

        # Have it fail if it takes the wrong turn
        """
        d = np.argmax(self.network.out)
        try:
            c = empties[d]
            outval = list(self.network.out)[d]
            found = True
        except KeyError:
            return None"""

        b.x = c[0]
        b.y = c[1]
        b.value = outval
        self.board.cells[c] = b
        self.groups[b.nr].blocksLeft.pop()
        self.groups[b.nr].blocksPlaced.append(b)
        if self.groups[b.nr].finished:
            self.finishedGroups[b.nr] = self.groups[b.nr]
            self.groups.pop(b.nr)
        return b

    def look(self, block):
        array = self.input_values_as_array
        group = self.groups[block.nr]
        views = group.zones

        empties = {}
        i = 0
        for p in views:
            wallDist = self.board.getWallDist(group.point, p)+1
            free = False
            otherDist = False
            pj = 0
            for j in views[p]:
                for c in views[p][j]:
                    if self.board.cells[c[0]] == None:
                        if not free:
                            free = j
                            empties[i] = c[0]
                    if self.board.cells[c[0]] != None and self.board.cells[c[0]].nr != block.nr:
                        if not otherDist:
                            otherDist = j
                    if free and otherDist:
                        break
                if free and otherDist:
                    break

            if free:
                array[i] = 1/free
            else:
                array[i] = 0

            if (wallDist - free):
                array[i + 8] = 1/(wallDist - free)
            else:
                array[i + 8] = 1

            if otherDist:
                array[i + 16] = 1/otherDist
            else:
                array[i + 16] = 0
            i+=1
        if not empties:
            print(views)
        return empties

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
