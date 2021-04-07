from math import sqrt
from random import randint


def generateARandomPermutation(n):
    perm = [i for i in range(n)]
    pos1 = randint(0, n - 1)
    pos2 = randint(0, n - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    return perm


def get_dist(a, b):
    return sqrt(((a[0] - b[0]) * (a[0] - b[0])) + (
                (a[1] - b[1]) * (a[1] - b[1])))

