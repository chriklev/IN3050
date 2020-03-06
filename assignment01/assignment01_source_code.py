import time
import pandas as pd
import itertools as it
import sys
import numpy as np

city_distances = pd.read_csv('european_cities.csv', sep=';').to_numpy()


def get_shortest_permutation(cities, weights=city_distances):

    smallest_distance = 999999
    shortest_permutation = None

    for perm in it.permutations(cities):

        distance = weights[perm[0], perm[-1]]

        for i in range(1, len(perm)):
            diff = weights[perm[i-1], perm[i]]
            distance += diff

        if distance < smallest_distance:
            smallest_distance = distance
            shortest_permutation = perm

    return shortest_permutation, smallest_distance


if __name__ == "__main__":
    city_indices = range(int(sys.argv[1]))
    t0 = time.time()
    perm, dist = get_shortest_permutation(city_indices, city_distances)
    t = time.time() - t0
    print("Shortest path: ", perm)
    print("Shortest distance: ", dist)
    print("Time: ", t)
