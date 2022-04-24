import numpy as np
from geopy import distance


class SimulatedAnnealing:
    def __init__(self, cities, annealing_rate, T):
        self.primary_cities = cities
        self.annealing_rate = annealing_rate
        self.T = T

    def _dist(self, c1, c2):
        return distance.geodesic(c1['coordinates'], c2['coordinates']).km

    def _shuffle_cities(self):
        return np.random.permutation(self.primary_cities)

    def calculate_distance(self, cities):
        d = sum([self._dist(cities[i - 1], cities[i]) for i in range(1, len(cities))])
        d += self._dist(cities[0], cities[-1])
        return d

    def _generate_candidates(self, cities):
        ids = sorted(np.random.choice(len(cities), 2, replace=False))
        new_cities = []
        for i in range(ids[0]):
            new_cities.append(cities[i])
        for i in range(ids[1], ids[0] - 1, -1):
            new_cities.append(cities[i])
        for i in range(ids[1] + 1, len(cities)):
            new_cities.append(cities[i])
        return new_cities

    def _next_state(self, cities):
        candidates = self._generate_candidates(cities)
        prev_dist = self.calculate_distance(cities)
        new_dist = self.calculate_distance(candidates)
        alpha = np.exp((prev_dist - new_dist) / (self.T + 1e-14))
        p = np.random.rand(1)[0]
        if p <= alpha:
            return candidates
        else:
            return cities

    def anneal(self):
        state = self._shuffle_cities()
        sequence = [state]
        while self.T > 1:
            state = self._next_state(state)
            sequence.append(state)
            self.T = self.T * (1 - self.annealing_rate)
        return sequence
