from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from run_type import RunType


def argmax(arr, comp=lambda x, y: x < y):
    max_item = None
    max_index = -1
    for index, item in enumerate(arr):
        if max_item is None or comp(item, max_item):
            max_item = item
            max_index = index
    return max_index


class Wpa(object):
    def __init__(
            self,
            n: int,
            dims: int,
            function: Callable[[ndarray], float],
            initial_wolves: Optional[ndarray] = None,
            global_step: float = 0.08,
            h_min: int = 2,
            h_max: int = 10,
            k_max: int = 50,
            t_max: int = 8,
            beta: int = 2,
            precision: float = 1e-4,
            near_distance: float = 0.2,
            distance_function: Callable[[ndarray, ndarray], float] = lambda a, b: np.linalg.norm(a - b),
            run_type: RunType = RunType.min,
    ):
        self.n = n
        self.dims = dims
        self.function = function
        self.h_min = h_min
        self.h_max = h_max

        if initial_wolves is None:
            initial_wolves = np.array([n * np.random.random(self.dims) for _ in range(self.n)])
        initial_wolves_shape = initial_wolves.shape
        assert len(initial_wolves_shape) == 2 and initial_wolves_shape[0] == self.n \
               and initial_wolves_shape[1] == self.dims
        self.initial_wolves = initial_wolves

        self.step_a = global_step
        self.step_b = global_step * 2
        self.step_c = global_step / 2

        self.k_max = k_max
        self.t_max = t_max
        self.beta = beta

        self.near_distance = near_distance
        self.distance_function = distance_function

        self.renewal_coeff = np.random.randint(n // (2 * self.beta), n // self.beta)

        self.precision = precision

        self.run_type = run_type
        if self.run_type is RunType.min:
            self.comp = lambda x, y: x < y
            self.comp_eq = lambda x, y: x <= y
        else:
            self.comp = lambda x, y: x > y
            self.comp_eq = lambda x, y: x >= y

    def run(self):
        wolves = np.copy(self.initial_wolves)
        current_max = None
        lead_wolf = None
        for _ in range(self.k_max):
            while True:
                lead_wolf_index = argmax(wolves, lambda x, y: self.comp(self.function(x), self.function(y)))
                lead_wolf = wolves[lead_wolf_index]
                lead_wolf_y = self.function(lead_wolf)

                for wolf_index in range(self.n):
                    if wolf_index == lead_wolf_index:
                        continue
                    new_wolf, new_wolf_y = self.__scouting_behaviour(wolves[wolf_index], lead_wolf_y)
                    wolves[wolf_index] = new_wolf
                    if self.comp(new_wolf_y, lead_wolf_y):
                        lead_wolf = new_wolf
                        lead_wolf_index = wolf_index
                        lead_wolf_y = new_wolf_y

                need_retry_scouting = False
                for wolf_index in range(self.n):
                    if wolf_index == lead_wolf_index:
                        continue
                    new_wolf, new_wolf_y = self.__calling_behaviour(wolves[wolf_index], lead_wolf)
                    wolves[wolf_index] = new_wolf
                    if self.comp_eq(new_wolf_y, lead_wolf_y):
                        lead_wolf = new_wolf
                        lead_wolf_index = wolf_index
                        lead_wolf_y = new_wolf_y
                        need_retry_scouting = True

                if not need_retry_scouting:
                    break

            for wolf_index in range(self.n):
                if wolf_index == lead_wolf_index:
                    continue
                wolves[wolf_index] = self.__besieging_behaviour_step(wolves[wolf_index], lead_wolf)

            lead_wolf_index = argmax(wolves, lambda x, y: self.comp(self.function(x), self.function(y)))
            lead_wolf = wolves[lead_wolf_index]
            lead_wolf_y = self.function(lead_wolf)
            for renew_ind in self.__minimum_wolves_indexes(wolves):
                wolves[renew_ind] = self.__renew_position(lead_wolf)

            if current_max is not None and np.abs(current_max - lead_wolf_y) < self.precision:
                print("Finish:", lead_wolf_y, lead_wolf)
                return lead_wolf_y, lead_wolf
            current_max = lead_wolf_y
            print(lead_wolf_y, lead_wolf)
        return current_max, lead_wolf

    def __scouting_behaviour_step(self, wolf: ndarray) -> Tuple[ndarray, float]:
        h = np.random.randint(self.h_min, self.h_max + 1)
        position_candidates = [
            np.array(
                [wolf[index] + np.sin(2 * np.pi * p / h) * self.step_a for index in range(self.dims)])
            for p in range(h)
        ]
        position_candidates_y = [
            self.function(position_candidate) for position_candidate in position_candidates
        ]
        if self.run_type is RunType.max:
            best_candidate_index = np.argmax(position_candidates_y)
        else:
            best_candidate_index = np.argmin(position_candidates_y)
        return position_candidates[best_candidate_index], position_candidates_y[best_candidate_index]

    def __scouting_behaviour(self, wolf: ndarray, lead_wolf_y: float) -> Tuple[ndarray, float]:
        new_wolf, new_wolf_y = wolf, self.function(wolf)
        for _ in range(self.t_max):
            new_wolf, new_wolf_y = self.__scouting_behaviour_step(wolf)
            if self.comp(new_wolf_y, lead_wolf_y):
                return new_wolf, new_wolf_y
        return new_wolf, new_wolf_y

    def __calling_behaviour_step(self, wolf: ndarray, lead_wolf: ndarray) -> ndarray:
        return np.array(
            [wolf[index] + self.step_b * (lead_wolf[index] - wolf[index])
             / np.abs(lead_wolf[index] - wolf[index])
             for index in range(self.dims)]
        )

    def __calling_behaviour(self, wolf: ndarray, lead_wolf: ndarray) -> Tuple[ndarray, float]:
        lead_wolf_y = self.function(lead_wolf)
        new_wolf = wolf

        while self.distance_function(new_wolf, lead_wolf) >= self.near_distance:
            new_wolf = self.__calling_behaviour_step(new_wolf, lead_wolf)
            new_wolf_y = self.function(new_wolf)
            if self.comp_eq(new_wolf_y, lead_wolf_y):
                return new_wolf, new_wolf_y
        return new_wolf, self.function(new_wolf)

    def __besieging_behaviour_step(self, wolf: ndarray, lead_wolf: ndarray) -> ndarray:
        llambda = np.random.uniform(-1, 1)
        return np.array(
            [
                wolf[index] + llambda * self.step_c * np.abs(lead_wolf[index] - wolf[index])
                for index in range(len(wolf))
            ]
        )

    def __renew_position(self, lead_wolf: ndarray) -> ndarray:
        coeffs = [np.random.uniform(-0.1, 0.1) for _ in range(self.dims)]
        return np.array([coeffs[i] * lead_wolf[i] for i in range(self.dims)])

    def __minimum_wolves_indexes(self, wolves: ndarray) -> ndarray:
        values = np.array([self.function(wolf) for wolf in wolves])
        indices = np.argsort(values)
        if self.run_type is RunType.min:
            return indices[-self.renewal_coeff:]
        else:
            return indices[:self.renewal_coeff]
