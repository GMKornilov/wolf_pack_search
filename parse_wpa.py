import json
from typing import Callable
import numpy as np
from numpy import ndarray

from run_type import RunType
from wpa import Wpa
import test_functions


def get_function_by_name(function_name: str) -> Callable[[ndarray], float]:
    if function_name == "rosenbrock":
        return test_functions.rosenbrock
    if function_name == "colville":
        return test_functions.colville
    if function_name == "sphere_function":
        return test_functions.sphere_function
    if function_name == "sum_squares_function":
        return test_functions.sum_squares_function
    if function_name == "booth_function":
        return test_functions.booth_function
    return lambda x: 0


def get_run_type_by_name(name: str) -> RunType:
    return RunType.max if name == "max" else RunType.min


__default_global_step: float = 0.08
__default_h_min: int = 2
__default_h_max: int = 10
__default_k_max: int = 50
__default_t_max: int = 8
__default_beta: int = 2
__default_precision: float = 1e-4
__default_near_distance: float = 0.2
__default_distance_function: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.linalg.norm(a - b)
__default_run_type_name: str = "min"


def read_from_file() -> Wpa:
    file = open("input.json")
    json_dict = json.load(file)
    file.close()

    n = json_dict["n"]
    dims = json_dict["dims"]
    function_name = json_dict["function_name"]
    function = get_function_by_name(function_name)

    global_step = json_dict.get("global_step") if json_dict.get("global_step") is not None else __default_global_step
    h_min = json_dict.get("h_min") if json_dict.get("h_min") is not None else __default_h_min
    h_max = json_dict.get("h_max") if json_dict.get("h_max") is not None else __default_h_max
    k_max = json_dict.get("k_max") if json_dict.get("k_max") is not None else __default_k_max
    t_max = json_dict.get("t_max") if json_dict.get("t_max") is not None else __default_t_max
    beta = json_dict.get("beta") if json_dict.get("beta") is not None else __default_beta
    precision = json_dict.get("precision") if json_dict.get("precision") is not None else __default_precision
    near_distance = json_dict.get("near_distance") if json_dict.get("near_distance") is not None \
        else __default_near_distance
    run_type_name = json_dict.get("run_type_name") if json_dict.get("run_type_name") is not None \
        else __default_run_type_name
    run_type = get_run_type_by_name(run_type_name)

    return Wpa(
        n=n,
        dims=dims,
        function=function,
        global_step=global_step,
        h_min=h_min,
        h_max=h_max,
        k_max=k_max,
        t_max=t_max,
        beta=beta,
        precision=precision,
        near_distance=near_distance,
        run_type=run_type,
    )
