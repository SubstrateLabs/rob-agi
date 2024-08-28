from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_505fff84.main import solve_505fff84
from rob_agi.attempts.c_505fff84.test import (
    test_505fff84_example_0,
    test_505fff84_example_1,
    test_505fff84_example_2,
    test_505fff84_example_3,
    test_505fff84_example_4,
)
import numpy as np

def analyze_grid(input_grid: ColoredGrid, output_grid: ColoredGrid):
    input_array = np.array(input_grid.values)
    output_array = np.array(output_grid.values)
    
    input_shape = input_array.shape
    output_shape = output_array.shape
    input_red_ratio = np.sum(input_array == 2) / input_array.size
    output_red_ratio = np.sum(output_array == 2) / output_array.size
    
    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
    print(f"Input red ratio: {input_red_ratio:.2f}, Output red ratio: {output_red_ratio:.2f}")
    print(f"Shape ratio: {output_shape[0]/input_shape[0]:.2f} x {output_shape[1]/input_shape[1]:.2f}")
    print("---")

def run_experiment():
    test_cases = [
        test_505fff84_example_0,
        test_505fff84_example_1,
        test_505fff84_example_2,
        test_505fff84_example_3,
        test_505fff84_example_4,
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"Example {i}:")
        input_grid = test_case.__closure__[0].cell_contents
        output_grid = solve_505fff84(input_grid)
        analyze_grid(input_grid, output_grid)

if __name__ == "__main__":
    run_experiment()
