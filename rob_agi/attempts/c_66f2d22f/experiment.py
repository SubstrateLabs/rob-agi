from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_66f2d22f.main import solve_66f2d22f
import pytest

def calculate_density(grid: ColoredGrid, colors: list) -> float:
    total_cells = grid.get_dimensions()[0] * grid.get_dimensions()[1]
    colored_cells = sum(1 for row in grid.values for cell in row if cell in colors)
    return colored_cells / total_cells

def analyze_density(input_grid: ColoredGrid, output_grid: ColoredGrid):
    input_density = calculate_density(input_grid, [2, 3])
    output_density = calculate_density(output_grid, [5])
    print(f"Input density: {input_density:.4f}")
    print(f"Output density: {output_density:.4f}")
    print(f"Ratio (output/input): {output_density/input_density:.4f}")

def run_experiment():
    test_cases = [
        pytest.lazy_fixture("test_66f2d22f_example_0"),
        pytest.lazy_fixture("test_66f2d22f_example_1"),
        pytest.lazy_fixture("test_66f2d22f_example_2"),
        pytest.lazy_fixture("test_66f2d22f_example_3"),
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nAnalyzing Example {i}:")
        input_grid = test_case.getfixturevalue("input_grid")
        expected_output = test_case.getfixturevalue("expected")
        actual_output = solve_66f2d22f(input_grid)
        
        print("Expected Output:")
        analyze_density(input_grid, expected_output)
        print("Actual Output:")
        analyze_density(input_grid, actual_output)

if __name__ == "__main__":
    run_experiment()
