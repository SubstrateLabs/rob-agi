from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0c9aba6e.test import (
    test_0c9aba6e_example_0,
    test_0c9aba6e_example_1,
    test_0c9aba6e_example_2,
    test_0c9aba6e_example_3,
)

def visualize_input_output(input_grid, expected_output):
    print("Input Grid (first 6 rows):")
    for row in input_grid.values[:6]:
        print(''.join(['R' if cell == 2 else '.' for cell in row]))
    print("\nExpected Output:")
    for row in expected_output.values:
        print(''.join(['S' if cell == 8 else '.' for cell in row]))
    print("\n")

def run_experiment():
    examples = [
        test_0c9aba6e_example_0,
        test_0c9aba6e_example_1,
        test_0c9aba6e_example_2,
        test_0c9aba6e_example_3,
    ]

    for i, example in enumerate(examples):
        print(f"Example {i}:")
        input_grid = example.__closure__[0].cell_contents
        expected_output = example.__closure__[1].cell_contents
        visualize_input_output(input_grid, expected_output)

if __name__ == "__main__":
    run_experiment()
