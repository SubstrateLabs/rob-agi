from rob_agi.colored_grid import ColoredGrid
from main import solve_20981f0e

def test_case(input_grid):
    result = solve_20981f0e(input_grid)
    print("Input:")
    print(input_grid)
    print("\nOutput:")
    print(result)
    print("\n" + "="*50 + "\n")

# Test case 1: Simple case with one section
test_case_1 = ColoredGrid(values=[
    [0, 2, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0]
])

# Test case 2: Multiple sections with different shapes
test_case_2 = ColoredGrid(values=[
    [0, 2, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 2, 0, 0, 0, 2, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0]
])

test_case(test_case_1)
test_case(test_case_2)
