from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b2d4a7c1.main import solve_b2d4a7c1

def run_test_case(input_grid, expected_output, case_number):
    result = solve_b2d4a7c1(ColoredGrid(values=input_grid))
    print(f"Test Case {case_number}:")
    print("Input:")
    for row in input_grid:
        print(row)
    print("\nOutput:")
    for row in result.values:
        print(row)
    print("\nExpected Output:")
    for row in expected_output:
        print(row)
    print("\nResult:", "PASS" if result.values == expected_output else "FAIL")
    print("=" * 50)

# Test Case 1
input_grid_1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]

expected_output_1 = [
    [0, 0, 0, 0, 0, 1, 2, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 2, 1]
]

run_test_case(input_grid_1, expected_output_1, 1)

# Test Case 2
input_grid_2 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
]

expected_output_2 = [
    [0, 0, 0, 0, 0, 2, 3, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 3, 2]
]

run_test_case(input_grid_2, expected_output_2, 2)

# Test Case 3
input_grid_3 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0]
]

expected_output_3 = [
    [0, 0, 0, 0, 0, 3, 4, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 4, 3]
]

run_test_case(input_grid_3, expected_output_3, 3)

# Test Case 4: Empty input grid
input_grid_4 = [[]]
expected_output_4 = [[]]
run_test_case(input_grid_4, expected_output_4, 4)

# Test Case 5: Grid with no anchor value
input_grid_5 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
expected_output_5 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
run_test_case(input_grid_5, expected_output_5, 5)

# Test Case 6: Grid with multiple anchor values
input_grid_6 = [[0, 0, 0], [0, 0, 0], [1, 2, 3]]
expected_output_6 = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
run_test_case(input_grid_6, expected_output_6, 6)

# Test Case 7: Maximum size grid (30x30)
input_grid_7 = [[0] * 30 for _ in range(29)] + [[0] * 29 + [1]]
expected_output_7 = [[0] * 29 + [1] for _ in range(30)]
expected_output_7[0][29] = 2
expected_output_7[-1][29] = 2
run_test_case(input_grid_7, expected_output_7, 7)
