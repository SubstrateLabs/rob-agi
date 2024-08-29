from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_623ea044.main import solve_623ea044

def print_grid(grid):
    for row in grid.values:
        print(''.join(str(cell) for cell in row))
    print()

def run_test_case(grid, description):
    print(f"Test case: {description}")
    print("Input:")
    print_grid(grid)
    print("Output:")
    print_grid(solve_623ea044(grid))
    print()

# Test case 1: 7x7 grid with colored cell in the center
input_grid1 = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

# Test case 2: 15x15 grid with colored cell off-center
input_grid2 = ColoredGrid(values=[[0] * 15 for _ in range(15)])
input_grid2.set_cell(5, 11, 7)

# Test case 3: 10x10 grid with colored cell in the corner
input_grid3 = ColoredGrid(values=[[0] * 10 for _ in range(10)])
input_grid3.set_cell(0, 0, 2)

# Test case 4: 5x5 grid with colored cell near the edge
input_grid4 = ColoredGrid(values=[[0] * 5 for _ in range(5)])
input_grid4.set_cell(1, 3, 4)

run_test_case(input_grid1, "7x7 grid with colored cell in the center")
run_test_case(input_grid2, "15x15 grid with colored cell off-center")
run_test_case(input_grid3, "10x10 grid with colored cell in the corner")
run_test_case(input_grid4, "5x5 grid with colored cell near the edge")
