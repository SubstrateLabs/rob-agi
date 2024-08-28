from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5207a7b5.main import solve_5207a7b5

def print_grid(grid):
    for row in grid.values:
        print(''.join(str(cell) for cell in row))
    print()

def run_experiment():
    # Test case 1: Gray line at column 3, length 5
    input_grid1 = ColoredGrid(values=[
        [0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    # Test case 2: Gray line at column 2, length 4
    input_grid2 = ColoredGrid(values=[
        [0, 0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    # Test case 3: Gray line at column 4, length 7
    input_grid3 = ColoredGrid(values=[
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])

    print("Test case 1 output:")
    print_grid(solve_5207a7b5(input_grid1))

    print("Test case 2 output:")
    print_grid(solve_5207a7b5(input_grid2))

    print("Test case 3 output:")
    print_grid(solve_5207a7b5(input_grid3))

if __name__ == "__main__":
    run_experiment()
