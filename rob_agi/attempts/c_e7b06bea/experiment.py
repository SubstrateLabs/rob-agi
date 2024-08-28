from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e7b06bea.main import solve_e7b06bea

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

def run_experiment():
    # Test case with even number of columns
    input_grid_even = ColoredGrid(values=[
        [5, 0, 0, 3, 1, 2],
        [5, 0, 0, 3, 1, 2],
        [0, 0, 0, 3, 1, 2],
        [0, 0, 0, 3, 1, 2]
    ])

    # Test case with odd number of columns
    input_grid_odd = ColoredGrid(values=[
        [5, 0, 0, 3, 1],
        [5, 0, 0, 3, 1],
        [0, 0, 0, 3, 1],
        [0, 0, 0, 3, 1]
    ])

    print("Input grid (even columns):")
    print_grid(input_grid_even)
    result_even = solve_e7b06bea(input_grid_even)
    print("Output grid (even columns):")
    print_grid(result_even)

    print("Input grid (odd columns):")
    print_grid(input_grid_odd)
    result_odd = solve_e7b06bea(input_grid_odd)
    print("Output grid (odd columns):")
    print_grid(result_odd)

if __name__ == "__main__":
    run_experiment()
