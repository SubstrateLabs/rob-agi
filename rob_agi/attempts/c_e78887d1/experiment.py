from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e78887d1.main import solve_e78887d1

def run_experiment():
    # Test case from example_0
    input_grid = ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 2, 0, 0, 1, 0, 0, 3, 3, 3],
        [2, 0, 2, 0, 1, 1, 1, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 1, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 1, 1, 1, 0, 3, 0, 3],
        [2, 2, 2, 0, 0, 0, 0, 0, 3, 0, 3],
        [0, 2, 0, 0, 1, 1, 1, 0, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 0, 1, 0, 1, 0, 0, 3, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 3, 3, 3],
        [2, 2, 2, 0, 1, 1, 1, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    result = solve_e78887d1(input_grid)
    print("Input grid:")
    print(input_grid)
    print("\nOutput grid:")
    print(result)

if __name__ == "__main__":
    run_experiment()
