from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0d87d2a6.main import solve_0d87d2a6

def run_experiment():
    # Test case 1: Simple grid with two blue dots
    input_grid1 = ColoredGrid(values=[
        [1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    result1 = solve_0d87d2a6(input_grid1)
    print("Test case 1 result:")
    print(result1)

    # Test case 2: Grid with blue dots on edges
    input_grid2 = ColoredGrid(values=[
        [1, 0, 0, 0, 1],
        [0, 0, 2, 0, 0],
        [1, 0, 0, 0, 1]
    ])
    result2 = solve_0d87d2a6(input_grid2)
    print("\nTest case 2 result:")
    print(result2)

    # Test case 3: Complex grid with multiple blue dots and red blocks
    input_grid3 = ColoredGrid(values=[
        [0, 0, 1, 0, 0, 0],
        [0, 2, 0, 0, 2, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 2, 0, 0, 2, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    result3 = solve_0d87d2a6(input_grid3)
    print("\nTest case 3 result:")
    print(result3)

if __name__ == "__main__":
    run_experiment()
