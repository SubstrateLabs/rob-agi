from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_770cc55f.main import solve_770cc55f

def run_experiment():
    # Test case 1: Single column overlap
    input_grid1 = ColoredGrid(values=[
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 3, 0, 0]
    ])
    result1 = solve_770cc55f(input_grid1)
    print("Test case 1 (Single column overlap):")
    print("Input:")
    print(input_grid1)
    print("Output:")
    print(result1)
    print()

    # Test case 2: No overlap
    input_grid2 = ColoredGrid(values=[
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 3]
    ])
    result2 = solve_770cc55f(input_grid2)
    print("Test case 2 (No overlap):")
    print("Input:")
    print(input_grid2)
    print("Output:")
    print(result2)
    print()

    # Test case 3: No red line
    input_grid3 = ColoredGrid(values=[
        [0, 3, 3, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 3, 3, 0]
    ])
    result3 = solve_770cc55f(input_grid3)
    print("Test case 3 (No red line):")
    print("Input:")
    print(input_grid3)
    print("Output:")
    print(result3)

if __name__ == "__main__":
    run_experiment()
