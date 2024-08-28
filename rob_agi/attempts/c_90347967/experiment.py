from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_90347967.main import solve_90347967

def run_experiment():
    # Create a 30x30 grid with all black cells
    input_grid = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])
    
    # Add some non-black cells to test the rotation and repositioning
    input_grid.values[29][0] = 1  # Bottom-left corner
    input_grid.values[29][29] = 2  # Bottom-right corner
    input_grid.values[0][29] = 3  # Top-right corner
    input_grid.values[0][0] = 4  # Top-left corner
    input_grid.values[15][15] = 5  # Center

    # Solve the grid
    result = solve_90347967(input_grid)

    # Print the result
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result)

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) if cell != 0 else '.' for cell in row))

if __name__ == "__main__":
    run_experiment()
