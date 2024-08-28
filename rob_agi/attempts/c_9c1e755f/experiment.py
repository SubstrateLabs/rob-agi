from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_9c1e755f.main import solve_9c1e755f

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

def run_experiment():
    # Test case 1: Conflicting patterns from left and top edges
    input_grid1 = ColoredGrid(values=[
        [1, 1, 1, 0, 0],
        [2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0]
    ])
    
    print("Test case 1 - Input:")
    print_grid(input_grid1)
    
    output_grid1 = solve_9c1e755f(input_grid1)
    
    print("Test case 1 - Output:")
    print_grid(output_grid1)
    
    # Test case 2: Pattern wrapping around grid boundary
    input_grid2 = ColoredGrid(values=[
        [1, 2, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4]
    ])
    
    print("Test case 2 - Input:")
    print_grid(input_grid2)
    
    output_grid2 = solve_9c1e755f(input_grid2)
    
    print("Test case 2 - Output:")
    print_grid(output_grid2)

if __name__ == "__main__":
    run_experiment()
