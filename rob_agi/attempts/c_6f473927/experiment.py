from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6f473927.main import solve_6f473927

def run_experiment():
    input_grid = ColoredGrid(values=[
        [0, 2, 0, 0, 2, 0],
        [0, 2, 2, 2, 0, 0],
        [0, 0, 2, 0, 0, 0]
    ])
    
    result = solve_6f473927(input_grid)
    
    print("Input Grid:")
    print(input_grid)
    print("\nOutput Grid:")
    print(result)

if __name__ == "__main__":
    run_experiment()
