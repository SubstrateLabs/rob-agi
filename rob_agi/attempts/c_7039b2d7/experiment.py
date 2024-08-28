import time
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_7039b2d7.main import solve_7039b2d7

def generate_grid(size):
    return ColoredGrid(values=[[5 if (i % 4 != 0 and j % 4 != 0) else 3 for j in range(size)] for i in range(size)])

def run_experiment():
    sizes = [5, 10, 15, 20, 25, 30]
    for size in sizes:
        grid = generate_grid(size)
        start_time = time.time()
        solve_7039b2d7(grid)
        end_time = time.time()
        print(f"Grid size: {size}x{size}, Time taken: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    run_experiment()
