from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_19bb5feb.main import solve_19bb5feb

def create_test_grid(colors):
    grid = [[8 for _ in range(10)] for _ in range(10)]
    for i, color in enumerate(colors):
        row = (i // 2) * 3
        col = (i % 2) * 3
        grid[row][col] = grid[row][col+1] = grid[row+1][col] = grid[row+1][col+1] = color
    return ColoredGrid(values=grid)

def run_experiment():
    test_cases = [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [5, 4, 3, 2, 1]
    ]

    for colors in test_cases:
        input_grid = create_test_grid(colors)
        output_grid = solve_19bb5feb(input_grid)
        print(f"Input colors: {colors}")
        print(f"Output grid:\n{output_grid.values}")
        print("---")

if __name__ == "__main__":
    run_experiment()
