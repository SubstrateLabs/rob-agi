from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a8610ef7.main import solve_a8610ef7

def visualize_grid(grid):
    color_map = {0: '⬛', 2: '🟥', 5: '⬜', 8: '🟦'}
    return '\n'.join([''.join([color_map[cell] for cell in row]) for row in grid.values])

def run_experiment():
    # Create a test grid
    test_grid = ColoredGrid(values=[
        [8, 8, 8, 0, 0, 8],
        [8, 8, 0, 8, 8, 8],
        [0, 8, 8, 8, 0, 8],
        [8, 0, 8, 8, 8, 0],
        [8, 8, 8, 0, 8, 8],
        [8, 0, 8, 8, 8, 8]
    ])

    print("Input Grid:")
    print(visualize_grid(test_grid))

    result = solve_a8610ef7(test_grid)

    print("\nOutput Grid:")
    print(visualize_grid(result))

    # Check checkerboard pattern
    print("\nCheckerboard Pattern Check:")
    for r in range(len(result.values)):
        for c in range(len(result.values[0])):
            if result.values[r][c] in [2, 5]:
                expected = 5 if (r + c) % 2 == 0 else 2
                if result.values[r][c] != expected:
                    print(f"Mismatch at ({r}, {c}): Expected {expected}, Got {result.values[r][c]}")

    # Check edge detection
    print("\nEdge Detection Check:")
    for r in range(len(test_grid.values)):
        for c in range(len(test_grid.values[0])):
            if test_grid.values[r][c] == 8 and result.values[r][c] == 5:
                is_edge = any(
                    0 <= nr < len(test_grid.values) and 0 <= nc < len(test_grid.values[0]) and test_grid.values[nr][nc] != 8
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                )
                if not is_edge:
                    print(f"Possible incorrect edge detection at ({r}, {c})")

if __name__ == "__main__":
    run_experiment()
