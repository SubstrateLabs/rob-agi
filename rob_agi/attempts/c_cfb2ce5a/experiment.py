from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_cfb2ce5a.main import solve_cfb2ce5a

def run_experiment():
    # Example input grid
    input_grid = ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 2, 1, 0, 0, 0, 8, 0],
        [0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    result = solve_cfb2ce5a(input_grid)

    print("Input Grid:")
    print(input_grid)
    print("\nOutput Grid:")
    print(result)

    # Analyze the result
    color_counts = result.get_color_frequencies()
    print("\nColor frequencies in the output:")
    for color, count in color_counts.items():
        if color != 0:
            print(f"Color {color}: {count}")

    # Check if the border is maintained
    border_maintained = all(result.values[r][c] == 0 for r in [0, 9] for c in range(10)) and \
                        all(result.values[r][c] == 0 for c in [0, 9] for r in range(10))
    print(f"\nBorder maintained: {border_maintained}")

if __name__ == "__main__":
    run_experiment()
