from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_67c52801.main import identify_color_groups, solve_67c52801

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

def experiment():
    input_grid = ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 6, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
        [6, 6, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    print("Input Grid:")
    print_grid(input_grid)

    color_groups = identify_color_groups(input_grid)
    print("Identified Color Groups:")
    for i, group in enumerate(color_groups):
        print(f"Group {i + 1}: Size = {len(group)}, Color = {group[0][2]}, Positions = {group}")

    output_grid = solve_67c52801(input_grid)
    print("\nOutput Grid:")
    print_grid(output_grid)

if __name__ == "__main__":
    experiment()
