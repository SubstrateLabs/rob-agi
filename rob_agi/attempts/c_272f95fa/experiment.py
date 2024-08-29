from rob_agi.colored_grid import ColoredGrid

def find_dividers(grid):
    horizontal = [i for i, row in enumerate(grid.values) if all(cell == 8 for cell in row)]
    vertical = [j for j in range(len(grid.values[0])) if all(row[j] == 8 for row in grid.values)]
    return horizontal, vertical

def visualize_grid(grid):
    rows, cols = grid.get_dimensions()
    h_dividers, v_dividers = find_dividers(grid)
    
    print("Grid structure:")
    for i in range(rows):
        row = ""
        for j in range(cols):
            if i in h_dividers or j in v_dividers:
                row += "8 "
            elif i == 0 or i == rows - 1 or j <= 1 or j >= cols - 2:
                row += "0 "
            elif i < h_dividers[0]:
                row += "T "  # Top section
            elif i > h_dividers[-1]:
                row += "B "  # Bottom section
            elif j < v_dividers[0]:
                row += "L "  # Left section
            elif j > v_dividers[-1]:
                row += "R "  # Right section
            else:
                row += "M "  # Middle section
        print(row)
    
    print("\nHorizontal dividers:", h_dividers)
    print("Vertical dividers:", v_dividers)

def analyze_last_columns(grid):
    rows, cols = grid.get_dimensions()
    _, v_dividers = find_dividers(grid)
    
    print("\nLast two columns analysis:")
    for i in range(rows):
        for j in range(cols - 2, cols):
            value = grid.get_cell(i, j)
            if value == 0:
                print(f"Cell ({i}, {j}): Preserve 0")
            elif j > v_dividers[-1]:
                print(f"Cell ({i}, {j}): Fill with color (current value: {value})")
            else:
                print(f"Cell ({i}, {j}): Part of vertical divider")

# Example grids from the test cases
example_grids = [
    ColoredGrid(values=[
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0]
    ]),
    ColoredGrid(values=[
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0]
    ])
]

for i, grid in enumerate(example_grids):
    print(f"\nExample Grid {i + 1}:")
    visualize_grid(grid)
    analyze_last_columns(grid)
