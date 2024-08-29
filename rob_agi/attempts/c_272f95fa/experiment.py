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
            else:
                row += "* "
        print(row)
    
    print("\nHorizontal dividers:", h_dividers)
    print("Vertical dividers:", v_dividers)

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
