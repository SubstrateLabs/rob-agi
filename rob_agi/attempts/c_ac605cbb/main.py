from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac605cbb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Expands colored cells (6, 3, 2, 1) in specific patterns.
    2. Magenta (6) and Green (3) expand vertically.
    3. Red (2) expands horizontally.
    4. Blue (1) expands based on its position relative to other colors.
    5. Uses gray (5) for pattern borders and filling.
    6. Maintains color hierarchy and creates connections between expansions.
    7. Balances the grid and fills isolated areas.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find colored cells
    colored_cells = [(r, c, input_grid.get_cell(r, c)) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) != 0]
    colored_cells.sort(key=lambda x: x[2], reverse=True)  # Sort by color priority
    
    for r, c, color in colored_cells:
        if color == 6 or color == 3:  # Magenta or Green
            expand_vertical(output_grid, r, c, color)
        elif color == 2:  # Red
            expand_horizontal(output_grid, r, c, color)
        elif color == 1:  # Blue
            expand_blue(output_grid, r, c)
    
    # Connect expansions and fill gaps
    connect_expansions(output_grid)
    fill_gaps(output_grid)
    
    return output_grid

def expand_vertical(grid: ColoredGrid, r: int, c: int, color: int):
    rows, _ = grid.get_dimensions()
    for i in range(r-1, -1, -1):  # Expand upwards
        if grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, 5)
        else:
            break
    grid.set_cell(r-1, c, color)  # Set the top cell to the original color
    
    for i in range(r+1, rows):  # Expand downwards
        if grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, 5)
        else:
            break
    grid.set_cell(r+1, c, color)  # Set the bottom cell to the original color

def expand_horizontal(grid: ColoredGrid, r: int, c: int, color: int):
    _, cols = grid.get_dimensions()
    for j in range(c-1, -1, -1):  # Expand left
        if grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, 5)
        else:
            break
    grid.set_cell(r, c-1, color)  # Set the leftmost cell to the original color
    
    for j in range(c+1, cols):  # Expand right
        if grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, 5)
        else:
            break
    grid.set_cell(r, c+1, color)  # Set the rightmost cell to the original color

def expand_blue(grid: ColoredGrid, r: int, c: int):
    rows, cols = grid.get_dimensions()
    if r < rows // 2:  # Upper half of the grid
        expand_horizontal(grid, r, c, 1)
    else:  # Lower half of the grid
        expand_vertical(grid, r, c, 1)

def connect_expansions(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0 and grid.get_cell(r, c) != 5:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid.get_cell(nr, nc) != 0 and grid.get_cell(nr, nc) != 5:
                            if abs(dr) + abs(dc) == 2:  # Diagonal
                                grid.set_cell((r+nr)//2, (c+nc)//2, 4)  # Yellow connection
                            else:  # Orthogonal
                                grid.set_cell(nr, nc, 5)  # Gray connection

def fill_gaps(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) != 0)
                if neighbors >= 2:
                    grid.set_cell(r, c, 5)
