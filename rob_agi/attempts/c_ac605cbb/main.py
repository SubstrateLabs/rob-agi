from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac605cbb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Expands colored cells (6, 3, 2, 1) in specific patterns.
    2. Magenta (6) and Green (3) expand vertically to the grid edges.
    3. Red (2) expands horizontally to the grid edges.
    4. Blue (1) creates an L-shape by moving diagonally and then expanding.
    5. Uses gray (5) for expansions and connections.
    6. Uses yellow (4) for diagonal connections and conflict resolution.
    7. Maintains color hierarchy and creates connections between expansions.
    8. Balances the grid and fills isolated areas.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find colored cells
    colored_cells = [(r, c, output_grid.get_cell(r, c)) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    colored_cells.sort(key=lambda x: x[2], reverse=True)  # Sort by color priority
    
    # Primary expansion
    for r, c, color in colored_cells:
        if color == 6 or color == 3:  # Magenta or Green
            expand_vertical(output_grid, r, c, color)
        elif color == 2:  # Red
            expand_horizontal(output_grid, r, c, color)
        elif color == 1:  # Blue
            expand_blue(output_grid, r, c)
    
    # Secondary connections and territory definition
    connect_expansions(output_grid)
    
    # Balancing and filling
    fill_gaps(output_grid)
    
    # Pattern completion and conflict resolution
    complete_patterns(output_grid)
    
    return output_grid

def expand_vertical(grid: ColoredGrid, r: int, c: int, color: int):
    rows, _ = grid.get_dimensions()
    for i in range(rows):
        if i == r:
            continue
        if grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, 5)
        elif grid.get_cell(i, c) != color:
            break
    grid.set_cell(0, c, color)
    grid.set_cell(rows-1, c, color)

def expand_horizontal(grid: ColoredGrid, r: int, c: int, color: int):
    _, cols = grid.get_dimensions()
    for j in range(cols):
        if j == c:
            continue
        if grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, 5)
        elif grid.get_cell(r, j) != color:
            break
    grid.set_cell(r, 0, color)
    grid.set_cell(r, cols-1, color)

def expand_blue(grid: ColoredGrid, r: int, c: int):
    rows, cols = grid.get_dimensions()
    new_r, new_c = r-1, c+1  # Move diagonally up and right
    if new_r < 0 or new_c >= cols:
        new_r, new_c = r+1, c-1  # Move diagonally down and left if out of bounds
    
    if grid.get_cell(new_r, new_c) == 0:
        grid.set_cell(new_r, new_c, 1)
        expand_horizontal(grid, new_r, new_c, 1)
        expand_vertical(grid, new_r, new_c, 1)

def connect_expansions(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r1 in range(rows):
        for c1 in range(cols):
            if grid.get_cell(r1, c1) not in [0, 5]:
                for r2 in range(rows):
                    for c2 in range(cols):
                        if grid.get_cell(r2, c2) not in [0, 5] and (r1, c1) != (r2, c2):
                            if r1 == r2 or c1 == c2:
                                connect_orthogonal(grid, r1, c1, r2, c2)
                            else:
                                connect_diagonal(grid, r1, c1, r2, c2)

def connect_orthogonal(grid: ColoredGrid, r1: int, c1: int, r2: int, c2: int):
    if r1 == r2:
        for c in range(min(c1, c2)+1, max(c1, c2)):
            if grid.get_cell(r1, c) == 0:
                grid.set_cell(r1, c, 5)
    else:
        for r in range(min(r1, r2)+1, max(r1, r2)):
            if grid.get_cell(r, c1) == 0:
                grid.set_cell(r, c1, 5)

def connect_diagonal(grid: ColoredGrid, r1: int, c1: int, r2: int, c2: int):
    r, c = r1, c1
    while r != r2 and c != c2:
        r += 1 if r2 > r1 else -1
        c += 1 if c2 > c1 else -1
        if grid.get_cell(r, c) == 0:
            grid.set_cell(r, c, 4)

def fill_gaps(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) != 0)
                if neighbors >= 2:
                    grid.set_cell(r, c, 5)

def complete_patterns(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for color in [6, 3, 2, 1]:
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == color:
                    if color in [6, 3]:
                        complete_vertical(grid, r, c, color)
                    elif color == 2:
                        complete_horizontal(grid, r, c, color)
                    elif color == 1:
                        complete_blue(grid, r, c)

def complete_vertical(grid: ColoredGrid, r: int, c: int, color: int):
    rows, _ = grid.get_dimensions()
    for i in range(r-1, -1, -1):
        if grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, 5)
        elif grid.get_cell(i, c) != color and grid.get_cell(i, c) != 5:
            break
    for i in range(r+1, rows):
        if grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, 5)
        elif grid.get_cell(i, c) != color and grid.get_cell(i, c) != 5:
            break

def complete_horizontal(grid: ColoredGrid, r: int, c: int, color: int):
    _, cols = grid.get_dimensions()
    for j in range(c-1, -1, -1):
        if grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, 5)
        elif grid.get_cell(r, j) != color and grid.get_cell(r, j) != 5:
            break
    for j in range(c+1, cols):
        if grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, 5)
        elif grid.get_cell(r, j) != color and grid.get_cell(r, j) != 5:
            break

def complete_blue(grid: ColoredGrid, r: int, c: int):
    rows, cols = grid.get_dimensions()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
            grid.set_cell(nr, nc, 5)
