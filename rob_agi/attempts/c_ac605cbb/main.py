from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac605cbb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Expands colored cells (6, 3, 2, 1) in specific patterns.
    2. Magenta (6) and Green (3) expand vertically.
    3. Red (2) expands horizontally.
    4. Blue (1) expands in a cross shape, favoring vertical expansion.
    5. Uses gray (5) for pattern borders and filling.
    6. Maintains color hierarchy and symmetry.
    7. Creates a connected structure expanding towards the center.
    8. Handles intersections and special cases.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find colored cells
    colored_cells = [(r, c, input_grid.get_cell(r, c)) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) != 0]
    colored_cells.sort(key=lambda x: x[2], reverse=True)  # Sort by color priority
    
    center_r, center_c = rows // 2, cols // 2
    
    for r, c, color in colored_cells:
        if color == 6 or color == 3:  # Magenta or Green
            expand_vertical(output_grid, r, c, center_r, color)
        elif color == 2:  # Red
            expand_horizontal(output_grid, r, c, center_c, color)
        elif color == 1:  # Blue
            expand_cross(output_grid, r, c, center_r, center_c, color)
    
    # Create frame and fill gaps
    create_frame(output_grid)
    fill_gaps(output_grid)
    
    return output_grid

def expand_vertical(grid: ColoredGrid, r: int, c: int, center_r: int, color: int):
    rows, _ = grid.get_dimensions()
    direction = 1 if r < center_r else -1
    for i in range(r, center_r + direction, direction):
        if grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, color if abs(i - r) <= 1 else 5)
        else:
            break

def expand_horizontal(grid: ColoredGrid, r: int, c: int, center_c: int, color: int):
    _, cols = grid.get_dimensions()
    direction = 1 if c < center_c else -1
    for j in range(c, center_c + direction, direction):
        if grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, color if abs(j - c) <= 2 else 5)
        else:
            break

def expand_cross(grid: ColoredGrid, r: int, c: int, center_r: int, center_c: int, color: int):
    expand_vertical(grid, r, c, center_r, color)
    expand_horizontal(grid, r, c, center_c, color)

def create_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                        grid.set_cell(nr, nc, 5)

def fill_gaps(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) != 0)
                if neighbors >= 2:
                    grid.set_cell(r, c, 5)

def expand_vertical(grid: ColoredGrid, r: int, c: int, length: int, color: int):
    rows, _ = grid.get_dimensions()
    for i in range(max(0, r - length), min(rows, r + length + 1)):
        if i == r:
            grid.set_cell(i, c, color)
        elif grid.get_cell(i, c) == 0:
            grid.set_cell(i, c, color if abs(i - r) <= length // 2 else 5)
        if c > 0 and grid.get_cell(i, c-1) == 0:
            grid.set_cell(i, c-1, 5)
        if c < grid.get_dimensions()[1] - 1 and grid.get_cell(i, c+1) == 0:
            grid.set_cell(i, c+1, 5)

def expand_cross(grid: ColoredGrid, r: int, c: int, length: int, color: int, primary: str):
    if primary == 'horizontal':
        expand_horizontal(grid, r, c, length, color)
        expand_vertical(grid, r, c, length // 2, color)
    else:
        expand_vertical(grid, r, c, length, color)
        expand_horizontal(grid, r, c, length // 2, color)

def expand_horizontal(grid: ColoredGrid, r: int, c: int, length: int, color: int):
    _, cols = grid.get_dimensions()
    for j in range(max(0, c - length), min(cols, c + length + 1)):
        if j == c:
            grid.set_cell(r, j, color)
        elif grid.get_cell(r, j) == 0:
            grid.set_cell(r, j, color if abs(j - c) <= length // 2 else 5)
        if r > 0 and grid.get_cell(r-1, j) == 0:
            grid.set_cell(r-1, j, 5)
        if r < grid.get_dimensions()[0] - 1 and grid.get_cell(r+1, j) == 0:
            grid.set_cell(r+1, j, 5)

def handle_intersections(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0 and grid.get_cell(r, c) != 5:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid.get_cell(nr, nc) != 0 and grid.get_cell(nr, nc) != 5:
                            if grid.get_cell(r, c) != grid.get_cell(nr, nc):
                                grid.set_cell(nr, nc, min(6, max(grid.get_cell(r, c), grid.get_cell(nr, nc)) + 1))

def fill_gaps(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) != 0)
                if neighbors >= 2:
                    grid.set_cell(r, c, 5)
