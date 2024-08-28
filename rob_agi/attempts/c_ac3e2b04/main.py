from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a symmetrical
    blue structure extending from green squares and connecting red lines.

    1. Identifies green squares (3x3 areas with a red center)
    2. Creates initial blue structure based on green squares and red lines
    3. Extends blue lines to connect green squares and red lines
    4. Ensures symmetry across both vertical and horizontal axes
    5. Fills gaps and ensures continuity of blue structures
    6. Preserves original red and green cells

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    green_squares = find_green_squares(output_grid)
    
    create_initial_blue_structure(output_grid, green_squares)
    extend_blue_structure(output_grid)
    ensure_symmetry(output_grid)
    fill_gaps(output_grid)
    preserve_original_colors(output_grid, input_grid)
    
    return output_grid

def find_green_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    green_squares = []
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid.get_cell(r, c) == 2 and
                all(grid.get_cell(r+dr, c+dc) == 3 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)])):
                green_squares.append((r, c))
    return green_squares

def create_initial_blue_structure(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r, c in green_squares:
        # Extend blue lines to nearest red lines or edges
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols:
                if grid.get_cell(nr, nc) == 2:  # Stop at red line
                    break
                if grid.get_cell(nr, nc) == 0:
                    grid.set_cell(nr, nc, 1)
                nr, nc = nr + dr, nc + dc

def ensure_symmetry(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                if grid.get_cell(r, cols - 1 - c) == 0:
                    grid.set_cell(r, cols - 1 - c, 1)
                if grid.get_cell(rows - 1 - r, c) == 0:
                    grid.set_cell(rows - 1 - r, c, 1)
                if grid.get_cell(rows - 1 - r, cols - 1 - c) == 0:
                    grid.set_cell(rows - 1 - r, cols - 1 - c, 1)

def extend_blue_structure(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                # Extend horizontally
                left = next((i for i in range(c-1, -1, -1) if grid.get_cell(r, i) in [1, 2, 3]), -1)
                right = next((i for i in range(c+1, cols) if grid.get_cell(r, i) in [1, 2, 3]), cols)
                for i in range(left+1, right):
                    if grid.get_cell(r, i) == 0:
                        grid.set_cell(r, i, 1)
                
                # Extend vertically
                top = next((i for i in range(r-1, -1, -1) if grid.get_cell(i, c) in [1, 2, 3]), -1)
                bottom = next((i for i in range(r+1, rows) if grid.get_cell(i, c) in [1, 2, 3]), rows)
                for i in range(top+1, bottom):
                    if grid.get_cell(i, c) == 0:
                        grid.set_cell(i, c, 1)

def fill_gaps(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)] 
                                if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 1)
                if neighbors >= 2:
                    grid.set_cell(r, c, 1)

def handle_intersections(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                blue_neighbors = [n for n in neighbors if 0 <= n[0] < rows and 0 <= n[1] < cols and grid.get_cell(n[0], n[1]) == 1]
                if len(blue_neighbors) >= 2:
                    for nr, nc in neighbors:
                        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                            grid.set_cell(nr, nc, 1)

def final_symmetry_check(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                grid.set_cell(rows - 1 - r, c, 1)
                grid.set_cell(r, cols - 1 - c, 1)
                grid.set_cell(rows - 1 - r, cols - 1 - c, 1)

def preserve_original_colors(output_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))
def connect_blue_networks(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        blue_cells = [c for c in range(cols) if grid.get_cell(r, c) == 1]
        if len(blue_cells) > 1:
            for c in range(min(blue_cells), max(blue_cells) + 1):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)
    
    for c in range(cols):
        blue_cells = [r for r in range(rows) if grid.get_cell(r, c) == 1]
        if len(blue_cells) > 1:
            for r in range(min(blue_cells), max(blue_cells) + 1):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)
