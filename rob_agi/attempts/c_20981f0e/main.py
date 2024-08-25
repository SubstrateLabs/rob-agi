from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_20981f0e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Rearrange blue cells (1) in each section between red dot (2) rows to form more organized and connected structures.
    The solution maintains the same number of blue cells in each section and preserves the positions of red dots.
    Steps:
    1. Identify sections between red dot rows
    2. For each section, rearrange blue cells to form more coherent shapes
    3. Balance the arrangement within each section
    4. Construct the output grid with the new arrangements
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Find red dot rows
    red_dot_rows = [i for i in range(rows) if 2 in input_grid.values[i]]
    
    # Process each section between red dot rows
    for start, end in zip([-1] + red_dot_rows, red_dot_rows + [rows]):
        if end - start > 1:  # Skip sections with no space for blue cells
            rearrange_section(output_grid, start + 1, end - 1, cols)
    
    return output_grid

def rearrange_section(grid: ColoredGrid, start: int, end: int, cols: int):
    blue_cells = []
    for r in range(start, end + 1):
        for c in range(cols):
            if grid.values[r][c] == 1:
                blue_cells.append((r, c))
                grid.values[r][c] = 0  # Clear blue cells
    
    # Sort blue cells by their horizontal position
    blue_cells.sort(key=lambda x: x[1])
    
    # Rearrange blue cells
    center_col = cols // 2
    for i, (r, c) in enumerate(blue_cells):
        new_r = start + (i // 4)  # Distribute vertically
        new_c = center_col + (i % 4) - 1  # Distribute around center
        if new_c >= cols:
            new_c = center_col - (i % 4) + 2
        grid.values[new_r][new_c] = 1

def get_neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors
