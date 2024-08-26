from rob_agi.colored_grid import ColoredGrid
from typing import Set, Tuple

def solve_8fbca751(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by outlining all blue shapes with red.
    
    This function identifies all blue (8) cells in the input grid, then outlines them
    collectively with red (2) cells. The outline includes diagonally adjacent cells
    but does not extend beyond the grid boundaries or overwrite existing non-black cells.
    All blue shapes are enclosed in a single outline, regardless of their connectivity.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with all blue shapes outlined in red.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    outline_cells = set()

    # Step 1: Identify all cells that should be part of the outline
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:  # Blue cell
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            outline_cells.add((nr, nc))

    # Step 2: Remove blue cells from the outline set
    outline_cells = {(r, c) for r, c in outline_cells if grid.values[r][c] != 8}

    # Step 3: Apply the outline to the grid
    for r, c in outline_cells:
        if grid.values[r][c] == 0:  # Only change black cells to red
            grid.values[r][c] = 2

    return grid
