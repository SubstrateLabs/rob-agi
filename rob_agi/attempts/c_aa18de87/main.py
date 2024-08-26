from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa18de87(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the aa18de87 challenge by filling the bounding rectangles of non-black colors with red.
    
    The function identifies all unique non-black colors in the grid, finds their bounding rectangles,
    and fills these rectangles with red (2) while preserving the original colored dots. It handles
    overlapping areas by prioritizing the original colors and red fill.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with filled red areas.
    """
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    
    # Get dimensions of the grid
    rows, cols = grid.get_dimensions()
    
    # Identify unique non-black colors
    unique_colors = set(cell for row in grid.values for cell in row if cell != 0)
    
    # Process each unique color
    for color in unique_colors:
        # Find all cells with this color
        color_cells = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == color]
        
        # Determine the bounding rectangle
        min_r = min(r for r, _ in color_cells)
        max_r = max(r for r, _ in color_cells)
        min_c = min(c for _, c in color_cells)
        max_c = max(c for _, c in color_cells)
        
        # Fill the bounding rectangle with red
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid.values[r][c] == 0 or (grid.values[r][c] != color and grid.values[r][c] != 2):
                    grid.values[r][c] = 2  # Red
    
    return grid
