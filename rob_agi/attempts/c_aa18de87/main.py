from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_aa18de87(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the aa18de87 challenge by filling areas visible to non-black colors with red.
    
    The function identifies all unique non-black colors in the grid, finds their bounding rectangles,
    and fills the areas within these rectangles that are visible to the colored dots in both row and column
    directions. Visibility is determined by line of sight without obstruction by other non-black colors.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with filled red areas.
    """
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    
    # Get dimensions of the grid
    rows, cols = grid.get_dimensions()
    
    def is_visible_in_row(row: int, col: int, color_cells: Set[Tuple[int, int]]) -> bool:
        # Check visibility to the left
        for c in range(col - 1, -1, -1):
            if (row, c) in color_cells:
                return True
            if grid.values[row][c] != 0:
                break
        # Check visibility to the right
        for c in range(col + 1, cols):
            if (row, c) in color_cells:
                return True
            if grid.values[row][c] != 0:
                break
        return False

    def is_visible_in_column(row: int, col: int, color_cells: Set[Tuple[int, int]]) -> bool:
        # Check visibility upwards
        for r in range(row - 1, -1, -1):
            if (r, col) in color_cells:
                return True
            if grid.values[r][col] != 0:
                break
        # Check visibility downwards
        for r in range(row + 1, rows):
            if (r, col) in color_cells:
                return True
            if grid.values[r][col] != 0:
                break
        return False

    # Identify unique non-black colors
    unique_colors = set(cell for row in grid.values for cell in row if cell != 0)
    
    # Process each unique color
    for color in unique_colors:
        # Find all cells with this color
        color_cells = set((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == color)
        
        # Determine the bounding rectangle
        min_r = min(r for r, _ in color_cells)
        max_r = max(r for r, _ in color_cells)
        min_c = min(c for _, c in color_cells)
        max_c = max(c for _, c in color_cells)
        
        # Fill the visible areas within the bounding rectangle
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid.values[r][c] == 0:
                    if is_visible_in_row(r, c, color_cells) and is_visible_in_column(r, c, color_cells):
                        grid.values[r][c] = 2  # Red
    
    return grid
