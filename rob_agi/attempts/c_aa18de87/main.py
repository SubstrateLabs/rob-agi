from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_aa18de87(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the aa18de87 challenge by filling areas between non-black colors with red.
    
    The function identifies all unique non-black colors in the grid, and for each color:
    1. Fills black cells between the leftmost and rightmost cells of this color in each row.
    2. Fills black cells between the topmost and bottommost cells of this color in each column.
    3. Ensures that each red cell is connected to at least one cell of the original color in both its row and column.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with filled red areas.
    """
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    
    # Get dimensions of the grid
    rows, cols = grid.get_dimensions()
    
    def fill_between(start: int, end: int, line: List[int], color: int):
        for i in range(start + 1, end):
            if line[i] == 0:
                line[i] = 2  # Red

    def is_connected(r: int, c: int, color: int) -> bool:
        # Check row connectivity
        row_connected = False
        for col in range(c - 1, -1, -1):
            if grid.values[r][col] == color:
                row_connected = True
                break
            if grid.values[r][col] != 0 and grid.values[r][col] != 2:
                break
        if not row_connected:
            for col in range(c + 1, cols):
                if grid.values[r][col] == color:
                    row_connected = True
                    break
                if grid.values[r][col] != 0 and grid.values[r][col] != 2:
                    break
        
        # Check column connectivity
        col_connected = False
        for row in range(r - 1, -1, -1):
            if grid.values[row][c] == color:
                col_connected = True
                break
            if grid.values[row][c] != 0 and grid.values[row][c] != 2:
                break
        if not col_connected:
            for row in range(r + 1, rows):
                if grid.values[row][c] == color:
                    col_connected = True
                    break
                if grid.values[row][c] != 0 and grid.values[row][c] != 2:
                    break
        
        return row_connected and col_connected

    # Identify unique non-black colors
    unique_colors = set(cell for row in grid.values for cell in row if cell != 0)
    
    # Process each unique color
    for color in unique_colors:
        # Fill rows
        for r in range(rows):
            color_cells = [c for c, val in enumerate(grid.values[r]) if val == color]
            if color_cells:
                fill_between(min(color_cells), max(color_cells), grid.values[r], color)
        
        # Fill columns
        for c in range(cols):
            color_cells = [r for r in range(rows) if grid.values[r][c] == color]
            if color_cells:
                for r in range(min(color_cells) + 1, max(color_cells)):
                    if grid.values[r][c] == 0:
                        grid.values[r][c] = 2  # Red
    
    # Remove unconnected red cells
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2 and not is_connected(r, c, color):
                grid.values[r][c] = 0  # Revert to black
    
    return grid
