from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_3490cc26(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting sky blue (8) squares with orange (7) paths.
    
    The solution follows these steps:
    1. Identify all 2x2 sky blue squares in the input grid.
    2. Process rows: Fill with orange (7) between leftmost and rightmost sky blue squares in each row.
    3. Process columns: Fill with orange (7) between topmost and bottommost sky blue squares in each column.
    4. Preserve the original sky blue squares.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with orange paths connecting sky blue squares.
    """
    output_grid = input_grid.deep_copy()
    sky_blue_squares = find_sky_blue_squares(output_grid)
    
    process_rows(output_grid, sky_blue_squares)
    process_columns(output_grid, sky_blue_squares)
    preserve_sky_blue_squares(output_grid, sky_blue_squares)
    
    return output_grid

def find_sky_blue_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Find all 2x2 sky blue squares in the grid."""
    squares = []
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) == 8 for dr in range(2) for dc in range(2)):
                squares.append((r, c))
    return squares

def process_rows(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Fill rows with orange between sky blue squares."""
    rows, _ = grid.get_dimensions()
    for r in range(rows):
        row_squares = [sq for sq in squares if sq[0] == r]
        if len(row_squares) >= 2:
            left = min(sq[1] for sq in row_squares)
            right = max(sq[1] for sq in row_squares)
            for c in range(left, right + 2):
                if grid.get_cell(r, c) != 8:
                    grid.set_cell(r, c, 7)

def process_columns(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Fill columns with orange between sky blue squares."""
    _, cols = grid.get_dimensions()
    for c in range(cols):
        col_squares = [sq for sq in squares if sq[1] == c]
        if len(col_squares) >= 2:
            top = min(sq[0] for sq in col_squares)
            bottom = max(sq[0] for sq in col_squares)
            for r in range(top, bottom + 2):
                if grid.get_cell(r, c) != 8:
                    grid.set_cell(r, c, 7)

def preserve_sky_blue_squares(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Ensure all original sky blue squares remain intact."""
    for r, c in squares:
        for dr in range(2):
            for dc in range(2):
                grid.set_cell(r + dr, c + dc, 8)
