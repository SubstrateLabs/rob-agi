from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_3490cc26(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting sky blue (8) squares with orange (7) paths.
    
    The solution follows these steps:
    1. Find all 2x2 sky blue squares in the input grid.
    2. Sort the blue squares from top to bottom, then left to right.
    3. Connect the squares with orange paths, always moving down or right.
    4. Expand the orange paths to be 2 cells wide.
    5. Ensure blue and red squares are not overwritten.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with orange paths connecting blue squares.
    """
    # Step 1: Find all 2x2 sky blue squares
    blue_squares = find_blue_squares(input_grid)
    
    # Step 2: Sort the blue squares
    blue_squares.sort(key=lambda x: (x[0], x[1]))
    
    # Create a copy of the input grid to modify
    output_grid = input_grid.deep_copy()
    
    # Step 3 & 4: Connect squares and expand paths
    if blue_squares:
        current_pos = (blue_squares[0][0] + 1, blue_squares[0][1] + 1)  # Start at center of first square
        for square in blue_squares[1:]:
            target_pos = (square[0] + 1, square[1] + 1)  # Center of target square
            draw_path(output_grid, current_pos, target_pos)
            current_pos = target_pos
    
    return output_grid

def find_blue_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Find all 2x2 sky blue squares in the grid."""
    blue_squares = []
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) == 8 for dr in range(2) for dc in range(2)):
                blue_squares.append((r, c))
    return blue_squares

def draw_path(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    """Draw and expand an orange path from start to end."""
    r1, c1 = start
    r2, c2 = end
    
    # Draw vertical path
    for r in range(min(r1, r2), max(r1, r2) + 1):
        color_cell(grid, r, c1)
        color_cell(grid, r, c1 - 1)
    
    # Draw horizontal path
    for c in range(min(c1, c2), max(c1, c2) + 1):
        color_cell(grid, r2, c)
        color_cell(grid, r2 - 1, c)

def color_cell(grid: ColoredGrid, r: int, c: int):
    """Color a cell orange if it's not part of a blue or red square."""
    if grid.get_cell(r, c) not in [2, 8]:
        grid.set_cell(r, c, 7)
