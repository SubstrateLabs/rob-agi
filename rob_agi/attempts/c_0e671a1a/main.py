from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_0e671a1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating an optimal gray path
    that connects all colored squares and partially fills the enclosed area.

    1. Find the colored squares (red, yellow, green).
    2. Determine the optimal path connecting these squares.
    3. Draw the gray path along the determined route.
    4. Partially fill the enclosed area with gray, stopping at colored squares.
    5. Preserve the original colored squares.

    Args:
    input_grid (ColoredGrid): The input grid with three colored squares.

    Returns:
    ColoredGrid: The transformed grid with the optimal gray path and partially filled area.
    """
    def find_colored_squares(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        squares = []
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if grid[r][c] in [2, 3, 4]:
                    squares.append((r, c, grid[r][c]))
        return squares

    output_grid = input_grid.deep_copy()
    colored_squares = find_colored_squares(output_grid)

    # Determine the optimal path
    left = min(square[1] for square in colored_squares)
    right = max(square[1] for square in colored_squares)
    top = min(square[0] for square in colored_squares)
    bottom = max(square[0] for square in colored_squares)

    # Draw the path and fill the enclosed area
    for r in range(top, bottom + 1):
        left_edge = right
        right_edge = left
        for c in range(left, right + 1):
            if output_grid[r][c] in [2, 3, 4, 5]:
                left_edge = min(left_edge, c)
                right_edge = max(right_edge, c)
        
        for c in range(left_edge, right_edge + 1):
            if output_grid[r][c] == 0:
                output_grid[r][c] = 5
            elif output_grid[r][c] in [2, 3, 4]:
                break

    # Draw vertical lines
    for c in [left, right]:
        for r in range(top, bottom + 1):
            if output_grid[r][c] == 0:
                output_grid[r][c] = 5

    return output_grid
