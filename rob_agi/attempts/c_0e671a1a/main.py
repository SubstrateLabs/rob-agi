from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_0e671a1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a rectangular gray path
    that encloses all colored squares and fills the enclosed area.

    1. Find the colored squares (red, yellow, green).
    2. Determine the bounding rectangle for these squares.
    3. Draw a gray rectangular path along the bounding rectangle.
    4. Fill the enclosed area with gray.
    5. Restore the original colored squares.

    Args:
    input_grid (ColoredGrid): The input grid with three colored squares.

    Returns:
    ColoredGrid: The transformed grid with the gray rectangular path and filled area.
    """
    def find_colored_squares(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        squares = []
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if grid[r][c] in [2, 3, 4]:
                    squares.append((r, c, grid[r][c]))
        return squares

    def get_bounding_rectangle(squares: List[Tuple[int, int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(s[0] for s in squares)
        min_c = min(s[1] for s in squares)
        max_r = max(s[0] for s in squares)
        max_c = max(s[1] for s in squares)
        return min_r, min_c, max_r, max_c

    output_grid = input_grid.deep_copy()
    colored_squares = find_colored_squares(output_grid)
    min_r, min_c, max_r, max_c = get_bounding_rectangle(colored_squares)

    # Draw rectangular path and fill enclosed area
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if r == min_r or r == max_r or c == min_c or c == max_c or (min_r < r < max_r and min_c < c < max_c):
                output_grid[r][c] = 5  # Gray

    # Restore original colored squares
    for r, c, color in colored_squares:
        output_grid[r][c] = color

    return output_grid
