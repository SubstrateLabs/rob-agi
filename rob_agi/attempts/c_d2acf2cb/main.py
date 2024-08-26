from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions in each row.
    2. For rows with yellow squares at both ends:
       - If the content between yellows is all magenta (6) or a mix of magenta and black (0):
         Replace with a pattern of sky (8) and orange (7), with two sky colors adjacent to yellows,
         and the rest filled with orange.
       - If the content includes other colors, preserve the original pattern.
    3. Preserves existing structures, yellow squares, and other colors in all other rows.
    4. Maintains overall grid structure and symmetry.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    for r in range(rows):
        yellow_positions = [c for c in range(cols) if output_grid.values[r][c] == 4]
        if len(yellow_positions) == 2:
            start, end = yellow_positions
            middle_content = output_grid.values[r][start+1:end]
            if all(color in [0, 6] for color in middle_content):
                output_grid.values[r][start+1] = 8
                output_grid.values[r][end-1] = 8
                for i in range(start + 2, end - 1):
                    output_grid.values[r][i] = 7

    return output_grid
