from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_05a7bcf2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Vertically expands colors 1-7 upward and 8-9 downward until hitting another color or edge.
    2. Fills empty space below horizontal sky blue (8) lines.
    3. Horizontally expands colors between instances of the same color in each row.
    4. Fills remaining empty cells with sky blue.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def vertical_expand():
        for color in range(9, 0, -1):
            for r in range(rows):
                for c in range(cols):
                    if grid.values[r][c] == color:
                        if color < 8:
                            # Expand upward
                            for i in range(r-1, -1, -1):
                                if grid.values[i][c] == 0:
                                    grid.values[i][c] = color
                                else:
                                    break
                        else:
                            # Expand downward
                            for i in range(r+1, rows):
                                if grid.values[i][c] == 0:
                                    grid.values[i][c] = color
                                else:
                                    break

    def sky_blue_fill():
        for r in range(rows):
            if 8 in grid.values[r]:
                for c in range(cols):
                    if grid.values[r][c] == 8:
                        for i in range(r+1, rows):
                            if grid.values[i][c] == 0:
                                grid.values[i][c] = 8
                            else:
                                break

    def horizontal_expand():
        for r in range(rows):
            original_colors = set(grid.values[r]) - {0, 8}
            start = 0
            while start < cols:
                if grid.values[r][start] == 8:
                    start += 1
                    continue
                end = start + 1
                while end < cols and grid.values[r][end] != 8:
                    end += 1
                for color in original_colors:
                    indices = [i for i in range(start, end) if grid.values[r][i] == color]
                    if len(indices) > 1:
                        for i in range(indices[0], indices[-1]+1):
                            if grid.values[r][i] == 0:
                                grid.values[r][i] = color
                start = end + 1

    def final_sky_blue_fill():
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 8

    vertical_expand()
    sky_blue_fill()
    horizontal_expand()
    final_sky_blue_fill()

    return grid
