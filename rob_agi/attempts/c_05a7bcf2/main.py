from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_05a7bcf2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Vertically expands all non-sky blue colors upward until hitting a sky blue barrier or the edge.
    2. Horizontally expands colors between instances of the same color within sections bounded by sky blue.
    3. Fills remaining empty cells with sky blue.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_sky_blue(cell: int) -> bool:
        return cell == 8

    def vertical_expand():
        for c in range(cols):
            for r in range(rows):
                if not is_sky_blue(grid.values[r][c]) and grid.values[r][c] != 0:
                    color = grid.values[r][c]
                    for i in range(r-1, -1, -1):
                        if grid.values[i][c] == 0:
                            grid.values[i][c] = color
                        elif is_sky_blue(grid.values[i][c]):
                            break
                        else:
                            break

    def find_sections():
        sections = []
        for r in range(rows):
            start = 0
            for c in range(cols):
                if is_sky_blue(grid.values[r][c]):
                    if start != c:
                        sections.append((r, start, c-1))
                    start = c + 1
            if start != cols:
                sections.append((r, start, cols-1))
        return sections

    def horizontal_expand(sections):
        for r, start, end in sections:
            colors = set(grid.values[r][start:end+1]) - {0, 8}
            for color in colors:
                left = right = -1
                for c in range(start, end+1):
                    if grid.values[r][c] == color:
                        if left == -1:
                            left = c
                        right = c
                if left != -1 and right != -1:
                    for c in range(left, right+1):
                        if grid.values[r][c] == 0:
                            grid.values[r][c] = color

    def sky_blue_fill():
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 8

    vertical_expand()
    sections = find_sections()
    horizontal_expand(sections)
    sky_blue_fill()

    return grid
