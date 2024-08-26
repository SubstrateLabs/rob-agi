from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_13713586(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored lines while preserving gray boundaries.
    
    The algorithm works as follows:
    1. Identify and preserve gray (5) boundary lines.
    2. Identify colored regions and determine if they are lines.
    3. Categorize lines based on their position (left, right, top, bottom, center).
    4. For each line, determine expansion direction(s) and limits.
    5. Perform expansions from the edges inward, allowing color coexistence.
    6. Ensure gray boundaries remain unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_line(region: List[Tuple[int, int]]) -> Tuple[bool, str]:
        if len(region) <= 1:
            return False, ""
        r_coords = [r for r, _ in region]
        c_coords = [c for _, c in region]
        if len(set(r_coords)) == 1:
            return True, "horizontal"
        if len(set(c_coords)) == 1:
            return True, "vertical"
        return False, ""

    def get_expansion_limits(r: int, c: int, direction: str) -> Tuple[int, int]:
        if direction == "right":
            for nc in range(c + 1, cols):
                if grid.values[r][nc] in [5] or (grid.values[r][nc] != 0 and grid.values[r][nc] != grid.values[r][c]):
                    return r, nc - 1
            return r, cols - 1
        elif direction == "left":
            for nc in range(c - 1, -1, -1):
                if grid.values[r][nc] in [5] or (grid.values[r][nc] != 0 and grid.values[r][nc] != grid.values[r][c]):
                    return r, nc + 1
            return r, 0
        elif direction == "down":
            for nr in range(r + 1, rows):
                if grid.values[nr][c] in [5] or (grid.values[nr][c] != 0 and grid.values[nr][c] != grid.values[r][c]):
                    return nr - 1, c
            return rows - 1, c
        elif direction == "up":
            for nr in range(r - 1, -1, -1):
                if grid.values[nr][c] in [5] or (grid.values[nr][c] != 0 and grid.values[nr][c] != grid.values[r][c]):
                    return nr + 1, c
            return 0, c

    def expand_line(start_r: int, start_c: int, color: int, directions: List[str]):
        for direction in directions:
            r, c = start_r, start_c
            while True:
                if direction == "right":
                    c += 1
                elif direction == "left":
                    c -= 1
                elif direction == "down":
                    r += 1
                elif direction == "up":
                    r -= 1
                
                if r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] == 5:
                    break
                
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
                elif grid.values[r][c] != color:
                    break

    # Preserve gray boundaries
    gray_boundaries = []
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 5 and (r == 0 or r == rows - 1 or c == 0 or c == cols - 1):
                gray_boundaries.append((r, c))

    # Identify and expand colored lines
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        regions = grid.find_connected_regions(color)
        for region in regions:
            is_line_region, orientation = is_line(region)
            if is_line_region:
                r, c = region[0]
                if orientation == "horizontal":
                    if r < rows // 2:
                        expand_line(r, c, color, ["down"])
                    elif r >= rows // 2:
                        expand_line(r, c, color, ["up"])
                    else:
                        expand_line(r, c, color, ["up", "down"])
                elif orientation == "vertical":
                    if c < cols // 2:
                        expand_line(r, c, color, ["right"])
                    elif c >= cols // 2:
                        expand_line(r, c, color, ["left"])
                    else:
                        expand_line(r, c, color, ["left", "right"])

    # Restore gray boundaries
    for r, c in gray_boundaries:
        grid.values[r][c] = 5

    return grid
