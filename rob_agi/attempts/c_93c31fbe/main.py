from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting and extending blue (1) elements
    while maintaining other colored elements. The solution:
    1. Analyzes the input grid to identify blue elements and their distribution
    2. Creates a connectivity map of blue elements
    3. Identifies potential shapes or patterns
    4. Develops a symmetry plan based on the center of mass
    5. Forms patterns by extending lines and shapes from the largest cluster
    6. Connects isolated elements to the main pattern
    7. Balances the overall distribution of blue elements
    8. Refines the shape to create a cohesive, intentional appearance
    9. Respects boundaries of other colored elements throughout the process
    10. Performs a final symmetry check and makes minor adjustments if needed
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_blue_dots() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1]

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.values[r][c] in [0, 1]

    def connect_dots(start: Tuple[int, int], end: Tuple[int, int]):
        r1, c1 = start
        r2, c2 = end
        dx = c2 - c1
        dy = r2 - r1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return
        for i in range(steps + 1):
            r = round(r1 + i * dy / steps)
            c = round(c1 + i * dx / steps)
            if is_valid(r, c):
                grid.values[r][c] = 1

    blue_dots = get_blue_dots()

    # Calculate center of mass
    if blue_dots:
        center_r = sum(r for r, _ in blue_dots) / len(blue_dots)
        center_c = sum(c for _, c in blue_dots) / len(blue_dots)
    else:
        center_r, center_c = rows // 2, cols // 2

    # Connect nearby blue dots and extend patterns
    for i, (r1, c1) in enumerate(blue_dots):
        for r2, c2 in blue_dots[i+1:]:
            if abs(r1 - r2) + abs(c1 - c2) <= 3:
                connect_dots((r1, c1), (r2, c2))

    # Create symmetry
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                mirror_r = int(2 * center_r - r)
                mirror_c = int(2 * center_c - c)
                if is_valid(mirror_r, mirror_c):
                    grid.values[mirror_r][mirror_c] = 1

    # Connect isolated elements and fill gaps
    blue_regions = grid.find_connected_regions(1)
    if len(blue_regions) > 1:
        main_region = max(blue_regions, key=len)
        for region in blue_regions:
            if region != main_region:
                start = region[0]
                end = min(main_region, key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]))
                connect_dots(start, end)

    # Refine shape
    for _ in range(2):
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 1:
                    neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)] if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 1)
                    if neighbors <= 1:
                        grid.values[r][c] = 0
                else:
                    neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)] if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 1)
                    if neighbors >= 5:
                        grid.values[r][c] = 1

    return grid
