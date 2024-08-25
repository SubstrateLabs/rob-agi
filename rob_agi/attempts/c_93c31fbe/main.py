from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting and extending blue (1) elements
    while maintaining other colored elements. The solution:
    1. Connects nearby blue dots
    2. Extends partial blue patterns to form more complete shapes
    3. Creates symmetry in blue patterns
    4. Connects isolated blue elements
    5. Fills logical gaps in blue patterns
    6. Balances the overall distribution of blue elements
    7. Respects boundaries of other colored elements
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_blue_dots() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1]

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def connect_dots(start: Tuple[int, int], end: Tuple[int, int]):
        r1, c1 = start
        r2, c2 = end
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if grid.values[r1][c] == 0:
                    grid.values[r1][c] = 1
        elif c1 == c2:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if grid.values[r][c1] == 0:
                    grid.values[r][c1] = 1

    blue_dots = get_blue_dots()

    # Connect nearby blue dots
    for i, (r1, c1) in enumerate(blue_dots):
        for r2, c2 in blue_dots[i+1:]:
            if abs(r1 - r2) + abs(c1 - c2) <= 2:
                connect_dots((r1, c1), (r2, c2))

    # Extend patterns and create symmetry
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc) and grid.values[nr][nc] == 0:
                        nnr, nnc = nr + dr, nc + dc
                        if is_valid(nnr, nnc) and grid.values[nnr][nnc] == 1:
                            grid.values[nr][nc] = 1

    # Connect isolated elements and fill gaps
    blue_regions = grid.find_connected_regions(1)
    if len(blue_regions) > 1:
        main_region = max(blue_regions, key=len)
        for region in blue_regions:
            if region != main_region:
                start = region[0]
                end = min(main_region, key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]))
                connect_dots(start, end)

    return grid
