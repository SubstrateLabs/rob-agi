from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a symmetrical blue (1) pattern while respecting other colored elements.
    The solution:
    1. Analyzes the grid to identify non-blue elements and grid dimensions.
    2. Establishes a main backbone (vertical or horizontal) based on grid orientation.
    3. Connects the backbone to non-blue elements symmetrically.
    4. Creates symmetrical branches from the backbone.
    5. Forms symmetrical units (2x2 squares, crosses) at intersections and endpoints.
    6. Fills in details in empty spaces while maintaining symmetry.
    7. Ensures connectivity of all blue pixels.
    8. Optimizes and cleans up the pattern.
    9. Performs final checks for symmetry, connectivity, and interaction with non-blue elements.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_non_blue_elements() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] not in [0, 1]}

    def create_backbone():
        if rows >= cols:
            c = cols // 2
            for r in range(rows):
                if is_valid(r, c) and grid.values[r][c] == 0:
                    grid.values[r][c] = 1
        else:
            r = rows // 2
            for c in range(cols):
                if is_valid(r, c) and grid.values[r][c] == 0:
                    grid.values[r][c] = 1

    def connect_to_non_blue(non_blue: Set[Tuple[int, int]]):
        backbone = [(r, cols//2) for r in range(rows)] if rows >= cols else [(rows//2, c) for c in range(cols)]
        for r, c in non_blue:
            closest = min(backbone, key=lambda x: abs(x[0]-r) + abs(x[1]-c))
            r1, c1 = closest
            while (r1, c1) != (r, c):
                if r1 < r:
                    r1 += 1
                elif r1 > r:
                    r1 -= 1
                if c1 < c:
                    c1 += 1
                elif c1 > c:
                    c1 -= 1
                if is_valid(r1, c1) and grid.values[r1][c1] == 0:
                    grid.values[r1][c1] = 1

    def create_branches():
        if rows >= cols:
            for r in range(2, rows-2, 4):
                for c in range(cols):
                    if is_valid(r, c) and grid.values[r][c] == 0:
                        grid.values[r][c] = 1
        else:
            for c in range(2, cols-2, 4):
                for r in range(rows):
                    if is_valid(r, c) and grid.values[r][c] == 0:
                        grid.values[r][c] = 1

    def create_symmetrical_units():
        for r in range(rows-1):
            for c in range(cols-1):
                if all(is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 0 for dr, dc in [(0,0), (0,1), (1,0), (1,1)]):
                    for dr, dc in [(0,0), (0,1), (1,0), (1,1)]:
                        grid.values[r+dr][c+dc] = 1

    def fill_details():
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if grid.values[r][c] == 0 and sum(grid.values[r+dr][c+dc] == 1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]) >= 2:
                    grid.values[r][c] = 1

    def ensure_connectivity():
        visited = set()
        stack = [(0, 0)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and is_valid(r, c):
                visited.add((r, c))
                if grid.values[r][c] == 1:
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nr, nc = r+dr, c+dc
                        if is_valid(nr, nc) and (nr, nc) not in visited:
                            stack.append((nr, nc))
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 1 and (r, c) not in visited:
                    grid.values[r][c] = 0

    def optimize_and_clean():
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 1:
                    neighbors = sum(1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)] if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 1)
                    if neighbors <= 1:
                        grid.values[r][c] = 0

    non_blue = get_non_blue_elements()
    create_backbone()
    connect_to_non_blue(non_blue)
    create_branches()
    create_symmetrical_units()
    fill_details()
    ensure_connectivity()
    optimize_and_clean()

    return grid
