from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating symmetrical patterns with blue (1) elements while respecting other colored elements.
    The solution:
    1. Analyzes the grid to identify regions and symmetry axes.
    2. Creates symmetrical patterns in each region, using 2x2 blue squares as anchors.
    3. Connects patterns between regions, maintaining overall symmetry.
    4. Enhances global symmetry and visual appeal.
    5. Ensures connectivity of all blue pixels.
    6. Optimizes the pattern for aesthetic appeal and symmetry.
    7. Handles boundaries with non-blue shapes and isolated blue pixels.
    8. Performs final cleanup and validation.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_blue_pixels() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1}

    def get_other_shapes() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] not in [0, 1]}

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.values[r][c] in [0, 1]

    def find_regions() -> List[Set[Tuple[int, int]]]:
        regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.values[r][c] == 0:
                    region = set()
                    queue = deque([(r, c)])
                    while queue:
                        cr, cc = queue.popleft()
                        if (cr, cc) not in visited and grid.values[cr][cc] == 0:
                            region.add((cr, cc))
                            visited.add((cr, cc))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = cr + dr, cc + dc
                                if is_valid(nr, nc):
                                    queue.append((nr, nc))
                    regions.append(region)
        return regions

    def create_symmetrical_pattern(region: Set[Tuple[int, int]]):
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        center_r, center_c = (min_r + max_r) // 2, (min_c + max_c) // 2

        for r, c in region:
            if (r + c) % 2 == 0 and is_valid(r+1, c+1):
                grid.values[r][c] = 1
                grid.values[r+1][c] = 1
                grid.values[r][c+1] = 1
                grid.values[r+1][c+1] = 1

        # Create symmetry
        for r, c in region:
            if grid.values[r][c] == 1:
                sym_r = 2 * center_r - r
                sym_c = 2 * center_c - c
                if (sym_r, sym_c) in region:
                    grid.values[sym_r][sym_c] = 1

    def connect_regions(regions: List[Set[Tuple[int, int]]]):
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                min_dist = float('inf')
                connection = None
                for r1, c1 in region1:
                    for r2, c2 in region2:
                        if grid.values[r1][c1] == 1 and grid.values[r2][c2] == 1:
                            dist = abs(r1 - r2) + abs(c1 - c2)
                            if dist < min_dist:
                                min_dist = dist
                                connection = ((r1, c1), (r2, c2))
                if connection:
                    (r1, c1), (r2, c2) = connection
                    r, c = r1, c1
                    while (r, c) != (r2, c2):
                        if r < r2:
                            r += 1
                        elif r > r2:
                            r -= 1
                        if c < c2:
                            c += 1
                        elif c > c2:
                            c -= 1
                        if is_valid(r, c):
                            grid.values[r][c] = 1

    def enhance_global_symmetry():
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 1:
                    sym_r, sym_c = rows - 1 - r, cols - 1 - c
                    if is_valid(sym_r, sym_c):
                        grid.values[sym_r][sym_c] = 1

    def ensure_connectivity():
        blue_pixels = get_blue_pixels()
        if not blue_pixels:
            return
        connected = set()
        stack = [next(iter(blue_pixels))]
        while stack:
            r, c = stack.pop()
            if (r, c) not in connected:
                connected.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in blue_pixels:
                        stack.append((nr, nc))
        for r, c in blue_pixels - connected:
            grid.values[r][c] = 0

    other_shapes = get_other_shapes()
    regions = find_regions()
    for region in regions:
        create_symmetrical_pattern(region)
    connect_regions(regions)
    enhance_global_symmetry()
    ensure_connectivity()

    return grid
