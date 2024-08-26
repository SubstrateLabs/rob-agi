from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_e0fb7511(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a meandering path of sky blue (8) squares
    that connects different parts of the grid, favoring diagonal connections and
    maintaining some of the original black (0) squares.

    The function performs the following steps:
    1. Identify anchor points (black squares near corners/edges or central)
    2. Create a main path between anchor points, transforming squares to sky blue
    3. Expand sky blue regions, prioritizing diagonal connections
    4. Create secondary paths from remaining black regions
    5. Fine-tune the pattern by addressing isolated black squares
    6. Balance the transformation to ensure a prominent yet faithful pattern

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with the sky blue path pattern
    """
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    # Helper functions
    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1] 
                if 0 <= r+dr < rows and 0 <= c+dc < cols and (dr != 0 or dc != 0)]

    def is_black(r: int, c: int) -> bool:
        return grid.get_cell(r, c) == 0

    def is_blue(r: int, c: int) -> bool:
        return grid.get_cell(r, c) == 1

    def set_sky_blue(r: int, c: int):
        grid.set_cell(r, c, 8)

    # Find anchor points (black squares near corners/edges or central)
    anchor_points = []
    for r in [0, rows//2, rows-1]:
        for c in [0, cols//2, cols-1]:
            if is_black(r, c):
                anchor_points.append((r, c))
    
    if not anchor_points:
        # If no anchor points found, use random black squares
        black_squares = [(r, c) for r in range(rows) for c in range(cols) if is_black(r, c)]
        if black_squares:
            anchor_points = random.sample(black_squares, min(2, len(black_squares)))

    # Create main path between anchor points
    for start, end in zip(anchor_points, anchor_points[1:] + [anchor_points[0]]):
        r, c = start
        while (r, c) != end:
            set_sky_blue(r, c)
            dr = (end[0] - r) // max(1, abs(end[0] - r))
            dc = (end[1] - c) // max(1, abs(end[1] - c))
            r, c = r + dr, c + dc

    # Expand sky blue regions
    for _ in range(2):  # Repeat to create more prominent patterns
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8:
                    for nr, nc in get_neighbors(r, c):
                        if is_black(nr, nc) and random.random() < 0.7:  # 70% chance to expand
                            set_sky_blue(nr, nc)

    # Create secondary paths from remaining black regions
    black_regions = grid.find_connected_regions(0)
    for region in black_regions:
        if len(region) > 1:
            start = random.choice(region)
            nearest_sky = min((r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 8,
                              key=lambda x: abs(x[0]-start[0]) + abs(x[1]-start[1]))
            r, c = start
            while (r, c) != nearest_sky:
                set_sky_blue(r, c)
                dr = (nearest_sky[0] - r) // max(1, abs(nearest_sky[0] - r))
                dc = (nearest_sky[1] - c) // max(1, abs(nearest_sky[1] - c))
                r, c = r + dr, c + dc

    # Fine-tune the pattern
    for r in range(rows):
        for c in range(cols):
            if is_black(r, c):
                neighbors = get_neighbors(r, c)
                if sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) in [1, 8]) >= 6:
                    set_sky_blue(r, c)

    return grid
