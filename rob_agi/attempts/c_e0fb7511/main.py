from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_e0fb7511(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a meandering path of sky blue (8) squares
    that connects different parts of the grid, reflecting the distribution of original
    black (0) squares while maintaining some of them.

    The function performs the following steps:
    1. Analyze the input grid to identify black square distribution
    2. Select anchor points from black squares
    3. Generate a sky blue path connecting anchor points
    4. Expand the sky blue pattern while preserving some original black squares
    5. Fine-tune the pattern for balance and connectivity
    6. Ensure the final pattern reflects the original distribution while creating a cohesive design

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

    def set_sky_blue(r: int, c: int):
        grid.set_cell(r, c, 8)

    # Analyze black square distribution
    black_squares = [(r, c) for r in range(rows) for c in range(cols) if is_black(r, c)]
    total_black = len(black_squares)

    # Select anchor points
    num_anchors = max(2, total_black // 3)
    anchor_points = random.sample(black_squares, min(num_anchors, total_black))

    # Generate sky blue path
    for start, end in zip(anchor_points, anchor_points[1:] + [anchor_points[0]]):
        r, c = start
        while (r, c) != end:
            set_sky_blue(r, c)
            dr = (end[0] - r) // max(1, abs(end[0] - r))
            dc = (end[1] - c) // max(1, abs(end[1] - c))
            r, c = r + dr, c + dc
            # Add some randomness to the path
            if random.random() < 0.3:
                r += random.choice([-1, 0, 1])
                c += random.choice([-1, 0, 1])
            r = max(0, min(r, rows-1))
            c = max(0, min(c, cols-1))

    # Expand sky blue pattern
    for _ in range(2):
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8:
                    for nr, nc in get_neighbors(r, c):
                        if is_black(nr, nc) and random.random() < 0.6:
                            set_sky_blue(nr, nc)

    # Connect disconnected segments
    sky_blue_regions = grid.find_connected_regions(8)
    if len(sky_blue_regions) > 1:
        for region in sky_blue_regions[1:]:
            start = region[0]
            nearest_sky = min((r, c) for r in range(rows) for c in range(cols) 
                              if grid.get_cell(r, c) == 8 and (r, c) not in region,
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
                if sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) == 8) >= 5:
                    set_sky_blue(r, c)

    # Ensure some original black squares remain
    sky_blue_count = sum(1 for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 8)
    if sky_blue_count > total_black * 0.7:
        black_to_keep = random.sample(black_squares, total_black // 3)
        for r, c in black_to_keep:
            grid.set_cell(r, c, 0)

    return grid
