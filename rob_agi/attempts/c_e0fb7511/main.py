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
    2. Choose a starting point in a high-density area of black squares
    3. Grow a sky blue path using a modified random walk algorithm
    4. Ensure connectivity by implementing a "jumping" mechanism
    5. Preserve a portion of the original black squares
    6. Fine-tune the pattern for balance and connectivity
    7. Perform a final connectivity check and adjustments

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
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def is_black(r: int, c: int) -> bool:
        return grid.get_cell(r, c) == 0

    def set_sky_blue(r: int, c: int):
        grid.set_cell(r, c, 8)

    # Analyze black square distribution
    black_squares = [(r, c) for r in range(rows) for c in range(cols) if is_black(r, c)]
    total_black = len(black_squares)

    # Choose a starting point
    start = max(black_squares, key=lambda x: sum(1 for nr, nc in get_neighbors(*x) if is_black(nr, nc)))
    path = [start]
    set_sky_blue(*start)

    # Grow the path
    target_length = int(total_black * random.uniform(0.5, 0.8))
    while len(path) < target_length:
        r, c = path[-1]
        neighbors = get_neighbors(r, c)
        valid_moves = [
            (nr, nc) for nr, nc in neighbors
            if grid.get_cell(nr, nc) in [0, 1] and (nr, nc) not in path
        ]
        if not valid_moves:
            # Implement jumping mechanism
            unvisited_black = [sq for sq in black_squares if sq not in path]
            if not unvisited_black:
                break
            jump_to = min(unvisited_black, key=lambda x: abs(x[0]-r) + abs(x[1]-c))
            while (r, c) != jump_to:
                r += (jump_to[0] - r) // max(1, abs(jump_to[0] - r))
                c += (jump_to[1] - c) // max(1, abs(jump_to[1] - c))
                set_sky_blue(r, c)
                path.append((r, c))
        else:
            next_move = max(valid_moves, key=lambda x: 2 if is_black(*x) else 1)
            set_sky_blue(*next_move)
            path.append(next_move)

    # Preserve original black squares
    black_to_keep = random.sample(black_squares, k=int(total_black * random.uniform(0.2, 0.4)))
    for r, c in black_to_keep:
        if (r, c) not in path:
            grid.set_cell(r, c, 0)

    # Fine-tune the pattern
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                neighbors = get_neighbors(r, c)
                sky_blue_neighbors = sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) == 8)
                if sky_blue_neighbors <= 1:
                    grid.set_cell(r, c, 1)  # Convert isolated sky blue to blue

    # Final connectivity check
    sky_blue_regions = grid.find_connected_regions(8)
    if len(sky_blue_regions) > 1:
        main_region = max(sky_blue_regions, key=len)
        for region in sky_blue_regions:
            if region != main_region:
                start = region[0]
                end = min(main_region, key=lambda x: abs(x[0]-start[0]) + abs(x[1]-start[1]))
                while start != end:
                    r, c = start
                    r += (end[0] - r) // max(1, abs(end[0] - r))
                    c += (end[1] - c) // max(1, abs(end[1] - c))
                    set_sky_blue(r, c)
                    start = (r, c)

    return grid
