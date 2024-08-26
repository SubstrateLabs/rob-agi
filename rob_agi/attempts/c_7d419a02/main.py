from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

import math
from typing import List, Tuple

def solve_7d419a02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue (8) regions to yellow (4) based on their position and structure.
    
    The transformation follows these rules:
    1. Blue regions closer to the edges are more likely to be changed to yellow.
    2. A blue "core" is maintained in the center of the grid.
    3. Single-width blue lines and isolated blue cells usually remain blue.
    4. Black (0) and magenta (6) cells remain unchanged.
    5. The transformation is applied consistently across the entire grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    threshold = max(rows, cols) * 0.3
    processed = set()

    def distance_from_center(r: int, c: int) -> float:
        return math.sqrt((r - center_row)**2 + (c - center_col)**2)

    def is_single_width_line(region: List[Tuple[int, int]]) -> bool:
        for r, c in region:
            non_blue_neighbors = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                     if 0 <= r + dr < rows and 0 <= c + dc < cols and not is_blue(grid.values[r + dr][c + dc]))
            if non_blue_neighbors < 2:
                return False
        return True

    def should_transform(region: List[Tuple[int, int]]) -> bool:
        if is_single_width_line(region) or len(region) == 1:
            return False
        avg_distance = sum(distance_from_center(r, c) for r, c in region) / len(region)
        return avg_distance > threshold

    for row in range(rows):
        for col in range(cols):
            if is_blue(grid.values[row][col]) and (row, col) not in processed:
                region = flood_fill(grid, row, col, processed)
                if should_transform(region):
                    for r, c in region:
                        grid.values[r][c] = 4  # Change to yellow

    return grid

def is_blue(cell: int) -> bool:
    return cell == 8

def flood_fill(grid: ColoredGrid, row: int, col: int, processed: set) -> List[Tuple[int, int]]:
    stack = [(row, col)]
    region = []
    while stack:
        r, c = stack.pop()
        if (r, c) not in processed and is_blue(grid.values[r][c]):
            processed.add((r, c))
            region.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    stack.append((nr, nc))
    return region
