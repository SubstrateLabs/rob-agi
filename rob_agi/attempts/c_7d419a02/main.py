from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7d419a02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue (8) regions to yellow (4).
    
    The transformation follows these rules:
    1. Blue regions are changed to yellow based on their size and position.
    2. Larger blue regions (3x3 or larger) are more likely to be changed to yellow.
    3. Blue regions closer to the edges are more likely to be changed to yellow.
    4. Single-width blue lines and isolated blue cells usually remain blue.
    5. Black (0) and magenta (6) cells remain unchanged.
    6. The transformation is applied consistently across the entire grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    processed = set()

    def is_edge_region(region: List[Tuple[int, int]]) -> bool:
        return any(r == 0 or r == rows - 1 or c == 0 or c == cols - 1 for r, c in region)

    def should_transform(region: List[Tuple[int, int]]) -> bool:
        size = len(region)
        if size < 4:  # Keep 1x1 and most 2x2 regions blue
            return False
        if size >= 9:  # Always transform 3x3 or larger
            return True
        return is_edge_region(region) or size >= 6  # Transform 2x3 or larger edge regions

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
