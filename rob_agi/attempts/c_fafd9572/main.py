from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_fafd9572(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying connected regions of non-black colors,
    sorting them by their top-left-most pixel, and then assigning new colors in a cyclic manner.
    Existing red (2), green (3), and yellow (4) regions are preserved.
    New colors are assigned in the order: red (2) -> green (3) -> yellow (4) -> red (2).
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    regions = identify_and_sort_regions(input_grid)
    new_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    next_color = 2
    for region, color in regions:
        if color in [2, 3, 4]:
            for x, y in region:
                new_grid.values[y][x] = color
        else:
            for x, y in region:
                new_grid.values[y][x] = next_color
            next_color = next_color + 1 if next_color < 4 else 2
    
    return new_grid

def identify_and_sort_regions(grid: ColoredGrid) -> List[Tuple[List[Tuple[int, int]], int]]:
    visited = set()
    regions = []
    
    for y in range(grid.num_rows):
        for x in range(grid.num_cols):
            if (x, y) not in visited and grid.values[y][x] != 0:
                region = []
                color = grid.values[y][x]
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited and 0 <= cx < grid.num_cols and 0 <= cy < grid.num_rows and grid.values[cy][cx] == color:
                        visited.add((cx, cy))
                        region.append((cx, cy))
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            stack.append((cx + dx, cy + dy))
                regions.append((region, color))
    
    return sorted(regions, key=lambda r: min(r[0]))
