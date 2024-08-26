from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating an inner diagonal skeleton of sky blue (8) regions with red (2).
    
    This function applies the following steps:
    1. Identify all sky blue (8) regions in the input grid.
    2. For each region, find the inner corners and create a minimal spanning structure connecting them.
    3. Mark the cells along this structure as red (2) in the output grid.
    4. Set all originally sky blue cells to black (0) in the output grid.
    
    The transformation creates a "skeleton" of the original sky blue regions,
    marking their inner diagonal structure with red.
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions on a black background.
    
    Returns:
    ColoredGrid: A new grid with red markings representing the inner diagonal skeleton of the original sky blue regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def is_inner_corner(r: int, c: int) -> bool:
        if input_grid.values[r][c] != 8:
            return False
        adjacent_black = sum(1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols and input_grid.values[r+dr][c+dc] == 0)
        return adjacent_black >= 2

    def find_inner_corners(region: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [cell for cell in region if is_inner_corner(*cell)]

    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def connect_corners(corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(corners) <= 1:
            return corners
        connected = [corners[0]]
        remaining = corners[1:]
        while remaining:
            current = connected[-1]
            nearest = min(remaining, key=lambda x: manhattan_distance(current, x))
            connected.append(nearest)
            remaining.remove(nearest)
        return connected

    def draw_line(start: Tuple[int, int], end: Tuple[int, int]) -> None:
        r1, c1 = start
        r2, c2 = end
        dr = 1 if r2 > r1 else -1 if r2 < r1 else 0
        dc = 1 if c2 > c1 else -1 if c2 < c1 else 0
        r, c = r1, c1
        while (r, c) != (r2, c2):
            if input_grid.values[r][c] == 8:
                output_grid.values[r][c] = 2
            r += dr
            c += dc
        if input_grid.values[r2][c2] == 8:
            output_grid.values[r2][c2] = 2

    sky_blue_regions = input_grid.find_connected_regions(8)
    
    for region in sky_blue_regions:
        corners = find_inner_corners(region)
        if corners:
            connected_corners = connect_corners(corners)
            for i in range(len(connected_corners) - 1):
                draw_line(connected_corners[i], connected_corners[i+1])
        elif len(region) > 1:
            # For regions without corners (e.g., straight lines), mark the middle cell
            mid = len(region) // 2
            output_grid.values[region[mid][0]][region[mid][1]] = 2
    
    return output_grid
