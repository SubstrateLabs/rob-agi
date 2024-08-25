from rob_agi.colored_grid import ColoredGrid
from typing import Dict, Tuple

def solve_3194b014(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 3194b014 challenge by finding the most significant square-like region in the input grid
    and creating a 3x3 grid filled with the color of that region.

    The solution follows these steps:
    1. Scan the input grid and perform a flood fill for each unvisited cell.
    2. For each flood fill, analyze the region to find the largest possible square-like area.
    3. Keep track of the best square-like region for each color.
    4. Select the color with the largest square-like region as the most significant.
    5. Create and return a 3x3 grid filled with the most significant color.

    Args:
    input_grid (ColoredGrid): The input grid to be processed.

    Returns:
    ColoredGrid: A 3x3 grid filled with the most significant color.
    """
    best_squares = find_best_squares(input_grid)
    most_significant_color = max(best_squares, key=lambda c: best_squares[c]['side_length'])
    return ColoredGrid(values=[[most_significant_color for _ in range(3)] for _ in range(3)])

def find_best_squares(grid: ColoredGrid) -> Dict[int, Dict]:
    rows, cols = grid.get_dimensions()
    visited = set()
    best_squares = {}

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                color = grid.get_cell(r, c)
                region_info = flood_fill(grid, r, c, color, visited)
                update_best_square(best_squares, color, region_info)

    return best_squares

def flood_fill(grid: ColoredGrid, r: int, c: int, color: int, visited: set) -> Dict:
    stack = [(r, c)]
    region = set()
    min_x, min_y, max_x, max_y = c, r, c, r

    while stack:
        curr_r, curr_c = stack.pop()
        if (curr_r, curr_c) in visited or grid.get_cell(curr_r, curr_c) != color:
            continue

        visited.add((curr_r, curr_c))
        region.add((curr_r, curr_c))
        min_x, max_x = min(min_x, curr_c), max(max_x, curr_c)
        min_y, max_y = min(min_y, curr_r), max(max_y, curr_r)

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                stack.append((nr, nc))

    return {
        'total_cells': len(region),
        'width': max_x - min_x + 1,
        'height': max_y - min_y + 1
    }

def update_best_square(best_squares: Dict[int, Dict], color: int, region_info: Dict):
    side_length = min(region_info['width'], region_info['height'])
    squareness = region_info['total_cells'] / (side_length ** 2)

    if squareness >= 0.8 and (color not in best_squares or side_length > best_squares[color]['side_length']):
        best_squares[color] = {
            'side_length': side_length,
            'squareness': squareness
        }
