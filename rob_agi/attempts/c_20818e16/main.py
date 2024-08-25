from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_20818e16(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by simplifying and rearranging colored regions.
    
    1. Identify significant color regions, excluding the background.
    2. Simplify each region into a rectangle, maintaining relative area proportions.
    3. Arrange simplified rectangles in a compact manner, prioritizing larger regions.
    4. Remove background color and optimize grid size.
    5. Ensure all colors (except background) are represented in the output.
    
    Returns a new ColoredGrid with the transformed and simplified arrangement.
    """
    color_regions = identify_color_regions(input_grid)
    simplified_shapes = simplify_shapes(color_regions, input_grid.get_dimensions())
    arranged_grid = arrange_shapes(simplified_shapes)
    optimized_grid = optimize_grid(arranged_grid)
    return ColoredGrid(values=optimized_grid)

def identify_color_regions(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    color_regions = {}
    rows, cols = grid.get_dimensions()
    background_color = most_common_color(grid)
    
    for r in range(rows):
        for c in range(cols):
            color = grid.values[r][c]
            if color != background_color:
                if color not in color_regions:
                    color_regions[color] = []
                color_regions[color].append((r, c))
    
    return color_regions

def most_common_color(grid: ColoredGrid) -> int:
    color_count = {}
    for row in grid.values:
        for color in row:
            color_count[color] = color_count.get(color, 0) + 1
    return max(color_count, key=color_count.get)

def simplify_shapes(color_regions: Dict[int, List[Tuple[int, int]]], grid_dimensions: Tuple[int, int]) -> List[Tuple[int, int, int]]:
    total_area = sum(len(region) for region in color_regions.values())
    simplified_shapes = []
    
    for color, region in color_regions.items():
        area = len(region)
        relative_area = area / total_area
        side_length = int(math.sqrt(relative_area * grid_dimensions[0] * grid_dimensions[1]))
        simplified_shapes.append((color, side_length, side_length))
    
    return sorted(simplified_shapes, key=lambda x: x[1] * x[2], reverse=True)

def arrange_shapes(shapes: List[Tuple[int, int, int]]) -> List[List[int]]:
    total_area = sum(w * h for _, w, h in shapes)
    grid_size = math.ceil(math.sqrt(total_area))
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    def can_place(r, c, w, h):
        if r + h > grid_size or c + w > grid_size:
            return False
        return all(grid[i][j] == 0 for i in range(r, r + h) for j in range(c, c + w))
    
    for color, w, h in shapes:
        placed = False
        for r in range(grid_size):
            for c in range(grid_size):
                if can_place(r, c, w, h):
                    for i in range(r, r + h):
                        for j in range(c, c + w):
                            grid[i][j] = color
                    placed = True
                    break
            if placed:
                break
    
    return grid

def optimize_grid(grid: List[List[int]]) -> List[List[int]]:
    rows = [r for r, row in enumerate(grid) if any(cell != 0 for cell in row)]
    cols = [c for c in range(len(grid[0])) if any(row[c] != 0 for row in grid)]
    return [[grid[r][c] for c in cols] for r in rows]
