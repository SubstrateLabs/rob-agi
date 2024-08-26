from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_d56f2372(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by finding the topmost, then leftmost, complete non-edge-touching shape in the input grid.
    
    1. Scans the grid from top to bottom, left to right.
    2. Identifies the first complete shape (not touching edges and all pixels connected).
    3. Extracts the shape into a new grid, maintaining its relative position and original color.
    4. If no complete shape is found, returns a 1x1 grid with value 0.
    
    Args:
        input_grid (ColoredGrid): The input grid to process.
    
    Returns:
        ColoredGrid: A new grid containing only the extracted shape or a 1x1 grid with value 0.
    """
    visited = set()
    valid_shape = None
    valid_shape_top = float('inf')
    valid_shape_left = float('inf')

    for y in range(input_grid.num_rows):
        for x in range(input_grid.num_cols):
            if (x, y) not in visited and input_grid.values[y][x] != 0:
                shape_pixels = flood_fill(input_grid, x, y, input_grid.values[y][x])
                if is_shape_complete(shape_pixels, (input_grid.num_rows, input_grid.num_cols)):
                    shape_top = min(y for _, y in shape_pixels)
                    shape_left = min(x for x, _ in shape_pixels)
                    if shape_top < valid_shape_top or (shape_top == valid_shape_top and shape_left < valid_shape_left):
                        valid_shape = shape_pixels
                        valid_shape_top = shape_top
                        valid_shape_left = shape_left
                visited.update(shape_pixels)

    if valid_shape:
        return extract_shape(valid_shape, input_grid)
    return ColoredGrid(values=[[0]])

def flood_fill(grid: ColoredGrid, start_x: int, start_y: int, color: int) -> Set[Tuple[int, int]]:
    stack = [(start_x, start_y)]
    shape_pixels = set()
    while stack:
        x, y = stack.pop()
        if (x, y) in shape_pixels or x < 0 or x >= grid.num_cols or y < 0 or y >= grid.num_rows or grid.values[y][x] != color:
            continue
        shape_pixels.add((x, y))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            stack.append((x + dx, y + dy))
    return shape_pixels

def is_shape_complete(shape_pixels: Set[Tuple[int, int]], grid_dimensions: Tuple[int, int]) -> bool:
    rows, cols = grid_dimensions
    for x, y in shape_pixels:
        if x == 0 or x == cols - 1 or y == 0 or y == rows - 1:
            return False
    return True

def extract_shape(shape_pixels: Set[Tuple[int, int]], original_grid: ColoredGrid) -> ColoredGrid:
    min_x = min(x for x, _ in shape_pixels)
    max_x = max(x for x, _ in shape_pixels)
    min_y = min(y for _, y in shape_pixels)
    max_y = max(y for _, y in shape_pixels)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    color = original_grid.values[next(iter(shape_pixels))[1]][next(iter(shape_pixels))[0]]
    for x, y in shape_pixels:
        new_grid.values[y - min_y][x - min_x] = color
    
    return new_grid
