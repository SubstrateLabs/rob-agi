from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

import math
from typing import List, Tuple, Set

from typing import List, Tuple, Set
import math

def solve_1c56ad9f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid to create a dynamic wave-like effect on shapes.
    The transformation follows these rules:
    1. Identify all non-zero connected shapes in the grid.
    2. For each shape:
       a. Preserve the overall structure, including internal holes and lines.
       b. Apply a wave-like transformation that shifts columns left and right.
       c. The wave effect is more pronounced in the middle and less at the top and bottom.
       d. Maintain connectivity of all parts of the shape.
    3. Retain the original background (black/0 areas).
    4. Ensure consistent application of the wave-like effect across different colors and shape sizes.
    5. Handle edge cases to prevent out-of-bounds errors and maintain shape integrity.
    This creates a dynamic, wave-like effect on the shapes while preserving their overall structure and internal features.
    """
    result = input_grid.deep_copy()
    shapes = find_shapes(input_grid)
    
    for shape in shapes:
        process_shape(result, shape)
    
    return result

def find_shapes(grid: ColoredGrid) -> List[Tuple[int, int, int, int, int, Set[Tuple[int, int]]]]:
    """Find all shapes (connected non-zero areas) in the grid."""
    shapes = []
    visited = set()
    
    for row in range(grid.num_rows):
        for col in range(grid.num_cols):
            if grid.values[row][col] != 0 and (row, col) not in visited:
                shape = flood_fill(grid, row, col, visited)
                shapes.append(shape)
    
    return shapes

def flood_fill(grid: ColoredGrid, row: int, col: int, visited: set) -> Tuple[int, int, int, int, int, Set[Tuple[int, int]]]:
    """Perform flood fill to find the boundaries and points of a shape."""
    color = grid.values[row][col]
    stack = [(row, col)]
    shape_points = set()
    min_row, max_row, min_col, max_col = row, row, col, col
    
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == color:
            visited.add((r, c))
            shape_points.add((r, c))
            min_row, max_row = min(min_row, r), max(max_row, r)
            min_col, max_col = min(min_col, c), max(max_col, c)
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
    
    return (min_row, max_row, min_col, max_col, color, shape_points)

def process_shape(grid: ColoredGrid, shape: Tuple[int, int, int, int, int, Set[Tuple[int, int]]]):
    """Process a single shape according to the transformation rules."""
    min_row, max_row, min_col, max_col, color, shape_points = shape
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Create a temporary grid for the shape
    temp_grid = [[0 for _ in range(width)] for _ in range(height)]
    for r, c in shape_points:
        temp_grid[r - min_row][c - min_col] = 1
    
    # Apply wave transformation
    transformed_grid = apply_wave_transform(temp_grid, height, width)
    
    # Transfer the transformed shape back to the main grid
    for r in range(height):
        for c in range(width):
            if transformed_grid[r][c] == 1:
                new_r, new_c = r + min_row, c + min_col
                if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
                    grid.values[new_r][new_c] = color
            elif (r + min_row, c + min_col) in shape_points:
                grid.values[r + min_row][c + min_col] = 0

def apply_wave_transform(grid: List[List[int]], height: int, width: int) -> List[List[int]]:
    """Apply a wave-like transformation to the grid."""
    result = [[0 for _ in range(width)] for _ in range(height)]
    center_row = height // 2
    
    for r in range(height):
        vertical_factor = 1 - abs(r - center_row) / center_row
        for c in range(width):
            shift = int(3 * vertical_factor * math.sin(2 * math.pi * c / width))
            new_c = (c + shift) % width
            if grid[r][c] == 1:
                result[r][new_c] = 1
    
    # Ensure vertical connectivity
    for c in range(width):
        for r in range(1, height - 1):
            if result[r][c] == 0 and result[r-1][c] == 1 and result[r+1][c] == 1:
                result[r][c] = 1
    
    return result
