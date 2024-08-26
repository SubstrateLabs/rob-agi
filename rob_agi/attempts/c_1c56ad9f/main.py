from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

import math
from typing import List, Tuple, Set

def solve_1c56ad9f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid to create a dynamic wave-like effect on shapes.
    The transformation follows these rules:
    1. Identify all non-zero connected shapes in the grid.
    2. For each shape:
       a. Apply a wave function to determine horizontal shifts for each point.
       b. Shifts are larger near the center and smaller near the top and bottom.
       c. Allow slight modifications to top and bottom rows.
       d. Create outward bulges on left and right edges where space permits.
    3. Maintain vertical and horizontal connectivity within shapes.
    4. Preserve shape integrity and handle potential overlaps.
    5. Retain the original background (black/0 areas).
    This creates a dynamic, wave-like effect on the shapes while preserving their overall structure and connectivity.
    """
    result = input_grid.deep_copy()
    shapes = find_shapes(input_grid)
    
    for shape in shapes:
        process_shape(result, shape)
    
    maintain_connectivity(result, shapes)
    
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
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    new_points = {}
    for row, col in shape_points:
        shift = wave_function(row - center_row, col - center_col, height, width)
        new_col = col + shift
        if min_col <= new_col <= max_col:
            new_points[(row, col)] = (row, new_col)
    
    # Apply shifts
    for old, new in new_points.items():
        if grid.values[new[0]][new[1]] == 0:  # Only move if the new position is empty
            grid.values[new[0]][new[1]] = color
            if old != new:
                grid.values[old[0]][old[1]] = 0

def wave_function(dy: float, dx: float, height: int, width: int) -> int:
    """Calculate the horizontal shift based on the point's position within the shape."""
    vertical_factor = 1 - abs(2 * dy / height)
    horizontal_factor = math.cos(2 * math.pi * dx / width)
    shift = int(round(2 * vertical_factor * horizontal_factor))
    return max(-2, min(2, shift))  # Limit the shift to a maximum of 2 in either direction

def maintain_connectivity(grid: ColoredGrid, shapes: List[Tuple[int, int, int, int, int, Set[Tuple[int, int]]]]):
    """Ensure shapes remain connected after transformation."""
    for shape in shapes:
        min_row, max_row, min_col, max_col, color, _ = shape
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if grid.values[row][col] == color:
                    neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    for nr, nc in neighbors:
                        if min_row <= nr <= max_row and min_col <= nc <= max_col and grid.values[nr][nc] == 0:
                            if any(grid.values[r][c] == color for r, c in [(nr-1, nc), (nr+1, nc), (nr, nc-1), (nr, nc+1)] if (r, c) != (row, col)):
                                grid.values[nr][nc] = color
