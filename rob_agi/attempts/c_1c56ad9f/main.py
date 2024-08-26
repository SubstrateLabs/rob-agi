from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1c56ad9f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid to create a 3D bulging effect on shapes.
    The transformation follows these rules:
    1. Identify all non-zero connected shapes in the grid.
    2. For each shape:
       a. Top and bottom rows remain unchanged.
       b. Internal rows are shifted based on their distance from the vertical center:
          - Odd-numbered rows shift left elements to the left.
          - Even-numbered rows shift right elements to the right.
          - Shift magnitude is largest at the center and decreases towards edges.
    3. Maintain vertical connectivity by filling gaps created by shifts.
    4. Preserve shape integrity and handle shape intersections.
    5. Retain the original background (black/0 areas).
    This creates a 3D bulging effect on the shapes while preserving their overall structure and connectivity.
    """
    result = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    shapes = find_shapes(input_grid)
    
    for shape in shapes:
        process_shape(input_grid, result, shape)
    
    # Copy any remaining black areas
    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            if result.values[row][col] == 0:
                result.values[row][col] = input_grid.values[row][col]
    
    return result

def find_shapes(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """Find all shapes (connected non-zero areas) in the grid."""
    shapes = []
    visited = set()
    
    for row in range(grid.num_rows):
        for col in range(grid.num_cols):
            if grid.values[row][col] != 0 and (row, col) not in visited:
                shape = flood_fill(grid, row, col, visited)
                shapes.append(shape)
    
    return shapes

def flood_fill(grid: ColoredGrid, row: int, col: int, visited: set) -> Tuple[int, int, int, int]:
    """Perform flood fill to find the boundaries of a shape."""
    color = grid.values[row][col]
    stack = [(row, col)]
    min_row, max_row, min_col, max_col = row, row, col, col
    
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == color:
            visited.add((r, c))
            min_row, max_row = min(min_row, r), max(max_row, r)
            min_col, max_col = min(min_col, c), max(max_col, c)
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
    
    return (min_row, max_row, min_col, max_col)

def process_shape(input_grid: ColoredGrid, result: ColoredGrid, shape: Tuple[int, int, int, int]):
    """Process a single shape according to the transformation rules."""
    min_row, max_row, min_col, max_col = shape
    
    # Copy top and bottom rows
    result.values[min_row] = input_grid.values[min_row][min_col:max_col+1]
    result.values[max_row] = input_grid.values[max_row][min_col:max_col+1]
    
    # Process internal rows
    for row in range(min_row + 1, max_row):
        row_index = row - min_row
        if row_index % 4 == 1:  # Shift left
            shift_left(input_grid, result, row, min_col, max_col)
        elif row_index % 4 == 3:  # Shift right
            shift_right(input_grid, result, row, min_col, max_col)
        else:  # Copy as-is
            for col in range(min_col, max_col + 1):
                result.values[row][col] = input_grid.values[row][col]
    
    # Maintain vertical connectivity
    maintain_vertical_connectivity(result, shape)

def shift_left(input_grid: ColoredGrid, result: ColoredGrid, row: int, min_col: int, max_col: int):
    """Shift the leftmost non-zero element to the left."""
    for col in range(min_col, max_col + 1):
        if input_grid.values[row][col] != 0:
            if col > 0 and result.values[row][col-1] == 0:
                result.values[row][col-1] = input_grid.values[row][col]
            else:
                result.values[row][col] = input_grid.values[row][col]
            break
    for col in range(col + 1, max_col + 1):
        result.values[row][col] = input_grid.values[row][col]

def shift_right(input_grid: ColoredGrid, result: ColoredGrid, row: int, min_col: int, max_col: int):
    """Shift the rightmost non-zero element to the right."""
    for col in range(max_col, min_col - 1, -1):
        if input_grid.values[row][col] != 0:
            if col < input_grid.num_cols - 1 and result.values[row][col+1] == 0:
                result.values[row][col+1] = input_grid.values[row][col]
            else:
                result.values[row][col] = input_grid.values[row][col]
            break
    for col in range(min_col, col):
        result.values[row][col] = input_grid.values[row][col]

def maintain_vertical_connectivity(result: ColoredGrid, shape: Tuple[int, int, int, int]):
    """Ensure vertical lines remain connected."""
    min_row, max_row, min_col, max_col = shape
    for col in range(min_col, max_col + 1):
        for row in range(min_row, max_row + 1):
            if result.values[row][col] == 0:
                above = result.values[row-1][col] if row > min_row else 0
                below = result.values[row+1][col] if row < max_row else 0
                if above != 0 and below != 0:
                    result.values[row][col] = above
