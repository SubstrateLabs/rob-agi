from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring specific 'L'-shaped or extended 'L'-shaped
    regions of black (0) squares to green (3). The algorithm follows these steps:
    1. Check corners and edges for L-shapes
    2. Spiral inward, identifying and transforming L-shapes
    3. Ensure a balanced distribution of green shapes
    4. Make a final pass to catch any missed opportunities

    The transformation aims to create a balanced distribution of green shapes while maintaining
    the overall aesthetic of the grid, prioritizing edge-to-center progression.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    transformed_shapes = []

    def is_valid_L_shape(r: int, c: int, size: int) -> bool:
        if size == 3:
            shape = [(r, c), (r+1, c), (r, c+1)]
        else:  # size == 5
            shape = [(r, c), (r+1, c), (r+2, c), (r, c+1), (r, c+2)]
        
        if not all(0 <= x < rows and 0 <= y < cols for x, y in shape):
            return False
        
        if not all(output_grid.get_cell(x, y) == 0 for x, y in shape):
            return False
        
        neighbors = set((x+dx, y+dy) for x, y in shape for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)])
        neighbors -= set(shape)
        return all(not (0 <= nx < rows and 0 <= ny < cols) or output_grid.get_cell(nx, ny) in [0, 8] for nx, ny in neighbors)

    def transform_L_shape(r: int, c: int, size: int):
        shape = [(r, c), (r+1, c), (r, c+1)] if size == 3 else [(r, c), (r+1, c), (r+2, c), (r, c+1), (r, c+2)]
        for x, y in shape:
            output_grid.set_cell(x, y, 3)
        transformed_shapes.append(shape)

    def check_and_transform(r: int, c: int):
        if is_valid_L_shape(r, c, 5):
            transform_L_shape(r, c, 5)
            return True
        elif is_valid_L_shape(r, c, 3):
            transform_L_shape(r, c, 3)
            return True
        return False

    # Check corners
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    for r, c in corners:
        check_and_transform(r, c)

    # Check edges
    for r in range(1, rows-1):
        check_and_transform(r, 0)
        check_and_transform(r, cols-1)
    for c in range(1, cols-1):
        check_and_transform(0, c)
        check_and_transform(rows-1, c)

    # Spiral inward
    top, bottom, left, right = 1, rows-2, 1, cols-2
    while top <= bottom and left <= right:
        for c in range(left, right+1):
            if check_and_transform(top, c):
                break
        top += 1

        for r in range(top, bottom+1):
            if check_and_transform(r, right):
                break
        right -= 1

        if top <= bottom:
            for c in range(right, left-1, -1):
                if check_and_transform(bottom, c):
                    break
            bottom -= 1

        if left <= right:
            for r in range(bottom, top-1, -1):
                if check_and_transform(r, left):
                    break
            left += 1

    # Final pass
    if len(transformed_shapes) < 2:
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0 and check_and_transform(r, c):
                    if len(transformed_shapes) >= 2:
                        break
            if len(transformed_shapes) >= 2:
                break

    return output_grid
