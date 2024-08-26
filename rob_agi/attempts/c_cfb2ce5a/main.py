from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_cfb2ce5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding each color into the largest possible rectangle
    while preserving the original pattern and maintaining a black border.

    The algorithm works as follows:
    1. Initialize a copy of the input grid.
    2. For each unique non-zero color:
       a. Find the initial bounding box of the color.
       b. Expand the bounding box in all directions until it hits a border or another color.
       c. Fill the expanded bounding box with the color, preserving original non-zero values.
    3. Repeat step 2 until no further expansion is possible.
    4. Fill any remaining zero cells with the nearest non-zero neighbor's color.
    5. Return the modified grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion algorithm.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def get_unique_colors() -> List[int]:
        return list(set(grid.values[r][c] for r in range(rows) for c in range(cols) if grid.values[r][c] != 0))
    
    def find_bounding_box(color: int) -> Tuple[int, int, int, int]:
        top = min((r for r in range(rows) for c in range(cols) if grid.values[r][c] == color), default=rows)
        bottom = max((r for r in range(rows) for c in range(cols) if grid.values[r][c] == color), default=-1)
        left = min((c for r in range(rows) for c in range(cols) if grid.values[r][c] == color), default=cols)
        right = max((c for r in range(rows) for c in range(cols) if grid.values[r][c] == color), default=-1)
        return top, left, bottom, right
    
    def expand_bounding_box(color: int, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        top, left, bottom, right = box
        while top > 0 and all(grid.values[top-1][c] in (0, color) for c in range(left, right+1)):
            top -= 1
        while bottom < rows-1 and all(grid.values[bottom+1][c] in (0, color) for c in range(left, right+1)):
            bottom += 1
        while left > 0 and all(grid.values[r][left-1] in (0, color) for r in range(top, bottom+1)):
            left -= 1
        while right < cols-1 and all(grid.values[r][right+1] in (0, color) for r in range(top, bottom+1)):
            right += 1
        return top, left, bottom, right
    
    def fill_bounding_box(color: int, box: Tuple[int, int, int, int]):
        top, left, bottom, right = box
        for r in range(top, bottom+1):
            for c in range(left, right+1):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
    
    def fill_remaining_zeros():
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if grid.values[r][c] == 0:
                    neighbors = [grid.values[r+dr][c+dc] for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr != 0 or dc != 0)]
                    non_zero_neighbors = [n for n in neighbors if n != 0]
                    if non_zero_neighbors:
                        grid.values[r][c] = non_zero_neighbors[0]
    
    changed = True
    while changed:
        changed = False
        for color in get_unique_colors():
            old_box = find_bounding_box(color)
            new_box = expand_bounding_box(color, old_box)
            if new_box != old_box:
                fill_bounding_box(color, new_box)
                changed = True
    
    fill_remaining_zeros()
    return grid
