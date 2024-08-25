from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ecaa0ec1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the ecaa0ec1 challenge by reorganizing the colored cells.
    
    1. Find the bounding box of non-black cells.
    2. Calculate the center of the bounding box.
    3. Create a fixed 3x3 structure with sky blue corners and blue center/sides.
    4. Place the structure centered on the calculated center.
    5. If yellow exists in the input, place one yellow cell adjacent to the structure.
    6. Clear the rest of the grid.
    7. Return the new grid with the reorganized structure.
    """
    rows, cols = input_grid.get_dimensions()
    bounding_box = find_bounding_box(input_grid)
    if not bounding_box:
        return input_grid.deep_copy()

    center = calculate_center(bounding_box)
    structure = create_3x3_structure()
    
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    place_structure(output_grid, structure, center)
    
    if has_yellow(input_grid):
        place_yellow(output_grid, center)
    
    return output_grid

def find_bounding_box(grid: ColoredGrid) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    min_row, min_col = rows, cols
    max_row, max_col = -1, -1
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                min_row = min(min_row, r)
                min_col = min(min_col, c)
                max_row = max(max_row, r)
                max_col = max(max_col, c)
    
    return (min_row, min_col, max_row, max_col) if max_row != -1 else None

def count_colors(grid: ColoredGrid, bbox: Tuple[int, int, int, int]) -> dict:
    counts = {1: 0, 4: 0, 8: 0}
    min_row, min_col, max_row, max_col = bbox
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            color = grid.values[r][c]
            if color in counts:
                counts[color] += 1
    return counts

def calculate_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    min_row, min_col, max_row, max_col = bbox
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    return (center_row, center_col)

def create_3x3_structure() -> List[List[int]]:
    return [
        [8, 1, 8],
        [1, 1, 1],
        [8, 1, 8]
    ]

def place_structure(grid: ColoredGrid, structure: List[List[int]], center: Tuple[int, int]):
    center_row, center_col = center
    for r in range(3):
        for c in range(3):
            grid_row = center_row + r - 1
            grid_col = center_col + c - 1
            if 0 <= grid_row < len(grid.values) and 0 <= grid_col < len(grid.values[0]):
                grid.values[grid_row][grid_col] = structure[r][c]

def has_yellow(grid: ColoredGrid) -> bool:
    return any(4 in row for row in grid.values)

def place_yellow(grid: ColoredGrid, center: Tuple[int, int]):
    center_row, center_col = center
    directions = [(-2, 0), (0, -2), (2, 0), (0, 2), (-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)]
    for dr, dc in directions:
        r, c = center_row + dr, center_col + dc
        if 0 <= r < len(grid.values) and 0 <= c < len(grid.values[0]) and grid.values[r][c] == 0:
            grid.values[r][c] = 4
            break
