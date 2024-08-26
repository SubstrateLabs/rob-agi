from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_ecaa0ec1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the ecaa0ec1 challenge by reorganizing the colored cells.
    
    1. Analyze the input grid to find the bounding box of non-black cells.
    2. Identify or create a valid 3x3 structure with blue (1) and sky blue (8) cells.
    3. Place the 3x3 structure centered on the bounding box.
    4. If yellow (4) exists in the input, place one yellow cell adjacent to the structure.
    5. Clear all other cells to black (0).
    6. Return the new grid with the reorganized structure.
    """
    rows, cols = input_grid.get_dimensions()
    bounding_box = find_bounding_box(input_grid)
    if not bounding_box:
        return input_grid.deep_copy()

    center = calculate_center(bounding_box)
    structure = find_or_create_structure(input_grid, center)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    place_structure(output_grid, structure, center)
    
    if has_yellow(input_grid):
        place_yellow(output_grid, structure, center)
    
    return output_grid

def find_bounding_box(grid: ColoredGrid) -> Optional[Tuple[int, int, int, int]]:
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

def calculate_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    min_row, min_col, max_row, max_col = bbox
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    return (center_row, center_col)

def find_or_create_structure(grid: ColoredGrid, center: Tuple[int, int]) -> List[List[int]]:
    existing = find_existing_structure(grid, center)
    if is_valid_structure(existing):
        return existing
    return create_valid_structure()

def find_existing_structure(grid: ColoredGrid, center: Tuple[int, int]) -> List[List[int]]:
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    structure = [[0 for _ in range(3)] for _ in range(3)]
    
    for r in range(3):
        for c in range(3):
            grid_row = center_row + r - 1
            grid_col = center_col + c - 1
            if 0 <= grid_row < rows and 0 <= grid_col < cols:
                structure[r][c] = grid.values[grid_row][grid_col]
    
    return structure

def is_valid_structure(structure: List[List[int]]) -> bool:
    blue_count = sum(row.count(1) for row in structure)
    sky_blue_count = sum(row.count(8) for row in structure)
    return blue_count >= 3 and sky_blue_count >= 2 and blue_count + sky_blue_count == 9

def create_valid_structure() -> List[List[int]]:
    return [[8, 8, 1], [1, 8, 1], [8, 1, 1]]

def place_structure(grid: ColoredGrid, structure: List[List[int]], center: Tuple[int, int]):
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    for r in range(3):
        for c in range(3):
            grid_row = center_row + r - 1
            grid_col = center_col + c - 1
            if 0 <= grid_row < rows and 0 <= grid_col < cols:
                grid.values[grid_row][grid_col] = structure[r][c]

def has_yellow(grid: ColoredGrid) -> bool:
    return any(4 in row for row in grid.values)

def place_yellow(grid: ColoredGrid, structure: List[List[int]], center: Tuple[int, int]):
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    directions = [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1), (2, 0)]
    for dr, dc in directions:
        r, c = center_row + dr, center_col + dc
        if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 0:
            grid.values[r][c] = 4
            break
