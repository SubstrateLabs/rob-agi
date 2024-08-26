from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

from typing import List, Tuple, Optional

def solve_ecaa0ec1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the ecaa0ec1 challenge by reorganizing the colored cells.
    
    1. Find the bounding box of non-black cells in the input grid.
    2. Calculate the center of this bounding box.
    3. Extract or create a valid 3x3 structure with blue (1) and sky blue (8) cells.
    4. Create a new grid and place the 3x3 structure at its center.
    5. If yellow (4) exists in the input, place one yellow cell adjacent to the structure.
    6. Ensure all other cells are black (0).
    7. Return the new grid with the reorganized structure.
    """
    bbox = find_bounding_box(input_grid)
    if not bbox:
        return input_grid.deep_copy()

    center = calculate_center(bbox)
    structure = extract_or_create_structure(input_grid, center)
    
    output_grid = create_output_grid(input_grid.get_dimensions())
    place_structure(output_grid, structure, center)
    
    if has_yellow(input_grid):
        place_yellow(output_grid, center)
    
    return output_grid

def find_bounding_box(grid: ColoredGrid) -> Optional[Tuple[int, int, int, int]]:
    rows, cols = grid.get_dimensions()
    min_row, min_col = rows, cols
    max_row, max_col = -1, -1
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                min_row, max_row = min(min_row, r), max(max_row, r)
                min_col, max_col = min(min_col, c), max(max_col, c)
    
    return (min_row, min_col, max_row, max_col) if max_row != -1 else None

def calculate_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    min_row, min_col, max_row, max_col = bbox
    return ((min_row + max_row) // 2, (min_col + max_col) // 2)

def extract_or_create_structure(grid: ColoredGrid, center: Tuple[int, int]) -> List[List[int]]:
    structure = extract_structure(grid, center)
    if is_valid_structure(structure):
        return structure
    return create_valid_structure(structure)

def extract_structure(grid: ColoredGrid, center: Tuple[int, int]) -> List[List[int]]:
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    structure = [[0 for _ in range(3)] for _ in range(3)]
    
    for r in range(3):
        for c in range(3):
            grid_row, grid_col = center_row + r - 1, center_col + c - 1
            if 0 <= grid_row < rows and 0 <= grid_col < cols:
                structure[r][c] = grid.values[grid_row][grid_col]
    
    return structure

def is_valid_structure(structure: List[List[int]]) -> bool:
    blue_count = sum(cell == 1 for row in structure for cell in row)
    sky_blue_count = sum(cell == 8 for row in structure for cell in row)
    return blue_count >= 3 and sky_blue_count >= 2 and blue_count + sky_blue_count == 9

def create_valid_structure(existing: List[List[int]]) -> List[List[int]]:
    blue_count = sum(cell == 1 for row in existing for cell in row)
    sky_blue_count = sum(cell == 8 for row in existing for cell in row)
    
    new_structure = [row[:] for row in existing]
    
    for r in range(3):
        for c in range(3):
            if new_structure[r][c] not in (1, 8):
                if blue_count < 3:
                    new_structure[r][c] = 1
                    blue_count += 1
                elif sky_blue_count < 2:
                    new_structure[r][c] = 8
                    sky_blue_count += 1
                else:
                    new_structure[r][c] = 1
    
    return new_structure

def create_output_grid(dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    return ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

def place_structure(grid: ColoredGrid, structure: List[List[int]], center: Tuple[int, int]):
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    for r in range(3):
        for c in range(3):
            grid_row, grid_col = center_row + r - 1, center_col + c - 1
            if 0 <= grid_row < rows and 0 <= grid_col < cols:
                grid.values[grid_row][grid_col] = structure[r][c]

def has_yellow(grid: ColoredGrid) -> bool:
    return any(4 in row for row in grid.values)

def place_yellow(grid: ColoredGrid, center: Tuple[int, int]):
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    directions = [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1), (2, 0)]
    for dr, dc in directions:
        r, c = center_row + dr, center_col + dc
        if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 0:
            grid.values[r][c] = 4
            break
