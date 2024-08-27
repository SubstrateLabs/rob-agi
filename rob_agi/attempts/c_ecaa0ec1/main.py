from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

from typing import List, Tuple, Optional

def solve_ecaa0ec1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the ecaa0ec1 challenge by reorganizing the colored cells.
    
    1. Find the bounding box of non-black cells in the input grid.
    2. Extract a 3x3 structure from the center of the bounding box.
    3. Validate and adjust the 3x3 structure to have at least 3 blue (1) and 2 sky blue (8) cells.
    4. Create a new grid and place the 3x3 structure at its center.
    5. If yellow (4) exists in the input, place one yellow cell adjacent to the structure.
    6. Ensure all other cells are black (0).
    7. Return the new grid with the reorganized structure.
    """
    bbox = find_bounding_box(input_grid)
    if not bbox:
        return input_grid.deep_copy()

    structure = extract_and_validate_structure(input_grid, bbox)
    
    output_grid = create_output_grid(input_grid.get_dimensions())
    place_structure(output_grid, structure)
    
    if has_yellow(input_grid):
        place_yellow(output_grid)
    
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

def extract_and_validate_structure(grid: ColoredGrid, bbox: Tuple[int, int, int, int]) -> List[List[int]]:
    min_row, min_col, max_row, max_col = bbox
    center_row, center_col = (min_row + max_row) // 2, (min_col + max_col) // 2
    structure = [[0 for _ in range(3)] for _ in range(3)]
    
    for r in range(3):
        for c in range(3):
            grid_row, grid_col = center_row + r - 1, center_col + c - 1
            if 0 <= grid_row < grid.num_rows and 0 <= grid_col < grid.num_cols:
                structure[r][c] = grid.values[grid_row][grid_col]
    
    blue_count = sum(cell == 1 for row in structure for cell in row)
    sky_blue_count = sum(cell == 8 for row in structure for cell in row)
    
    # Ensure at least 3 blue cells
    while blue_count < 3:
        for r in range(3):
            for c in range(3):
                if structure[r][c] not in (1, 8):
                    structure[r][c] = 1
                    blue_count += 1
                    if blue_count == 3:
                        break
            if blue_count == 3:
                break
    
    # Ensure at least 2 sky blue cells
    while sky_blue_count < 2:
        for r in range(3):
            for c in range(3):
                if structure[r][c] not in (1, 8):
                    structure[r][c] = 8
                    sky_blue_count += 1
                    if sky_blue_count == 2:
                        break
            if sky_blue_count == 2:
                break
    
    # Fill remaining cells with blue
    for r in range(3):
        for c in range(3):
            if structure[r][c] not in (1, 8):
                structure[r][c] = 1
    
    return structure

def create_output_grid(dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    return ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

def place_structure(grid: ColoredGrid, structure: List[List[int]]):
    center_row, center_col = grid.num_rows // 2, grid.num_cols // 2
    for r in range(3):
        for c in range(3):
            grid_row, grid_col = center_row + r - 1, center_col + c - 1
            if 0 <= grid_row < grid.num_rows and 0 <= grid_col < grid.num_cols:
                grid.values[grid_row][grid_col] = structure[r][c]

def has_yellow(grid: ColoredGrid) -> bool:
    return any(4 in row for row in grid.values)

def place_yellow(grid: ColoredGrid):
    center_row, center_col = grid.num_rows // 2, grid.num_cols // 2
    directions = [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1), (2, 0)]
    for dr, dc in directions:
        r, c = center_row + dr, center_col + dc
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == 0:
            grid.values[r][c] = 4
            break
