from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_ecaa0ec1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the ecaa0ec1 challenge by reorganizing the colored cells.
    
    1. Analyze the input grid to find the bounding box and center of non-black cells.
    2. Identify or create a valid 3x3 structure with blue (1) and sky blue (8) cells.
    3. Place the 3x3 structure near the center of the non-black cells.
    4. If yellow (4) exists in the input, place one yellow cell adjacent to the structure.
    5. Clear all other cells to black (0).
    6. Return the new grid with the reorganized structure.
    """
    rows, cols = input_grid.get_dimensions()
    bounding_box = find_bounding_box(input_grid)
    if not bounding_box:
        return input_grid.deep_copy()

    center = calculate_center(bounding_box)
    existing_structure = find_existing_structure(input_grid, center)
    structure = create_valid_structure(existing_structure)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
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

def create_valid_structure(existing_structure: List[List[int]]) -> List[List[int]]:
    valid_structures = [
        [[1, 8, 1], [8, 1, 1], [1, 1, 8]],
        [[1, 8, 1], [1, 1, 1], [1, 8, 8]],
        [[1, 8, 1], [8, 1, 8], [1, 8, 1]]
    ]
    
    # Count matching cells with each valid structure
    matches = [sum(existing_structure[r][c] == struct[r][c] for r in range(3) for c in range(3))
               for struct in valid_structures]
    
    # Return the valid structure with the most matches
    return valid_structures[matches.index(max(matches))]

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

def place_yellow(grid: ColoredGrid, center: Tuple[int, int]):
    center_row, center_col = center
    rows, cols = grid.get_dimensions()
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in directions:
        r, c = center_row + dr, center_col + dc
        if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 0:
            grid.values[r][c] = 4
            break
