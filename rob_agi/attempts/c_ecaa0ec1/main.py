from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ecaa0ec1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the ecaa0ec1 challenge by reorganizing the colored cells.
    
    1. Find the bounding box of non-black cells.
    2. Count blue (1), sky blue (8), and yellow (4) cells within the bounding box.
    3. Create a 3x3 structure based on color counts:
       - If blue > sky blue: blue corners, sky blue sides and center
       - If sky blue >= blue: sky blue corners, blue sides and center
    4. Place the structure centered within the bounding box.
    5. If yellow exists, place one yellow cell adjacent to the structure.
    6. Clear the rest of the grid.
    7. Return the new grid with the reorganized structure.
    """
    rows, cols = input_grid.get_dimensions()
    bounding_box = find_bounding_box(input_grid)
    if not bounding_box:
        return input_grid.deep_copy()

    color_counts = count_colors(input_grid, bounding_box)
    top_left = calculate_top_left(bounding_box)
    structure = create_3x3_structure(color_counts)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    place_structure(output_grid, structure, top_left)
    
    if color_counts[4] > 0:
        place_yellow(output_grid, top_left)
    
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

def calculate_top_left(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    min_row, min_col, max_row, max_col = bbox
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    return (center_row - 1, center_col - 1)

def create_3x3_structure(color_counts: dict) -> List[List[int]]:
    structure = [[0 for _ in range(3)] for _ in range(3)]
    if color_counts[1] > color_counts[8]:
        corners = 1
        other = 8
    else:
        corners = 8
        other = 1
    
    structure[0][0] = structure[0][2] = structure[2][0] = structure[2][2] = corners
    structure[1][1] = other
    structure[0][1] = structure[1][0] = structure[1][2] = structure[2][1] = other
    
    return structure

def place_structure(grid: ColoredGrid, structure: List[List[int]], top_left: Tuple[int, int]):
    top_row, left_col = top_left
    for r in range(3):
        for c in range(3):
            grid_row = top_row + r
            grid_col = left_col + c
            if 0 <= grid_row < len(grid.values) and 0 <= grid_col < len(grid.values[0]):
                grid.values[grid_row][grid_col] = structure[r][c]

def place_yellow(grid: ColoredGrid, top_left: Tuple[int, int]):
    top_row, left_col = top_left
    directions = [(0, -1), (1, -1), (2, -1), (3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3), (-1, 2), (-1, 1), (-1, 0)]
    for dr, dc in directions:
        r, c = top_row + dr, left_col + dc
        if 0 <= r < len(grid.values) and 0 <= c < len(grid.values[0]) and grid.values[r][c] == 0:
            grid.values[r][c] = 4
            break
