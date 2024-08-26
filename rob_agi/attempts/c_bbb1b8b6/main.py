from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 4x4 grid based on the following rules:
    
    1. Extract the left half of the input grid (before the gray column).
    2. Create a 4x4 base grid from the left half, padding or truncating as needed.
    3. If the left half forms a complete shape, return the base grid as is.
    4. If the left half is incomplete, fill internal black cells with colors from the right half.
    5. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    # Step 1: Extract left half
    gray_index = input_grid.values[0].index(5)
    left_half = [row[:gray_index] for row in input_grid.values]

    # Step 2: Create 4x4 base grid
    base_grid = create_base_grid(left_half)

    # Step 3: Check if left half forms a complete shape
    if is_complete_shape(base_grid):
        return ColoredGrid(values=base_grid)

    # Step 4: Fill internal black cells from right half
    right_half = [row[gray_index+1:] for row in input_grid.values]
    filled_grid = fill_internal_cells(base_grid, right_half)

    # Step 5: Return result
    return ColoredGrid(values=filled_grid)

def create_base_grid(left_half: List[List[int]]) -> List[List[int]]:
    base_grid = []
    for i in range(4):
        if i < len(left_half):
            row = left_half[i][:4]  # Take up to 4 elements
            row += [0] * (4 - len(row))  # Pad with zeros if needed
        else:
            row = [0] * 4  # Add empty rows if left_half has fewer than 4 rows
        base_grid.append(row)
    return base_grid

def is_complete_shape(grid: List[List[int]]) -> bool:
    non_black = next((cell for row in grid for cell in row if cell != 0), None)
    if non_black is None:
        return True  # All black is considered complete

    visited = set()
    flood_fill(grid, 0, 0, non_black, visited)

    return all(cell == 0 or (r, c) in visited 
               for r, row in enumerate(grid) 
               for c, cell in enumerate(row))

def flood_fill(grid: List[List[int]], r: int, c: int, color: int, visited: set):
    if (r < 0 or r >= 4 or c < 0 or c >= 4 or 
        (r, c) in visited or grid[r][c] != color):
        return

    visited.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        flood_fill(grid, r + dr, c + dc, color, visited)

def is_internal_cell(grid: List[List[int]], r: int, c: int) -> bool:
    if grid[r][c] != 0:
        return False
    return all(0 <= r+dr < 4 and 0 <= c+dc < 4 and grid[r+dr][c+dc] != 0
               for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)])

def fill_internal_cells(base_grid: List[List[int]], right_half: List[List[int]]) -> List[List[int]]:
    filled_grid = [row[:] for row in base_grid]
    for r in range(4):
        for c in range(4):
            if is_internal_cell(base_grid, r, c):
                if r < len(right_half) and c < len(right_half[r]) and right_half[r][c] != 0:
                    filled_grid[r][c] = right_half[r][c]
    return filled_grid
