from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 4x4 grid based on the following rules:
    
    1. Extract the left and right halves of the input grid (split by the gray column).
    2. Create a 4x4 base grid from the left half.
    3. Complete the shape by combining information from both halves:
       - Use the left half as the base shape
       - Fill in missing parts using the right half, maintaining symmetry
       - If the right half doesn't provide enough information, use the main color from the left half
    4. Ensure the final shape touches all four edges of the 4x4 grid
    5. Return the resulting 4x4 grid as a ColoredGrid object
    """
    # Step 1: Extract left and right halves
    gray_index = input_grid.values[0].index(5)
    left_half = [row[:gray_index] for row in input_grid.values]
    right_half = [row[gray_index+1:] for row in input_grid.values]

    # Step 2: Create 4x4 base grid from left half
    base_grid = create_base_grid(left_half)

    # Step 3 & 4: Complete the shape
    completed_grid = complete_shape(base_grid, right_half)

    # Step 5: Return result
    return ColoredGrid(values=completed_grid)

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

def complete_shape(base_grid: List[List[int]], right_half: List[List[int]]) -> List[List[int]]:
    completed_grid = [row[:] for row in base_grid]
    main_color = get_main_color(base_grid)

    # Complete horizontally
    for r in range(4):
        left_edge = next((c for c in range(4) if base_grid[r][c] != 0), None)
        right_edge = next((c for c in range(3, -1, -1) if base_grid[r][c] != 0), None)
        if left_edge is not None and right_edge is not None:
            for c in range(left_edge + 1, right_edge):
                completed_grid[r][c] = get_fill_color(r, c, right_half, main_color)

    # Complete vertically
    for c in range(4):
        top_edge = next((r for r in range(4) if completed_grid[r][c] != 0), None)
        bottom_edge = next((r for r in range(3, -1, -1) if completed_grid[r][c] != 0), None)
        if top_edge is not None and bottom_edge is not None:
            for r in range(top_edge + 1, bottom_edge):
                completed_grid[r][c] = get_fill_color(r, c, right_half, main_color)

    return completed_grid

def get_main_color(grid: List[List[int]]) -> int:
    colors = [cell for row in grid for cell in row if cell != 0]
    return max(set(colors), key=colors.count) if colors else 1  # Default to blue if no colors

def get_fill_color(r: int, c: int, right_half: List[List[int]], main_color: int) -> int:
    if r < len(right_half) and c < len(right_half[r]):
        return right_half[r][c] if right_half[r][c] != 0 else main_color
    return main_color
