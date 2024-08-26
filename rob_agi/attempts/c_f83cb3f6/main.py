from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_f83cb3f6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving dots of a specific color to form line(s) adjacent to a base line.
    
    1. Identifies the base line structure of color 8 (sky blue).
    2. Determines the most frequent non-zero, non-8 color as the color to be moved.
    3. Creates new line(s) adjacent to the base line:
       - For horizontal base line: one line above and one below.
       - For vertical base line: one line to the right.
    4. Distributes the moved color along the new line(s), prioritizing:
       - Placement near the ends of base line segments
       - Creating longer continuous lines or clusters
       - Maintaining the structure of the base line
    5. Clears the rest of the grid, keeping only the base line and new line(s) with moved dots.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    base_line_structure = find_base_line_structure(input_grid)
    color_to_move = find_most_frequent_color(input_grid, exclude=[0, 8])
    return create_output_grid(input_grid, base_line_structure, color_to_move)

def find_base_line_structure(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) == 8]

def find_most_frequent_color(grid: ColoredGrid, exclude: List[int]) -> int:
    color_counter = Counter(grid.get_cell(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) 
                            if grid.get_cell(r, c) not in exclude)
    return color_counter.most_common(1)[0][0]

def create_output_grid(input_grid: ColoredGrid, base_line_structure: List[Tuple[int, int]], color_to_move: int) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy base line to output grid
    for r, c in base_line_structure:
        output_grid.set_cell(r, c, 8)
    
    orientation = 'horizontal' if all(r == base_line_structure[0][0] for r, _ in base_line_structure) else 'vertical'
    total_dots = sum(1 for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == color_to_move)
    
    if orientation == 'horizontal':
        base_row = base_line_structure[0][0]
        distribute_dots_horizontal(output_grid, base_row, total_dots, color_to_move)
    else:
        base_col = base_line_structure[0][1]
        distribute_dots_vertical(output_grid, base_col, total_dots, color_to_move)
    
    return output_grid

def distribute_dots_horizontal(grid: ColoredGrid, base_row: int, total_dots: int, color: int):
    cols = grid.num_cols
    above_line = [0] * cols
    below_line = [0] * cols
    
    # Mark positions to skip
    for c in range(cols):
        if grid.get_cell(base_row, c) == 8:
            above_line[c] = below_line[c] = -1
    
    # Prioritize placement near ends and in the middle
    priority_positions = [0, cols-1, cols//2] + [i for i in range(1, cols-1) if i != cols//2]
    
    dots_placed = 0
    for pos in priority_positions:
        if dots_placed >= total_dots:
            break
        if below_line[pos] != -1:
            below_line[pos] = color
            dots_placed += 1
        if dots_placed >= total_dots:
            break
        if above_line[pos] != -1:
            above_line[pos] = color
            dots_placed += 1
    
    # Transfer to output grid
    for c in range(cols):
        if above_line[c] == color:
            grid.set_cell(base_row - 1, c, color)
        if below_line[c] == color:
            grid.set_cell(base_row + 1, c, color)

def distribute_dots_vertical(grid: ColoredGrid, base_col: int, total_dots: int, color: int):
    rows = grid.num_rows
    right_line = [0] * rows
    
    # Mark positions to skip
    for r in range(rows):
        if grid.get_cell(r, base_col) == 8:
            right_line[r] = -1
    
    # Find the last row with a base line cell
    last_base_row = max(r for r in range(rows) if grid.get_cell(r, base_col) == 8)
    
    # Prioritize placement near the top and middle
    priority_positions = [0, last_base_row//2] + [i for i in range(1, last_base_row+1) if i != last_base_row//2]
    
    dots_placed = 0
    for pos in priority_positions:
        if dots_placed >= total_dots or pos > last_base_row:
            break
        if right_line[pos] != -1:
            right_line[pos] = color
            dots_placed += 1
    
    # Transfer to output grid
    for r in range(rows):
        if right_line[r] == color:
            grid.set_cell(r, base_col + 1, color)
