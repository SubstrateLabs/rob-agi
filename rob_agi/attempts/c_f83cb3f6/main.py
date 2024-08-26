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
    4. Evenly distributes the moved color along the new line(s), maintaining the structure of the base line.
    5. Clears the rest of the grid, keeping only the base line and new line(s) with moved dots.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    # Step 1: Identify the base line structure
    base_line_structure = find_base_line_structure(input_grid)
    
    # Step 2: Identify the color to be moved
    color_to_move = find_most_frequent_color(input_grid, exclude=[0, 8])
    
    # Step 3-5: Create new grid with moved dots
    output_grid = create_output_grid(input_grid, base_line_structure, color_to_move)
    
    return output_grid

def find_base_line_structure(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    base_line = []
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                base_line.append((r, c))
    return base_line

def find_most_frequent_color(grid: ColoredGrid, exclude: List[int]) -> int:
    color_counter = Counter()
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color not in exclude:
                color_counter[color] += 1
    return color_counter.most_common(1)[0][0]

def create_output_grid(input_grid: ColoredGrid, base_line_structure: List[Tuple[int, int]], color_to_move: int) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Determine orientation
    orientation = 'horizontal' if all(r == base_line_structure[0][0] for r, _ in base_line_structure) else 'vertical'
    
    # Copy base line to output grid
    for r, c in base_line_structure:
        output_grid.set_cell(r, c, 8)
    
    # Count total dots to move
    total_dots = sum(1 for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == color_to_move)
    
    # Create mirroring lines
    if orientation == 'horizontal':
        base_row = base_line_structure[0][0]
        above_line = [0] * cols
        below_line = [0] * cols
        for c in range(cols):
            if output_grid.get_cell(base_row, c) == 8:
                above_line[c] = below_line[c] = -1  # Mark positions to skip
        
        # Distribute dots
        dots_above = dots_below = total_dots // 2
        for c in range(cols):
            if above_line[c] != -1 and dots_above > 0:
                above_line[c] = color_to_move
                dots_above -= 1
            if below_line[c] != -1 and dots_below > 0:
                below_line[c] = color_to_move
                dots_below -= 1
        
        # Transfer to output grid
        for c in range(cols):
            if above_line[c] == color_to_move:
                output_grid.set_cell(base_row - 1, c, color_to_move)
            if below_line[c] == color_to_move:
                output_grid.set_cell(base_row + 1, c, color_to_move)
    else:  # vertical
        base_col = base_line_structure[0][1]
        right_line = [0] * rows
        for r in range(rows):
            if output_grid.get_cell(r, base_col) == 8:
                right_line[r] = -1  # Mark positions to skip
        
        # Distribute dots
        dots_right = total_dots
        for r in range(rows):
            if right_line[r] != -1 and dots_right > 0:
                right_line[r] = color_to_move
                dots_right -= 1
        
        # Transfer to output grid
        for r in range(rows):
            if right_line[r] == color_to_move:
                output_grid.set_cell(r, base_col + 1, color_to_move)
    
    return output_grid
