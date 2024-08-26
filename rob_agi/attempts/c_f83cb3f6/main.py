from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_f83cb3f6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving dots of a specific color to form line(s) adjacent to a base line.
    
    1. Identifies the longest continuous line of color 8 (sky blue) as the base line.
    2. Determines the most frequent non-zero, non-8 color as the color to be moved.
    3. Creates new line(s) adjacent to the base line:
       - For horizontal base line: one line above and one below.
       - For vertical base line: one line to the right.
    4. Moves dots of the identified color to the nearest new line, maintaining their position along the base line's axis.
    5. Clears the rest of the grid, keeping only the base line and new line(s) with moved dots.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    # Step 1: Identify the base line
    base_line = find_longest_line(input_grid, 8)
    
    # Step 2: Identify the color to be moved
    color_to_move = find_most_frequent_color(input_grid, exclude=[0, 8])
    
    # Step 3-5: Create new grid with moved dots
    output_grid = create_output_grid(input_grid, base_line, color_to_move)
    
    return output_grid

def find_longest_line(grid: ColoredGrid, color: int) -> Tuple[str, int, int, int]:
    rows, cols = grid.get_dimensions()
    max_length = 0
    orientation = ''
    start_pos = 0
    length = 0
    
    # Check horizontal lines
    for r in range(rows):
        current_length = 0
        for c in range(cols):
            if grid.get_cell(r, c) == color:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    orientation = 'horizontal'
                    start_pos = r
                    length = max_length
                current_length = 0
        if current_length > max_length:
            max_length = current_length
            orientation = 'horizontal'
            start_pos = r
            length = max_length
    
    # Check vertical lines
    for c in range(cols):
        current_length = 0
        for r in range(rows):
            if grid.get_cell(r, c) == color:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    orientation = 'vertical'
                    start_pos = c
                    length = max_length
                current_length = 0
        if current_length > max_length:
            max_length = current_length
            orientation = 'vertical'
            start_pos = c
            length = max_length
    
    return orientation, start_pos, length, max_length

def find_most_frequent_color(grid: ColoredGrid, exclude: List[int]) -> int:
    color_counter = Counter()
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color not in exclude:
                color_counter[color] += 1
    return color_counter.most_common(1)[0][0]

def create_output_grid(input_grid: ColoredGrid, base_line: Tuple[str, int, int, int], color_to_move: int) -> ColoredGrid:
    orientation, start_pos, length, _ = base_line
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy base line to output grid
    if orientation == 'horizontal':
        for c in range(cols):
            output_grid.set_cell(start_pos, c, input_grid.get_cell(start_pos, c))
    else:  # vertical
        for r in range(rows):
            output_grid.set_cell(r, start_pos, input_grid.get_cell(r, start_pos))
    
    # Move dots to new lines
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == color_to_move:
                if orientation == 'horizontal':
                    new_r = start_pos - 1 if r < start_pos else start_pos + 1
                    output_grid.set_cell(new_r, c, color_to_move)
                else:  # vertical
                    new_c = start_pos + 1
                    output_grid.set_cell(r, new_c, color_to_move)
    
    return output_grid
