from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7b06bea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving non-black columns to the center:
    1. Identify the middle column (leftmost full-height non-black column or leftmost non-black column)
    2. Create a condensed pattern from other non-black columns
    3. Generate a condensed column by repeating the pattern
    4. Place the middle column and condensed column in the center of the output grid
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify non-black columns and the middle column
    non_black_cols = []
    middle_col = None
    for col in range(cols):
        if any(input_grid.get_cell(row, col) != 0 for row in range(rows)):
            non_black_cols.append(col)
            if middle_col is None and all(input_grid.get_cell(row, col) != 0 for row in range(rows)):
                middle_col = col
    
    if not non_black_cols:
        return input_grid  # If all columns are black, return the input grid
    
    if middle_col is None:
        middle_col = non_black_cols[0]  # Use leftmost non-black column if no full-height column
    
    # Step 2: Create the condensed pattern
    condensed_pattern = []
    for col in non_black_cols:
        if col != middle_col:
            current_color = None
            current_length = 0
            for row in range(rows):
                color = input_grid.get_cell(row, col)
                if color != 0:
                    if color != current_color:
                        if current_color is not None:
                            condensed_pattern.append((current_color, current_length))
                        current_color = color
                        current_length = 1
                    else:
                        current_length += 1
            if current_color is not None:
                condensed_pattern.append((current_color, current_length))
    
    # Step 3: Generate the condensed column
    condensed_col = []
    if condensed_pattern:
        pattern_length = sum(length for _, length in condensed_pattern)
        repetitions = (rows + pattern_length - 1) // pattern_length
        for _ in range(repetitions):
            for color, length in condensed_pattern:
                condensed_col.extend([color] * length)
        condensed_col = condensed_col[:rows]
    
    # Step 4: Construct the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    center = cols // 2
    
    # Place middle column
    for row in range(rows):
        output_values[row][center] = input_grid.get_cell(row, middle_col)
    
    # Place condensed column
    if condensed_col:
        condensed_pos = center + 1 if center + 1 < cols else center - 1
        for row in range(rows):
            output_values[row][condensed_pos] = condensed_col[row]
    
    return ColoredGrid(values=output_values)
