from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7b06bea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving non-black columns to the center:
    1. Identify the special column (leftmost full-height non-black column or leftmost non-black column)
    2. Create a condensed column from other non-black columns
    3. Place the special column in the center of the output grid
    4. Place the condensed column to the right of the special column (or left if not possible)
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify non-black columns and the special column
    non_black_cols = []
    special_col = None
    for col in range(cols):
        if any(input_grid.get_cell(row, col) != 0 for row in range(rows)):
            non_black_cols.append(col)
            if special_col is None and all(input_grid.get_cell(row, col) != 0 for row in range(rows)):
                special_col = col
    
    if not non_black_cols:
        return input_grid  # If all columns are black, return the input grid
    
    if special_col is None:
        special_col = non_black_cols[0]  # Use leftmost non-black column if no full-height column
    
    # Step 2: Create the condensed column
    condensed_col = []
    for col in non_black_cols:
        if col != special_col:
            for row in range(rows):
                color = input_grid.get_cell(row, col)
                if color != 0:
                    condensed_col.append(color)
    
    # Ensure condensed column has the same height as the input grid
    if condensed_col:
        condensed_col = (condensed_col * ((rows + len(condensed_col) - 1) // len(condensed_col)))[:rows]
    
    # Step 3 & 4: Construct the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    center = cols // 2
    
    # Place special column
    for row in range(rows):
        output_values[row][center] = input_grid.get_cell(row, special_col)
    
    # Place condensed column
    if condensed_col:
        condensed_pos = center + 1 if center + 1 < cols else center - 1
        for row in range(rows):
            output_values[row][condensed_pos] = condensed_col[row]
    
    return ColoredGrid(values=output_values)
