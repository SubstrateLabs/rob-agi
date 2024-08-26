from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_e7b06bea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by preserving the leftmost non-black column and condensing the remaining non-black columns.
    
    1. Identify non-black columns
    2. Preserve the leftmost non-black column exactly
    3. Create a condensed column from the remaining non-black columns, maintaining color order and uniqueness
    4. Position the condensed column based on the leftmost non-black column's position
    5. Construct and return the transformed grid
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify non-black columns
    non_black_cols = [col for col in range(cols) if any(input_grid.get_cell(row, col) != 0 for row in range(rows))]
    
    if not non_black_cols:
        return input_grid  # If all columns are black, return the input grid
    
    # Step 2: Preserve the leftmost non-black column
    leftmost_col = non_black_cols[0]
    
    # Step 3: Create the condensed column
    if len(non_black_cols) > 1:
        unique_colors = []
        for col in non_black_cols[1:]:
            for row in range(rows):
                color = input_grid.get_cell(row, col)
                if color != 0 and color not in unique_colors:
                    unique_colors.append(color)
        
        if not unique_colors:
            condensed_col = []
        else:
            condensed_col = (unique_colors * (rows // len(unique_colors) + 1))[:rows]
    else:
        condensed_col = []
    
    # Step 4: Determine position for condensed column
    if leftmost_col == 0:
        condensed_pos = cols // 2
    else:
        remaining_width = cols - leftmost_col - 1
        condensed_pos = leftmost_col + (remaining_width // 2)
    
    # Step 5: Construct the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Place leftmost non-black column
    for row in range(rows):
        output_values[row][leftmost_col] = input_grid.get_cell(row, leftmost_col)
    
    # Place condensed column
    if condensed_col:
        for row in range(rows):
            output_values[row][condensed_pos] = condensed_col[row]
    
    return ColoredGrid(values=output_values)
