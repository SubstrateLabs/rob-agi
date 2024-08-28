from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7b06bea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving non-black columns to the center:
    1. Identify the special column (leftmost column with maximum height)
    2. Create a condensed column from other non-black columns
    3. Place the special column to the left of the center in the output grid
    4. Place the condensed column to the right of the special column
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify non-black columns and the special column
    non_black_cols = []
    for col in range(cols):
        column_height = sum(1 for row in range(rows) if input_grid.get_cell(row, col) != 0)
        if column_height > 0:
            non_black_cols.append((col, column_height, [input_grid.get_cell(row, col) for row in range(rows)]))
    
    if not non_black_cols:
        return input_grid  # If all columns are black, return the input grid
    
    special_col, _, special_col_data = max(non_black_cols, key=lambda x: (x[1], -x[0]))
    
    # Step 2: Create the condensed column
    condensed_col = []
    for col, _, col_data in non_black_cols:
        if col != special_col:
            condensed_col.extend([color for color in col_data if color != 0])
    
    # Ensure condensed column has the same height as the input grid
    if condensed_col:
        condensed_col = (condensed_col * ((rows + len(condensed_col) - 1) // len(condensed_col)))[:rows]
    
    # Step 3 & 4: Construct the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    center = cols // 2
    
    # Determine new positions
    special_col_new_index = center - 1
    condensed_col_new_index = center
    
    # Place special column
    for row in range(rows):
        output_values[row][special_col_new_index] = special_col_data[row]
    
    # Place condensed column
    if condensed_col:
        for row in range(rows):
            output_values[row][condensed_col_new_index] = condensed_col[row]
    
    return ColoredGrid(values=output_values)
