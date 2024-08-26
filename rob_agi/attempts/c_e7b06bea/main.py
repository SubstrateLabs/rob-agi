from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7b06bea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by preserving the leftmost non-black column and condensing the remaining non-black columns.
    
    1. Identify non-black columns
    2. Preserve the leftmost non-black column exactly in its original position
    3. Create a condensed pattern from the remaining non-black columns, maintaining color order and segment lengths
    4. Create a condensed column by repeating the pattern to fill the grid height
    5. Position the condensed column immediately to the right of the leftmost non-black column
    6. Construct and return the transformed grid
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify non-black columns
    non_black_cols = [col for col in range(cols) if any(input_grid.get_cell(row, col) != 0 for row in range(rows))]
    
    if not non_black_cols:
        return input_grid  # If all columns are black, return the input grid
    
    # Step 2: Preserve the leftmost non-black column
    leftmost_col = non_black_cols[0]
    
    # Step 3: Create the condensed pattern
    condensed_pattern = []
    if len(non_black_cols) > 1:
        for col in non_black_cols[1:]:
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
    
    # Step 4: Create the condensed column
    condensed_col = []
    if condensed_pattern:
        while len(condensed_col) < rows:
            for color, length in condensed_pattern:
                condensed_col.extend([color] * length)
                if len(condensed_col) >= rows:
                    break
        condensed_col = condensed_col[:rows]
    
    # Step 5: Position for condensed column is always right next to the leftmost non-black column
    condensed_pos = leftmost_col + 1
    
    # Step 6: Construct the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Place leftmost non-black column
    for row in range(rows):
        output_values[row][leftmost_col] = input_grid.get_cell(row, leftmost_col)
    
    # Place condensed column
    if condensed_col:
        for row in range(rows):
            output_values[row][condensed_pos] = condensed_col[row]
    
    return ColoredGrid(values=output_values)
