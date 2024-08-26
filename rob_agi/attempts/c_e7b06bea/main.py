from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_e7b06bea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by condensing non-black columns while preserving the leftmost non-black column.
    
    1. Identify non-black columns
    2. Preserve the leftmost non-black column
    3. Condense remaining non-black columns into a single column
    4. Position the condensed column based on grid dimensions and leftmost non-black column position
    5. Construct and return the transformed grid
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify non-black columns
    non_black_cols = [(col, [input_grid.get_cell(row, col) for row in range(rows)]) 
                      for col in range(cols) 
                      if any(input_grid.get_cell(row, col) != 0 for row in range(rows))]
    
    if not non_black_cols:
        return input_grid  # If all columns are black, return the input grid
    
    # Step 2: Preserve the leftmost non-black column
    leftmost_col, leftmost_values = non_black_cols[0]
    
    # Step 3: Condense remaining non-black columns
    if len(non_black_cols) > 1:
        remaining_cols = [col for col, _ in non_black_cols[1:]]
        condensed_col = condense_columns(input_grid, remaining_cols, rows)
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
        output_values[row][leftmost_col] = leftmost_values[row]
    
    # Place condensed column
    if condensed_col:
        for row in range(rows):
            output_values[row][condensed_pos] = condensed_col[row]
    
    return ColoredGrid(values=output_values)

def condense_columns(grid: ColoredGrid, columns: List[int], height: int) -> List[int]:
    """Condense multiple columns into a single column based on color proportions."""
    color_counts = {}
    total_cells = 0
    
    for col in columns:
        for row in range(height):
            color = grid.get_cell(row, col)
            if color != 0:
                color_counts[color] = color_counts.get(color, 0) + 1
                total_cells += 1
    
    if total_cells == 0:
        return [0] * height
    
    # Calculate proportions and allocate cells
    condensed = []
    remaining = height
    for color in sorted(color_counts.keys()):
        proportion = color_counts[color] / total_cells
        allocated = round(proportion * height)
        condensed.extend([color] * allocated)
        remaining -= allocated
    
    # Adjust for rounding errors
    if remaining > 0:
        most_frequent = max(color_counts, key=color_counts.get)
        condensed.extend([most_frequent] * remaining)
    elif remaining < 0:
        while remaining < 0:
            most_frequent = max(color_counts, key=color_counts.get)
            condensed.remove(most_frequent)
            remaining += 1
    
    return condensed
