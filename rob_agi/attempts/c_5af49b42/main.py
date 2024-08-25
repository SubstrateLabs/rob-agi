from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5af49b42(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored dots based on a sequence.
    
    1. Extracts the expansion sequence from the bottom row.
    2. For each non-zero cell (except in the bottom row):
       a. Finds the starting index in the expansion sequence.
       b. Expands to the right, filling zero cells with colors from the sequence.
       c. Stops expansion at non-zero cells or the end of the row.
    3. Keeps the bottom row unchanged.
    4. Returns the transformed grid.
    """
    # Extract the expansion sequence
    expansion_sequence = [color for color in input_grid.values[-1] if color != 0]
    
    # Create a copy of the input grid
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    # Process each row (except the bottom row)
    for row in range(rows - 1):
        # Process each cell in the row
        for col in range(cols):
            if new_grid.values[row][col] != 0:
                color = new_grid.values[row][col]
                start_index = expansion_sequence.index(color)
                
                # Perform the expansion
                for i in range(col + 1, cols):
                    if new_grid.values[row][i] != 0:
                        break  # Stop at non-zero cell
                    expansion_index = (start_index + i - col) % len(expansion_sequence)
                    new_grid.values[row][i] = expansion_sequence[expansion_index]

    # Bottom row is preserved as we don't modify it

    return new_grid
