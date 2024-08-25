from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_5af49b42(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored dots based on multiple sequences.
    
    1. Extracts expansion sequences from the bottom row.
    2. For each non-zero cell (except in the bottom row):
       a. Selects the appropriate expansion sequence.
       b. Expands to the right, filling zero cells with colors from the sequence.
       c. Stops expansion at non-zero cells or the end of the row.
    3. Alternates between sequences for each expansion.
    4. Keeps the bottom row unchanged.
    5. Returns the transformed grid.
    """
    # Extract the expansion sequences
    bottom_row = input_grid.values[-1]
    expansion_sequences = []
    current_sequence = []
    for color in bottom_row:
        if color != 0:
            current_sequence.append(color)
        elif current_sequence:
            expansion_sequences.append(current_sequence)
            current_sequence = []
    if current_sequence:
        expansion_sequences.append(current_sequence)
    
    # Create a copy of the input grid
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    sequence_counter = 0
    # Process each row (except the bottom row)
    for row in range(rows - 1):
        # Process each cell in the row
        for col in range(cols):
            if new_grid.values[row][col] != 0:
                expansion_sequence = expansion_sequences[sequence_counter % len(expansion_sequences)]
                
                # Perform the expansion
                index = 0
                for i in range(col + 1, cols):
                    if new_grid.values[row][i] != 0:
                        break  # Stop at non-zero cell
                    new_grid.values[row][i] = expansion_sequence[index % len(expansion_sequence)]
                    index += 1
                
                sequence_counter += 1

    # Bottom row is preserved as we don't modify it

    return new_grid
