from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def identify_sequences(grid: ColoredGrid) -> List[List[int]]:
    """Identify unique color sequences in the grid."""
    sequences = []
    for row in grid.values:
        if 0 not in row:
            sequence = [color for color in row if color != 0]
            if sequence and sequence not in sequences:
                sequences.append(sequence)
    return sequences

def is_original_sequence(row: List[int], sequences: List[List[int]]) -> bool:
    """Check if a row is an original sequence."""
    return 0 not in row and row in sequences

def solve_5af49b42(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored dots based on multiple sequences.
    
    1. Identifies all unique color sequences in the grid.
    2. For each non-zero cell not part of an original sequence:
       a. Selects the next expansion sequence.
       b. Expands to the right, filling zero cells with colors from the sequence.
       c. Stops expansion at non-zero cells or the end of the row.
    3. Alternates between sequences for each expansion.
    4. Keeps original sequences unchanged.
    5. Returns the transformed grid.
    """
    sequences = identify_sequences(input_grid)
    if not sequences:
        return input_grid  # No sequences to expand

    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    sequence_counter = 0
    for row in range(rows):
        if is_original_sequence(new_grid.values[row], sequences):
            continue  # Skip rows that are original sequences
        for col in range(cols):
            if new_grid.values[row][col] != 0:
                expansion_sequence = sequences[sequence_counter % len(sequences)]
                sequence_counter += 1
                
                # Perform the expansion
                index = 0
                for i in range(col + 1, cols):
                    if new_grid.values[row][i] != 0:
                        break  # Stop at non-zero cell
                    new_grid.values[row][i] = expansion_sequence[index % len(expansion_sequence)]
                    index += 1

    return new_grid
