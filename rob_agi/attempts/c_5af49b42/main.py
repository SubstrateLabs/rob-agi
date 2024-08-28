from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def extract_sequences(grid: ColoredGrid) -> List[List[int]]:
    """Extract unique color sequences from the bottom row of the grid."""
    bottom_row = grid.values[-1]
    sequences = []
    current_sequence = []
    for color in bottom_row:
        if color != 0:
            current_sequence.append(color)
        elif current_sequence:
            if current_sequence not in sequences:
                sequences.append(current_sequence)
            current_sequence = []
    if current_sequence and current_sequence not in sequences:
        sequences.append(current_sequence)
    return sequences

def is_original_sequence(row: List[int], sequences: List[List[int]]) -> bool:
    """Check if a row matches any of the original sequences."""
    row_sequence = [color for color in row if color != 0]
    return row_sequence in sequences

def expand_sequence(row: List[int], start: int, sequence: List[int]) -> None:
    """Expand a sequence to the right, filling zeros."""
    seq_index = 0
    for i in range(start + 1, len(row)):
        if row[i] != 0:
            break
        row[i] = sequence[seq_index % len(sequence)]
        seq_index += 1

def solve_5af49b42(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored dots based on sequences from the bottom row.
    
    1. Extracts unique color sequences from the bottom row of the grid.
    2. For each row (except the bottom row):
       a. If the row matches an original sequence, it's left unchanged.
       b. Otherwise, for each non-zero cell:
          - Expands to the right using the next available sequence.
          - Stops expansion at non-zero cells or the end of the row.
    3. Alternates between sequences for each expansion within a row.
    4. Keeps the bottom row (original sequences) unchanged.
    5. Returns the transformed grid.
    """
    sequences = extract_sequences(input_grid)
    if not sequences:
        return input_grid  # No sequences to expand

    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    for row in range(rows - 1):  # Exclude the bottom row
        if is_original_sequence(new_grid.values[row], sequences):
            continue  # Skip rows that are original sequences
        
        sequence_index = 0
        for col in range(cols):
            if new_grid.values[row][col] != 0:
                expansion_sequence = sequences[sequence_index % len(sequences)]
                expand_sequence(new_grid.values[row], col, expansion_sequence)
                sequence_index += 1

    return new_grid
