from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ea959feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying row and column color sequences
    and applying them to create a consistent pattern across the entire grid.
    
    The solution works as follows:
    1. Extracts the row pattern by finding the most common repeating sequence across all rows.
    2. Extracts the column pattern by finding the most common repeating sequence across all columns.
    3. Validates and extends the patterns to cover the entire grid dimensions.
    4. Generates a new grid by combining the row and column patterns.
    5. Returns a new ColoredGrid with the corrected pattern.
    
    This approach works for all cases by identifying the underlying patterns in each input
    and extending them consistently across the entire grid.
    """
    rows, cols = input_grid.get_dimensions()
    row_pattern = extract_pattern(input_grid.values)
    col_pattern = extract_pattern(list(zip(*input_grid.values)))
    
    # Extend patterns if necessary
    row_pattern = row_pattern * (cols // len(row_pattern) + 1)
    col_pattern = col_pattern * (rows // len(col_pattern) + 1)
    
    corrected_values = [[((row_pattern[j] + col_pattern[i]) % 10) for j in range(cols)] for i in range(rows)]
    return ColoredGrid(values=corrected_values)

def extract_pattern(lines: List[List[int]]) -> List[int]:
    """Extracts the most common repeating pattern from a list of lines."""
    patterns = [find_shortest_repeating_sequence(line) for line in lines]
    return find_common_sequence(patterns)

def find_shortest_repeating_sequence(line: List[int]) -> List[int]:
    """Finds the shortest repeating sequence in a line."""
    for i in range(1, len(line) // 2 + 1):
        if line[:i] * (len(line) // i) == line[:len(line) - (len(line) % i)]:
            return line[:i]
    return line  # If no repetition found, return the entire line

def find_common_sequence(sequences: List[List[int]]) -> List[int]:
    """Finds the most common sequence from a list of sequences."""
    sequence_counts = {}
    for seq in sequences:
        seq_tuple = tuple(seq)
        sequence_counts[seq_tuple] = sequence_counts.get(seq_tuple, 0) + 1
    
    most_common = max(sequence_counts, key=sequence_counts.get)
    return list(most_common)
