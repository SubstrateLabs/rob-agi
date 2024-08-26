from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_0934a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by analyzing the input grid for repeating patterns and constructing a smaller output grid
    that captures the essence of the input's pattern and color distribution.
    
    The function performs the following steps:
    1. Analyzes the input grid for repeating horizontal sequences
    2. Identifies the most significant sequence based on frequency and distinctiveness
    3. Determines the output grid size based on the significant sequence
    4. Constructs an output grid that represents the key sequence and color distribution
    5. Ensures the output has at least 3 distinct colors and captures the input's essence
    """
    key_sequence = find_key_sequence(input_grid)
    output_size = determine_output_size(key_sequence)
    output_grid = construct_output_grid(input_grid, key_sequence, output_size)
    
    return ColoredGrid(values=output_grid)

def find_key_sequence(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    sequences = Counter()
    
    for row in grid.values:
        for length in range(4, 8):  # Look for sequences of length 4 to 7
            for i in range(cols - length + 1):
                seq = tuple(row[i:i+length])
                sequences[seq] += 1
    
    if not sequences:
        return list(grid.values[0][:4])  # Fallback to first 4 elements if no sequences found
    
    # Select the most frequent and distinctive sequence
    best_seq = max(sequences, key=lambda seq: (sequences[seq], len(set(seq))))
    return list(best_seq)

def determine_output_size(key_sequence: List[int]) -> Tuple[int, int]:
    width = len(key_sequence)
    height = min(max(3, width), 9)  # Ensure height is between 3 and 9
    return height, width

def construct_output_grid(input_grid: ColoredGrid, key_sequence: List[int], size: Tuple[int, int]) -> List[List[int]]:
    height, width = size
    output = []
    
    for i in range(height):
        row = key_sequence[:]  # Copy the key sequence for each row
        if i % 2 == 1:  # Alternate rows for more variety
            row = row[::-1]  # Reverse the sequence
        output.append(row)
    
    # Ensure color diversity
    distinct_colors = set(color for row in output for color in row)
    if len(distinct_colors) < 3:
        color_freq = Counter(color for row in input_grid.values for color in row)
        for color, _ in color_freq.most_common():
            if color not in distinct_colors:
                output[len(distinct_colors) % height][0] = color
                distinct_colors.add(color)
            if len(distinct_colors) >= 3:
                break
    
    return output
