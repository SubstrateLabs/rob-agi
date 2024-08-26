from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_c663677b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the full repeating pattern
    and applying it to the entire grid, including black (0) areas.

    The solution follows these steps:
    1. Analyze the input grid to find non-zero patterns in rows and columns.
    2. Identify the full repeating pattern by finding the longest repeating
       sequence in both horizontal and vertical directions.
    3. Reconstruct the full pattern based on the identified sequences.
    4. Validate the pattern against non-zero areas in the input grid.
    5. Create the output grid by applying the full pattern to all cells.

    Args:
        input_grid (ColoredGrid): The input grid with partial pattern and black areas.

    Returns:
        ColoredGrid: The solved grid with the full pattern applied to all cells.
    """
    # Step 1: Analyze the grid
    horizontal_patterns = analyze_patterns(input_grid.values)
    vertical_patterns = analyze_patterns(list(zip(*input_grid.values)))

    # Step 2: Identify full pattern
    h_pattern = find_longest_repeating_pattern(horizontal_patterns)
    v_pattern = find_longest_repeating_pattern(vertical_patterns)
    full_pattern_size = math.lcm(len(h_pattern), len(v_pattern))
    full_pattern = reconstruct_full_pattern(h_pattern, v_pattern, full_pattern_size)

    # Step 3: Validate pattern
    while not is_pattern_valid(input_grid, full_pattern):
        full_pattern_size *= 2
        full_pattern = reconstruct_full_pattern(h_pattern, v_pattern, full_pattern_size)

    # Step 4: Create output grid
    output_values = [
        [full_pattern[r % len(full_pattern)][c % len(full_pattern[0])] 
         for c in range(len(input_grid.values[0]))]
        for r in range(len(input_grid.values))
    ]

    return ColoredGrid(values=output_values)

def analyze_patterns(sequences: List[List[int]]) -> List[Tuple[List[int], int]]:
    patterns = []
    for seq in sequences:
        non_zero_seq = [color for color in seq if color != 0]
        if non_zero_seq:
            patterns.append((non_zero_seq, seq.index(non_zero_seq[0])))
    return patterns

def find_longest_repeating_pattern(patterns: List[Tuple[List[int], int]]) -> List[int]:
    longest_pattern = []
    for pattern, _ in patterns:
        if len(pattern) > len(longest_pattern):
            longest_pattern = pattern
    return longest_pattern

def reconstruct_full_pattern(h_pattern: List[int], v_pattern: List[int], size: int) -> List[List[int]]:
    full_pattern = [[0 for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(size):
            full_pattern[r][c] = h_pattern[c % len(h_pattern)]
    return full_pattern

def is_pattern_valid(grid: ColoredGrid, pattern: List[List[int]]) -> bool:
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if grid.values[r][c] != 0 and grid.values[r][c] != pattern[r % len(pattern)][c % len(pattern[0])]:
                return False
    return True
