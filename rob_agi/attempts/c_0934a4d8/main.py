from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0934a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by finding a subgrid within the input grid that contains a specific pattern of color transitions.
    
    The function searches for a subgrid that has the following properties:
    1. Contains at least 3 distinct colors
    2. Has a repeating pattern of color transitions along rows or columns
    3. The pattern is consistent across all rows or all columns
    
    The function uses a sliding window approach, starting from the smallest possible size
    and gradually increasing until it finds the correct subgrid. It systematically checks
    all possible positions and sizes to ensure finding the correct extraction.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Start with the smallest window size that could contain a valid pattern
    for window_height in range(3, rows + 1):
        for window_width in range(3, cols + 1):
            for top in range(rows - window_height + 1):
                for left in range(cols - window_width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, window_height, window_width)
                    if is_valid_solution(subgrid):
                        return subgrid
    
    # If no valid solution is found
    raise ValueError("No valid solution found")

def is_valid_solution(grid: ColoredGrid) -> bool:
    """
    Check if the given grid contains a valid pattern of color transitions.
    """
    rows, cols = grid.get_dimensions()
    
    # Check for patterns along rows
    row_pattern = get_color_transition_pattern(grid.values[0])
    if row_pattern and all(get_color_transition_pattern(row) == row_pattern for row in grid.values[1:]):
        return True
    
    # Check for patterns along columns
    col_pattern = get_color_transition_pattern([grid.values[r][0] for r in range(rows)])
    if col_pattern and all(get_color_transition_pattern([grid.values[r][c] for r in range(rows)]) == col_pattern for c in range(1, cols)):
        return True
    
    return False

def get_color_transition_pattern(sequence: List[int]) -> Tuple[int, ...]:
    """
    Extract the pattern of color transitions from a sequence of colors.
    Returns a tuple representing the pattern, or None if no valid pattern is found.
    """
    if len(set(sequence)) < 3:  # Ensure at least 3 distinct colors
        return None
    
    for pattern_length in range(2, len(sequence) // 2 + 1):
        pattern = tuple(sequence[:pattern_length])
        if all(sequence[i:i+pattern_length] == list(pattern) for i in range(0, len(sequence), pattern_length)):
            return pattern
    
    return None
