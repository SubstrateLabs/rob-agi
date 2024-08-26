from rob_agi.colored_grid import ColoredGrid
from typing import List

def get_missing_numbers(sequence: List[int]) -> List[int]:
    return [num for num in range(1, 5) if num not in sequence]

def solve_4cd1b7b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 4x4 grid by filling in black (0) squares with numbers 1-4,
    ensuring each row and column contains exactly one of each number 1-4.
    
    The function preserves non-zero numbers in their original positions and
    replaces all black (0) squares with appropriate numbers to complete the pattern.
    
    Args:
    input_grid (ColoredGrid): The input 4x4 grid with some filled and some black squares.
    
    Returns:
    ColoredGrid: The completed grid with all black squares filled.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    # Fill rows
    for r in range(rows):
        missing = get_missing_numbers(grid.values[r])
        for c in range(cols):
            if grid.values[r][c] == 0:
                grid.values[r][c] = missing.pop(0)

    # Fill columns
    for c in range(cols):
        column = [grid.values[r][c] for r in range(rows)]
        missing = get_missing_numbers(column)
        for r in range(rows):
            if grid.values[r][c] == 0:
                grid.values[r][c] = missing.pop(0)

    return grid
