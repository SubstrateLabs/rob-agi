from rob_agi.colored_grid import ColoredGrid
from typing import List

def get_missing_numbers(sequence: List[int]) -> List[int]:
    return [num for num in range(1, 5) if num not in sequence]

def is_valid_sequence(seq: List[int]) -> bool:
    return sorted(seq) == [1, 2, 3, 4]

def solve_4cd1b7b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 4x4 grid by filling in black (0) squares with numbers 1-4,
    ensuring each row and column contains exactly one of each number 1-4.
    
    The function preserves non-zero numbers in their original positions,
    replaces black (0) squares with appropriate numbers, and adjusts the grid
    to create a valid Latin square while maintaining the initially given numbers.
    
    Args:
    input_grid (ColoredGrid): The input 4x4 grid with some filled and some black squares.
    
    Returns:
    ColoredGrid: The completed grid with all squares filled to form a valid Latin square.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    # Fill rows
    for r in range(rows):
        missing = get_missing_numbers(grid.values[r])
        for c in range(cols):
            if grid.values[r][c] == 0:
                grid.values[r][c] = missing.pop(0)

    # Adjust columns
    for c in range(cols):
        column = [grid.values[r][c] for r in range(rows)]
        if not is_valid_sequence(column):
            missing = get_missing_numbers(column)
            zero_positions = [r for r in range(rows) if input_grid.values[r][c] == 0]
            for r in zero_positions:
                if grid.values[r][c] in missing:
                    missing.remove(grid.values[r][c])
                else:
                    grid.values[r][c] = missing.pop(0)

    # Final adjustment
    for c in range(cols):
        column = [grid.values[r][c] for r in range(rows)]
        if not is_valid_sequence(column):
            zero_positions = [r for r in range(rows) if input_grid.values[r][c] == 0]
            for i in range(len(zero_positions)):
                for j in range(i + 1, len(zero_positions)):
                    r1, r2 = zero_positions[i], zero_positions[j]
                    grid.values[r1][c], grid.values[r2][c] = grid.values[r2][c], grid.values[r1][c]
                    if is_valid_sequence([grid.values[r][c] for r in range(rows)]) and \
                       all(is_valid_sequence(grid.values[r]) for r in range(rows)):
                        break
                else:
                    continue
                break

    return grid
