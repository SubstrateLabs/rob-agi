from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_ca8f78db(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Fills in all cells in the output grid based on the identified pattern.
    
    The function analyzes the input grid to identify Type A (single color) rows
    and Type B (color sequence) rows. It then creates a new grid, filling all
    cells according to the alternating row pattern, regardless of the input
    grid's black cells. This ensures a consistent pattern across the entire grid.
    
    Args:
    input_grid (ColoredGrid): The input grid with the pattern to be analyzed.
    
    Returns:
    ColoredGrid: A new grid with all cells filled according to the identified pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    type_a_color = get_type_a_color(input_grid)
    type_b_sequence = get_type_b_sequence(input_grid)
    
    for r in range(rows):
        if is_type_a_row(r):
            for c in range(cols):
                output_grid.set_cell(r, c, type_a_color)
        else:
            for c in range(cols):
                sequence_position = c % len(type_b_sequence)
                output_grid.set_cell(r, c, type_b_sequence[sequence_position])
    
    return output_grid

def get_type_a_color(grid: ColoredGrid) -> int:
    for cell in grid.values[0]:
        if cell != 0:
            return cell
    return 1  # Default to blue if no non-zero color found

def get_type_b_sequence(grid: ColoredGrid) -> List[int]:
    for row in grid.values[1::2]:  # Check even-indexed rows (0-based index)
        sequence = []
        for cell in row:
            if cell != 0:
                if cell in sequence:
                    return sequence
                sequence.append(cell)
        if sequence:
            return sequence
    return [1, 4]  # Default sequence if no valid sequence found

def is_type_a_row(row_index: int) -> bool:
    return row_index % 2 == 0
