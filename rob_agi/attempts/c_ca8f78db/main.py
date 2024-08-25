from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_ca8f78db(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Fills in all black (0) cells in the input grid based on the surrounding pattern.
    
    The function analyzes the grid structure to identify Type A (single color) and
    Type B (color sequence) rows. It then fills black cells by extending the pattern,
    maintaining the alternating row types and color sequences. The solution handles
    edge cases and large black regions by extrapolating the surrounding patterns.
    
    Args:
    input_grid (ColoredGrid): The input grid with black cells to be filled.
    
    Returns:
    ColoredGrid: A new grid with all black cells filled according to the pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Analyze grid structure
    type_a_color, type_b_sequence = analyze_grid_structure(output_grid)
    
    # Fill black cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 0:
                if is_type_a_row(r):
                    output_grid.set_cell(r, c, type_a_color)
                else:
                    sequence_position = c % len(type_b_sequence)
                    output_grid.set_cell(r, c, type_b_sequence[sequence_position])
    
    return output_grid

def analyze_grid_structure(grid: ColoredGrid) -> Tuple[int, List[int]]:
    rows, cols = grid.get_dimensions()
    type_a_color = None
    type_b_sequence = []
    
    for r in range(rows):
        row_colors = [grid.get_cell(r, c) for c in range(cols) if grid.get_cell(r, c) != 0]
        if len(set(row_colors)) == 1:
            if type_a_color is None:
                type_a_color = row_colors[0]
        else:
            if not type_b_sequence:
                type_b_sequence = detect_sequence(row_colors)
    
    return type_a_color, type_b_sequence

def is_type_a_row(row_index: int) -> bool:
    return row_index % 2 == 0

def detect_sequence(colors: List[int]) -> List[int]:
    for length in range(1, len(colors) // 2 + 1):
        if colors[:length] * (len(colors) // length) == colors[:-(len(colors) % length) or None]:
            return colors[:length]
    return colors  # If no repeating sequence found, return the entire list
