from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from itertools import cycle

def solve_b457fec5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by filling gray areas with a diagonal pattern of colors.
    
    The solution follows these steps:
    1. Extract the color sequence from the input grid.
    2. Determine the fill direction based on the color cluster's position.
    3. Create a color cycle from the extracted sequence.
    4. Find the top-left corner of the first gray region.
    5. Process the input grid, replacing gray cells with the appropriate colors from the pattern.
    6. Return the transformed grid.
    
    The pattern starts from the top-left or top-right of the first gray region (depending on the fill direction),
    uses colors in the order they appear in the input, and follows a diagonal pattern
    while maintaining color sequence across all gray regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Step 1: Extract color sequence
    color_sequence = []
    for row in input_grid.values:
        for cell in row:
            if cell not in [0, 5] and cell not in color_sequence:
                color_sequence.append(cell)
    
    if not color_sequence:
        return output_grid  # No colors to fill with
    
    # Step 2: Determine fill direction
    color_positions = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] in color_sequence]
    avg_col = sum(c for _, c in color_positions) / len(color_positions)
    fill_direction = 1 if avg_col < cols / 2 else -1
    
    # Step 3: Create color cycle
    color_cycle = cycle(color_sequence)
    
    # Step 4: Find the top-left corner of the first gray region
    start_row, start_col = next((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5)
    
    # Step 5: Process the input grid
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 5:  # gray
                diagonal_index = (r - start_row) + ((c - start_col) if fill_direction == 1 else (start_col - c))
                color = next(color_cycle)
                for _ in range(diagonal_index % len(color_sequence)):
                    color = next(color_cycle)
                output_grid.values[r][c] = color
    
    return output_grid
