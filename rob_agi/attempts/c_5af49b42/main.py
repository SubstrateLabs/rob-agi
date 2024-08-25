from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5af49b42(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored dots based on a sequence.
    
    1. Extracts the expansion sequence from the bottom row.
    2. For each non-zero cell (except in the bottom row):
       a. Finds the starting index in the expansion sequence.
       b. Expands the full sequence from that index, wrapping around if needed.
       c. Places the expansion in the grid, wrapping to the next row if needed.
    3. Keeps the bottom row unchanged.
    4. Returns the transformed grid.
    """
    def get_expansion_sequence(grid: ColoredGrid) -> List[int]:
        return [color for color in grid.values[-1] if color != 0]

    expansion_sequence = get_expansion_sequence(input_grid)
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    for row in range(rows - 1):  # Exclude the bottom row
        for col in range(cols):
            if new_grid.values[row][col] != 0:
                color = new_grid.values[row][col]
                start_index = expansion_sequence.index(color)
                expansion = expansion_sequence[start_index:] + expansion_sequence[:start_index]
                
                for i, exp_color in enumerate(expansion):
                    new_col = (col + i) % cols
                    new_grid.values[row][new_col] = exp_color

    # Restore the bottom row
    new_grid.values[-1] = input_grid.values[-1][:]

    return new_grid
