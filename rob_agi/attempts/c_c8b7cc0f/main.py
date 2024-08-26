from rob_agi.colored_grid import ColoredGrid
from typing import Dict, Set, Tuple

def solve_c8b7cc0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x3 output grid based on the following rules:
    1. Ignores black (0) and blue (1) colors.
    2. Finds the non-black, non-blue color with the most distinct positions in the input grid.
    3. Creates a 3x3 output grid with the top-left cells filled with the most frequent color.
    4. The number of filled cells is determined as follows:
       - If the count of distinct positions is 1, 2, or 3, use that number.
       - If the count is 4 or more, use 4.
    5. If no valid colors are found, returns an all-black 3x3 grid.
    """
    # Initialize color position tracker
    color_positions: Dict[int, Set[Tuple[int, int]]] = {}

    # Analyze the input grid
    for row in range(len(input_grid.values)):
        for col in range(len(input_grid.values[0])):
            color = input_grid.values[row][col]
            if color not in [0, 1]:  # Ignore black and blue
                if color not in color_positions:
                    color_positions[color] = set()
                color_positions[color].add((row, col))

    # Find the target color
    if not color_positions:
        return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    target_color = max(color_positions, key=lambda x: len(color_positions[x]))

    # Determine the number of cells to fill
    distinct_positions = len(color_positions[target_color])
    cells_to_fill = min(distinct_positions, 4) if distinct_positions >= 4 else distinct_positions

    # Create the output grid
    output_values = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    fill_order = [(0, 0), (0, 1), (0, 2), (1, 0)]
    for i in range(cells_to_fill):
        row, col = fill_order[i]
        output_values[row][col] = target_color

    # Return the output grid
    return ColoredGrid(values=output_values)
