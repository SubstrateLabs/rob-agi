from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict

def solve_8597cfd7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Find the horizontal gray line.
    2. Count occurrences of each color above and below the gray line.
    3. Select the most significant color based on:
       - Presence both above and below the gray line
       - Number of occurrences below the gray line (primary factor)
       - Number of occurrences above the gray line (secondary factor)
       - Rightmost position in the grid (tertiary factor)
    4. Return a 2x2 grid filled with the chosen color, or black if no color qualifies.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the gray line
    gray_line_index = next(r for r in range(rows) if all(input_grid.values[r][c] == 5 for c in range(cols)))
    
    # Initialize counters
    above_counts = defaultdict(int)
    below_counts = defaultdict(int)
    rightmost_positions = {}

    # Scan the grid
    for col in range(cols):
        for row in range(rows):
            color = input_grid.values[row][col]
            if color != 0 and color != 5:
                if row < gray_line_index:
                    above_counts[color] += 1
                elif row > gray_line_index:
                    below_counts[color] += 1
                rightmost_positions[color] = col

    # Evaluate color significance
    max_significance = (-1, -1, -1)
    chosen_color = 0

    for color in set(above_counts.keys()) & set(below_counts.keys()):
        significance = (below_counts[color], above_counts[color], rightmost_positions[color])
        if significance > max_significance:
            max_significance = significance
            chosen_color = color

    # Create the output grid
    return ColoredGrid(values=[[chosen_color] * 2 for _ in range(2)])
