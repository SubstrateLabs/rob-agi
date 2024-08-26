from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict

def solve_67c52801(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving colored cells downward while preserving their column positions.
    
    The transformation follows these rules:
    1. The bottom row of the input grid remains unchanged.
    2. Colored cells move vertically downward, maintaining their original column positions.
    3. Connected regions of the same color maintain their shape and relative positions.
    4. Empty space (black/0) fills from the top down.
    
    The algorithm works as follows:
    1. Initialize the output grid with zeros and copy the bottom row from the input.
    2. Analyze the input grid to record positions of colored cells.
    3. Sort colored cells by color and original row position.
    4. Place colored cells in the output grid from bottom to top.
    5. Fill remaining space with zeros (black/empty).
    
    Returns:
    ColoredGrid: The transformed grid according to the specified rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the bottom row
    output_grid.values[-1] = input_grid.values[-1].copy()
    
    # Analyze the input grid
    color_positions = defaultdict(list)
    for row in range(rows - 2, -1, -1):  # Exclude bottom row
        for col in range(cols):
            color = input_grid.values[row][col]
            if color != 0:
                color_positions[color].append((row, col))
    
    # Place colored cells
    current_row = rows - 2
    for color in sorted(color_positions.keys()):
        for _, col in sorted(color_positions[color], key=lambda x: x[0], reverse=True):
            while current_row > 0 and output_grid.values[current_row][col] != 0:
                current_row -= 1
            if current_row >= 0:
                output_grid.values[current_row][col] = color
        current_row = rows - 2
    
    return output_grid
