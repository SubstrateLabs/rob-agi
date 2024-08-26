from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e78887d1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the most representative 3-row pattern.
    
    The function performs the following steps:
    1. Identifies non-empty 3-row sets in the input grid.
    2. Selects the most representative 3-row set based on pattern consistency.
    3. Extracts the selected 3-row set to form the output grid.
    4. Adjusts patterns if necessary to maintain consistency across the output.
    
    This approach prioritizes preserving the most common and consistent pattern
    from the input grid, adapting it to fit the 3-row output constraint.
    """
    rows, cols = input_grid.get_dimensions()
    non_empty_rows = [i for i in range(rows) if any(input_grid.values[i])]
    row_sets = [non_empty_rows[i:i+3] for i in range(0, len(non_empty_rows), 4) if i+3 <= len(non_empty_rows)]
    
    if not row_sets:
        return ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(3)])
    
    # Select the most representative row set
    best_set = max(row_sets, key=lambda s: sum(input_grid.values[r].count(0) for r in s))
    
    # Extract the selected 3-row set
    output_grid = ColoredGrid(values=[input_grid.values[r][:] for r in best_set])
    
    # Adjust patterns if necessary
    complete_patterns(output_grid)
    
    return output_grid

def complete_patterns(grid: ColoredGrid):
    """Completes patterns in the grid by filling in missing parts of shapes and ensuring consistency."""
    for col in range(grid.get_dimensions()[1]):
        if is_partial_vertical_line(grid, col):
            complete_vertical_line(grid, col)
        if is_partial_horizontal_line(grid, col):
            complete_horizontal_line(grid, col)
    
    # Ensure vertical alignment
    for col in range(grid.get_dimensions()[1]):
        align_vertically(grid, col)

def align_vertically(grid: ColoredGrid, col: int):
    """Aligns colors vertically in a column, moving non-zero values to the top."""
    colors = [grid.values[row][col] for row in range(3) if grid.values[row][col] != 0]
    for row in range(len(colors)):
        grid.values[row][col] = colors[row]
    for row in range(len(colors), 3):
        grid.values[row][col] = 0

def is_partial_vertical_line(grid: ColoredGrid, col: int) -> bool:
    """Checks if there's a partial vertical line in the given column."""
    colors = [grid.values[row][col] for row in range(3)]
    return colors.count(0) == 1 and len(set(colors) - {0}) == 1

def complete_vertical_line(grid: ColoredGrid, col: int):
    """Completes a partial vertical line in the given column."""
    color = max(set(grid.values[row][col] for row in range(3)) - {0})
    for row in range(3):
        if grid.values[row][col] == 0:
            grid.values[row][col] = color

def is_partial_horizontal_line(grid: ColoredGrid, col: int) -> bool:
    """Checks if there's a partial horizontal line starting from the given column."""
    return any(sum(1 for c in range(col, min(col+3, grid.get_dimensions()[1])) if grid.values[row][c] == color) == 2
               for row in range(3)
               for color in set(grid.values[row][c] for c in range(col, min(col+3, grid.get_dimensions()[1]))) - {0})

def complete_horizontal_line(grid: ColoredGrid, col: int):
    """Completes a partial horizontal line starting from the given column."""
    for row in range(3):
        colors = [grid.values[row][c] for c in range(col, min(col+3, grid.get_dimensions()[1]))]
        if len(set(colors) - {0}) == 1 and colors.count(0) == 1:
            color = max(set(colors) - {0})
            for c in range(col, min(col+3, grid.get_dimensions()[1])):
                if grid.values[row][c] == 0:
                    grid.values[row][c] = color
