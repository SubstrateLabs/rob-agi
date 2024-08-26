from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e78887d1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting and combining patterns from non-empty row sets.
    
    The function performs the following steps:
    1. Identifies non-empty rows and groups them into sets.
    2. Analyzes each column across all sets to determine the most representative pattern.
    3. Creates a 3-row output grid, filling it column by column based on the analysis.
    4. Completes and aligns patterns to ensure consistency across colors.
    
    This approach allows for both simple extractions and complex pattern combinations,
    adapting to the specific needs of each input grid.
    """
    rows, cols = input_grid.get_dimensions()
    non_empty_rows = [i for i in range(rows) if any(input_grid.values[i])]
    row_sets = [non_empty_rows[i:i+3] for i in range(0, len(non_empty_rows), 4)]
    
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(3)])
    
    for col in range(cols):
        column_colors = [input_grid.values[row][col] for row_set in row_sets for row in row_set]
        unique_colors = sorted(set(column_colors) - {0}, key=lambda x: column_colors.count(x), reverse=True)
        
        for i, color in enumerate(unique_colors[:3]):
            output_grid.values[i][col] = color
    
    complete_patterns(output_grid)
    return output_grid

def complete_patterns(grid: ColoredGrid):
    """Completes patterns in the grid by filling in missing parts of shapes."""
    for col in range(grid.get_dimensions()[1]):
        if is_partial_vertical_line(grid, col):
            complete_vertical_line(grid, col)
        if is_partial_horizontal_line(grid, col):
            complete_horizontal_line(grid, col)

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
