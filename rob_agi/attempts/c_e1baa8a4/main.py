from rob_agi.colored_grid import ColoredGrid
import itertools

def get_row_pattern(grid, row):
    return [color for color, group in itertools.groupby(grid.values[row])]

def get_column_pattern(grid, col):
    return [color for color, group in itertools.groupby(row[col] for row in grid.values)]

def solve_e1baa8a4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified representation by identifying unique
    color transition patterns in both horizontal and vertical directions.
    
    The function works as follows:
    1. Identifies unique horizontal color transition patterns.
    2. Identifies unique vertical color transition patterns.
    3. Creates an output grid where each cell represents the intersection of
       unique horizontal and vertical patterns from the input grid.
    
    This results in a condensed version of the input grid that preserves the
    essential color transition information.
    """
    # Get unique horizontal patterns
    horizontal_patterns = []
    for row in range(len(input_grid.values)):
        pattern = get_row_pattern(input_grid, row)
        if pattern not in horizontal_patterns:
            horizontal_patterns.append(pattern)

    # Get unique vertical patterns
    vertical_patterns = []
    for col in range(len(input_grid.values[0])):
        pattern = get_column_pattern(input_grid, col)
        if pattern not in vertical_patterns:
            vertical_patterns.append(pattern)

    # Create output grid
    output_height = len(horizontal_patterns)
    output_width = len(vertical_patterns)
    output_values = [[0 for _ in range(output_width)] for _ in range(output_height)]

    for i in range(output_height):
        for j in range(output_width):
            # Find intersection point
            row = next(r for r in range(len(input_grid.values)) if get_row_pattern(input_grid, r) == horizontal_patterns[i])
            col = next(c for c in range(len(input_grid.values[0])) if get_column_pattern(input_grid, c) == vertical_patterns[j])
            output_values[i][j] = input_grid.values[row][col]

    return ColoredGrid(values=output_values)
