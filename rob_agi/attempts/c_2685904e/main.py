from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_2685904e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending certain colors vertically.
    
    The solution follows these steps:
    1. Analyze the source row (second to last row) of the grid.
    2. Identify colors to extend:
       - Colors that appear exactly twice.
       - If no colors appear twice, use the least frequent color(s).
    3. Extend these colors vertically from two rows above the gray row up to the row above the source row.
    4. Leave the top rows, gray row, source row, bottom row, and non-extended columns unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Get the dimensions of the grid
    rows, cols = input_grid.get_dimensions()
    
    # Analyze the source row (second to last row)
    source_row = input_grid.values[-2]
    color_counts = Counter(source_row)
    
    # Identify colors to extend
    colors_appearing_twice = {color for color, count in color_counts.items() if count == 2}
    
    if colors_appearing_twice:
        colors_to_extend = colors_appearing_twice
    else:
        min_count = min(color_counts.values())
        colors_to_extend = {color for color, count in color_counts.items() if count == min_count}
    
    # Find the gray row (7th row from the top)
    gray_row_index = 6
    
    # Calculate the extension range
    extension_start = gray_row_index - 2
    extension_end = rows - 3  # Third to last row
    
    # Extend the identified colors
    for col in range(cols):
        if source_row[col] in colors_to_extend:
            for row in range(extension_start, extension_end):
                output_grid.values[row][col] = source_row[col]
    
    return output_grid
