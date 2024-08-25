from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_2685904e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending certain colors vertically.
    
    The solution follows these steps:
    1. Analyze the source row (second to last row) of the grid.
    2. Identify colors to extend:
       - Colors that appear more than once, excluding the most common color.
       - If no such colors, use the least frequent color(s) excluding the most common.
    3. Extend these colors vertically from the source row up to two rows above the gray row.
    4. Leave the top rows, gray row, bottom row, and non-extended columns unchanged.
    
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
    
    # Find the most common color and its count
    most_common_color, most_common_count = color_counts.most_common(1)[0]
    
    # Identify colors to extend
    colors_to_extend = [color for color, count in color_counts.items() 
                        if count > 1 and color != most_common_color]
    
    # If no colors appear more than once (excluding most common), use least frequent
    if not colors_to_extend:
        min_count = min(count for color, count in color_counts.items() if color != most_common_color)
        colors_to_extend = [color for color, count in color_counts.items() 
                            if count == min_count and color != most_common_color]
    
    # Find the gray row (should be the 7th row from the top)
    gray_row_index = next(i for i, row in enumerate(input_grid.values) if 5 in row)
    
    # Calculate the extension range
    extension_start = gray_row_index - 2
    extension_end = rows - 2  # Second to last row
    
    # Extend the identified colors
    for col in range(cols):
        if source_row[col] in colors_to_extend:
            for row in range(extension_start, extension_end):
                output_grid.values[row][col] = source_row[col]
    
    return output_grid
