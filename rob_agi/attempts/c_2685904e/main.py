from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_2685904e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending certain colors vertically.
    
    The solution follows these steps:
    1. Analyze the source row (second to last row) of the grid.
    2. Determine the target frequency:
       - If any color appears exactly twice, target frequency is 2.
       - Otherwise, target frequency is the minimum frequency in the row.
    3. Identify colors to extend: all colors that appear at the target frequency.
    4. Extend these colors vertically from two rows above the gray row up to the row above the source row.
    5. Leave the top 4 rows, gray row, source row, bottom row, and non-extended columns unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Analyze the source row (second to last row)
    source_row = input_grid.values[-2]
    color_counts = Counter(source_row)
    
    # Determine the target frequency
    if 2 in color_counts.values():
        target_frequency = 2
    else:
        target_frequency = min(color_counts.values())
    
    # Identify colors to extend
    colors_to_extend = {color for color, count in color_counts.items() if count == target_frequency}
    
    # Calculate the extension range
    extension_start = 4  # Two rows above the gray row
    extension_end = 7  # Row above the source row
    
    # Extend the identified colors
    for col in range(len(source_row)):
        if source_row[col] in colors_to_extend:
            for row in range(extension_start, extension_end + 1):
                output_grid.values[row][col] = source_row[col]
    
    return output_grid
