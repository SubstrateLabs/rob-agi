from rob_agi.colored_grid import ColoredGrid

def solve_2685904e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending certain colors vertically.
    
    The solution follows these steps:
    1. Analyze the colored sequence row (second to last row) of the grid.
    2. Identify target colors: colors that appear more than once but are not the most frequent.
       If no such colors exist, select the second least frequent color(s).
    3. For each target color, determine the extension height (min of color frequency and 3).
    4. Extend target colors upwards from the row above the colored sequence row,
       respecting the gray row and top 4 rows.
    5. Leave the top 4 rows, gray row, colored sequence row, bottom row, and non-extended columns unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Analyze the colored sequence row (second to last row)
    source_row = input_grid.values[-2]
    color_frequency = {}
    for color in source_row:
        if color != 0:  # Ignore black (empty space)
            color_frequency[color] = color_frequency.get(color, 0) + 1
    
    # Identify target colors
    max_freq = max(color_frequency.values())
    min_freq = min(color_frequency.values())
    target_colors = [color for color, freq in color_frequency.items() 
                     if freq > 1 and freq != max_freq]
    
    # If no colors meet the criteria, select the second least frequent color(s)
    if not target_colors:
        second_min_freq = min(freq for freq in color_frequency.values() if freq > min_freq)
        target_colors = [color for color, freq in color_frequency.items() 
                         if freq == second_min_freq]

    # Determine extension details for each target color
    extension_details = {}
    for color in target_colors:
        columns = [i for i, c in enumerate(source_row) if c == color]
        extension_height = min(color_frequency[color], 3)
        extension_details[color] = (columns, extension_height)

    # Extend the target colors
    for color, (columns, height) in extension_details.items():
        for col in columns:
            for row in range(-3, -3-height, -1):
                if 4 <= row < -2:  # Don't modify top 4 rows, gray row, or colored sequence row
                    output_grid.values[row][col] = color

    return output_grid
