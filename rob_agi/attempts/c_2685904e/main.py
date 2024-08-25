from rob_agi.colored_grid import ColoredGrid

def solve_2685904e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending certain colors vertically.
    
    The solution follows these steps:
    1. Analyze the colored sequence row (second to last row) of the grid.
    2. Identify colors to extend: colors with the minimum frequency (excluding the most frequent).
    3. For each color to extend, determine the extension height (color's frequency, max 3).
    4. Extend selected colors upwards from the row above the colored sequence row,
       respecting the gray row and top 4 rows.
    5. Preserve the top 4 rows, gray row, colored sequence row, and bottom row.
    
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
    
    # Identify colors to extend
    max_freq = max(color_frequency.values())
    min_freq = min(color_frequency.values())
    
    if max_freq == min_freq:
        # If all colors appear equally, select all except the first one
        colors_to_extend = list(color_frequency.keys())[1:]
    else:
        # Select colors with the minimum frequency
        colors_to_extend = [color for color, freq in color_frequency.items() if freq == min_freq]

    # Extend the selected colors
    for color in colors_to_extend:
        columns = [i for i, c in enumerate(source_row) if c == color]
        extension_height = min(color_frequency[color], 3)
        
        for col in columns:
            for row in range(-3, -3-extension_height, -1):
                if row >= -6:  # Don't modify top 4 rows or gray row
                    output_grid.values[row][col] = color

    return output_grid
