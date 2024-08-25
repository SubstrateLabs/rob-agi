from rob_agi.colored_grid import ColoredGrid

def solve_2685904e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending certain colors vertically.
    
    The solution follows these steps:
    1. Analyze the source row (second to last row) of the grid.
    2. Identify target colors: colors with the minimum frequency in the source row.
    3. For each target color, determine the extension height (min of color frequency and 3).
    4. Extend target colors upwards from the row above the source row, respecting the gray row and top 4 rows.
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
    color_frequency = {}
    for color in source_row:
        color_frequency[color] = color_frequency.get(color, 0) + 1
    min_frequency = min(color_frequency.values())

    # Identify target colors
    target_colors = [color for color, freq in color_frequency.items() if freq == min_frequency]

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
                if row >= 4:  # Don't modify rows 0-3
                    output_grid.values[row][col] = color

    return output_grid
