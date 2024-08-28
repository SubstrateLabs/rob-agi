from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the pattern, centering it, and adding a blue border.

    1. Finds the bounding box of all non-black cells in the input grid.
    2. Creates a new grid with dimensions that are the larger of:
       a) Pattern dimensions + 2 (for the border)
       b) 10x10 (minimum size requirement)
    3. Fills the new grid with blue (1) as a starting point.
    4. Copies the pattern from the input grid to the center of the new grid.
    5. Preserves the count and relative positions of all colors (except black) from the input grid.
    6. Ensures the outermost layer is entirely blue.
    7. Places isolated colors in specific corners if they're not part of the main pattern.
    8. Crops the output grid to remove unnecessary blue border, maintaining minimum 10x10 size.

    Returns a new ColoredGrid object representing the transformed pattern with a blue border.
    """
    # Find the bounding box
    rows, cols = input_grid.get_dimensions()
    min_row, max_row, min_col, max_col = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:  # Non-black cell
                min_row, max_row = min(min_row, r), max(max_row, r)
                min_col, max_col = min(min_col, c), max(max_col, c)
    
    # Calculate new grid dimensions
    pattern_height = max_row - min_row + 1
    pattern_width = max_col - min_col + 1
    output_height = max(pattern_height + 2, 10)
    output_width = max(pattern_width + 2, 10)
    
    # Create new grid filled with blue
    output_grid = ColoredGrid(values=[[1 for _ in range(output_width)] for _ in range(output_height)])
    
    # Calculate padding for centering
    pad_top = (output_height - pattern_height) // 2
    pad_left = (output_width - pattern_width) // 2
    
    # Copy non-black cells from input to output, centered
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] != 0:
                output_grid.values[r - min_row + pad_top][c - min_col + pad_left] = input_grid.values[r][c]
    
    # Count colors in input grid (excluding black)
    color_count = {i: input_grid.count_color(i) for i in range(1, 10)}
    
    # Find isolated colors
    isolated_colors = []
    for color, count in color_count.items():
        if count == 1 and output_grid.count_color(color) == 0:
            isolated_colors.append(color)
    
    # Place isolated colors in specific corners
    corners = [(1, 1), (1, output_width-2), (output_height-2, 1), (output_height-2, output_width-2)]
    for color, (r, c) in zip(isolated_colors, corners):
        output_grid.values[r][c] = color
    
    # Ensure color preservation for non-isolated colors
    for color, count in color_count.items():
        if color not in isolated_colors:
            while output_grid.count_color(color) < count:
                for r in range(1, output_height - 1):
                    for c in range(1, output_width - 1):
                        if output_grid.values[r][c] == 1:
                            output_grid.values[r][c] = color
                            break
                    if output_grid.count_color(color) == count:
                        break
    
    # Ensure border integrity
    for r in range(output_height):
        output_grid.values[r][0] = output_grid.values[r][-1] = 1
    for c in range(output_width):
        output_grid.values[0][c] = output_grid.values[-1][c] = 1
    
    # Crop the output grid to remove unnecessary blue border
    crop_top, crop_bottom, crop_left, crop_right = 0, output_height - 1, 0, output_width - 1
    while crop_top < output_height - 1 and all(output_grid.values[crop_top][c] == 1 for c in range(output_width)):
        crop_top += 1
    while crop_bottom > 0 and all(output_grid.values[crop_bottom][c] == 1 for c in range(output_width)):
        crop_bottom -= 1
    while crop_left < output_width - 1 and all(output_grid.values[r][crop_left] == 1 for r in range(output_height)):
        crop_left += 1
    while crop_right > 0 and all(output_grid.values[r][crop_right] == 1 for r in range(output_height)):
        crop_right -= 1
    
    # Ensure minimum size of 10x10
    crop_height = crop_bottom - crop_top + 1
    crop_width = crop_right - crop_left + 1
    if crop_height < 10:
        extra = 10 - crop_height
        crop_top = max(0, crop_top - extra // 2)
        crop_bottom = min(output_height - 1, crop_bottom + (extra + 1) // 2)
    if crop_width < 10:
        extra = 10 - crop_width
        crop_left = max(0, crop_left - extra // 2)
        crop_right = min(output_width - 1, crop_right + (extra + 1) // 2)
    
    # Create the final cropped grid
    final_grid = ColoredGrid(values=[row[crop_left:crop_right+1] for row in output_grid.values[crop_top:crop_bottom+1]])
    
    return final_grid
