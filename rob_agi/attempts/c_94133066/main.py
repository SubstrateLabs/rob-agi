from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the pattern, centering it, and adding a blue border.

    1. Finds the bounding box of all non-black cells in the input grid.
    2. Creates a new grid with dimensions that are the larger of:
       a) Pattern dimensions + 2 (for the border)
       b) 9x9 (minimum size requirement)
    3. Fills the new grid with blue (1) as a starting point.
    4. Copies the pattern from the input grid to the center of the new grid.
    5. Preserves the count of all colors (except black and blue) from the input grid.
    6. Ensures the outermost layer is entirely blue.

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
    output_height = max(pattern_height + 2, 9)  # Ensure minimum size of 9x9
    output_width = max(pattern_width + 2, 9)
    
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
    
    # Count colors in input grid
    color_count = {i: input_grid.count_color(i) for i in range(10) if i != 0 and i != 1}
    
    # Ensure color preservation
    for color, count in color_count.items():
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
    
    return output_grid
