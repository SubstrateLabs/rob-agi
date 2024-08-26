from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the smallest rectangular area containing all non-black cells from the input grid,
    adds a blue border, preserves all colors and their counts, and centers the pattern.

    1. Finds the bounding box of all non-black cells in the input grid.
    2. Creates a new grid with the dimensions of the bounding box plus a blue border.
    3. Copies all non-black cells from the input grid to their corresponding positions in the new grid.
    4. Ensures all colors from the input grid are preserved in the output grid.
    5. Centers the pattern within the new grid.
    6. Fills any remaining cells with blue.

    Returns a new ColoredGrid object representing the extracted, preserved, and centered pattern with a blue border.
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
    output_height = pattern_height + 2  # Add 2 for the blue border
    output_width = pattern_width + 2   # Add 2 for the blue border
    
    # Create new grid
    output_grid = ColoredGrid(values=[[1 for _ in range(output_width)] for _ in range(output_height)])
    
    # Copy non-black cells from input to output
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] != 0:
                output_grid.values[r - min_row + 1][c - min_col + 1] = input_grid.values[r][c]
    
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
    
    return output_grid
