from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the smallest rectangular area containing all non-black cells from the input grid,
    preserving their relative positions and colors.

    1. Finds the bounding box of all non-black cells in the input grid.
    2. Creates a new grid with the dimensions of the bounding box.
    3. Copies all non-black cells from the input grid to their corresponding positions in the new grid.
    4. Ensures all colors from the input grid are preserved in the output grid.

    Returns a new ColoredGrid object representing the extracted and preserved pattern.
    """
    # Find the bounding box
    rows, cols = input_grid.get_dimensions()
    min_row, max_row, min_col, max_col = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:  # Non-black cell
                min_row, max_row = min(min_row, r), max(max_row, r)
                min_col, max_col = min(min_col, c), max(max_col, c)
    
    # Create new grid with bounding box dimensions
    output_height = max_row - min_row + 1
    output_width = max_col - min_col + 1
    output_grid = ColoredGrid(values=[[1 for _ in range(output_width)] for _ in range(output_height)])
    
    # Copy non-black cells from input to output
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] != 0:
                output_grid.values[r - min_row][c - min_col] = input_grid.values[r][c]
    
    # Ensure color preservation
    color_count = {i: input_grid.count_color(i) for i in range(10) if i != 0}
    for color, count in color_count.items():
        while output_grid.count_color(color) < count:
            for r in range(output_height):
                for c in range(output_width):
                    if output_grid.values[r][c] == 1:
                        output_grid.values[r][c] = color
                        break
                if output_grid.count_color(color) == count:
                    break
    
    return output_grid
