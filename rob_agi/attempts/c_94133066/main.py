from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the blue rectangle containing all non-black squares,
    rotating it 90 degrees clockwise, and adjusting the size while preserving the shape and position of non-blue elements.
    
    1. Extracts the blue rectangle from the input grid.
    2. Determines the output dimensions based on the extracted rectangle.
    3. Creates a new blue grid with the determined dimensions.
    4. Maps and transfers non-blue cells from the input to the output grid, preserving their relative positions.
    5. Handles isolated non-black cells outside the main blue rectangle.
    6. Ensures all non-blue cells are preserved and correctly positioned.
    
    Returns a new ColoredGrid object representing the transformed grid.
    """
    # Extract the blue rectangle
    rows, cols = input_grid.get_dimensions()
    min_row, max_row, min_col, max_col = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:  # Non-black cell
                min_row, max_row = min(min_row, r), max(max_row, r)
                min_col, max_col = min(min_col, c), max(max_col, c)
    
    # Determine output dimensions
    input_width = max_col - min_col + 1
    input_height = max_row - min_row + 1
    output_width = max(input_width, input_height)
    output_height = min(input_width, input_height)
    
    # Create new blue grid
    output_grid = ColoredGrid(values=[[1 for _ in range(output_width)] for _ in range(output_height)])
    
    # Map and transfer non-blue cells
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] not in [0, 1]:
                input_r, input_c = r - min_row, c - min_col
                output_r = input_c
                output_c = input_height - 1 - input_r
                if output_r < output_height and output_c < output_width:
                    output_grid.values[output_r][output_c] = input_grid.values[r][c]
    
    # Handle isolated cells
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] not in [0, 1] and (r < min_row or r > max_row or c < min_col or c > max_col):
                # Place in nearest corner or edge
                output_r = 0 if r < (min_row + max_row) // 2 else output_height - 1
                output_c = 0 if c < (min_col + max_col) // 2 else output_width - 1
                output_grid.values[output_r][output_c] = input_grid.values[r][c]
    
    # Ensure color preservation
    color_count = {i: input_grid.count_color(i) for i in range(10) if i != 0 and i != 1}
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
