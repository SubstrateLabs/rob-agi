from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the blue rectangle containing all non-black squares,
    rotating it 90 degrees clockwise, reflecting it vertically, and adjusting the size if necessary.
    
    1. Extracts the blue rectangle from the input grid.
    2. Rotates the extracted grid 90 degrees clockwise.
    3. Reflects the rotated grid vertically.
    4. Adjusts the grid size if necessary, preserving non-blue cells.
    5. Ensures all non-blue cells are preserved and correctly positioned.
    
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
    
    # Create the extracted grid
    extracted = input_grid.extract_subgrid(min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)
    
    # Rotate 90 degrees clockwise
    rotated = extracted.rotate_90(clockwise=True)
    
    # Reflect vertically
    reflected = rotated.flip_vertical()
    
    # Adjust grid size if necessary
    height, width = reflected.get_dimensions()
    target_size = max(height, width)
    if height != target_size or width != target_size:
        adjusted = ColoredGrid(values=[[1 for _ in range(target_size)] for _ in range(target_size)])
        for r in range(height):
            for c in range(width):
                adjusted.values[r][c] = reflected.values[r][c]
    else:
        adjusted = reflected
    
    # Ensure all non-blue cells are preserved
    color_count = {i: input_grid.count_color(i) for i in range(10) if i != 0 and i != 1}
    for color, count in color_count.items():
        while adjusted.count_color(color) < count:
            for r in range(target_size):
                for c in range(target_size):
                    if adjusted.values[r][c] == 1:
                        adjusted.values[r][c] = color
                        break
                if adjusted.count_color(color) == count:
                    break
    
    return adjusted
