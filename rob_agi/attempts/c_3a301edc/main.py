from rob_agi.colored_grid import ColoredGrid

def solve_3a301edc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a border around the main shape.
    
    1. Identifies the main shape in the input grid.
    2. Finds the innermost color of the shape.
    3. Determines the border thickness based on the size of the inner color region.
    4. Creates a new grid with the original shape and adds a border of the inner color.
    5. The border thickness is 1 for small inner regions (1-3 cells) and 3 for larger regions.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find the bounding box of the main shape
    min_row, min_col, max_row, max_col = rows, cols, 0, 0
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                min_row = min(min_row, r)
                min_col = min(min_col, c)
                max_row = max(max_row, r)
                max_col = max(max_col, c)
    
    # Find the innermost color
    outer_color = input_grid.values[min_row][min_col]
    inner_color = outer_color
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] != 0 and input_grid.values[r][c] != outer_color:
                inner_color = input_grid.values[r][c]
                break
        if inner_color != outer_color:
            break
    
    # Determine the size of the inner color region
    inner_region = input_grid.find_connected_regions(inner_color)[0]
    border_thickness = 3 if len(inner_region) > 3 else 1
    
    # Copy the original shape
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            output_grid.values[r][c] = input_grid.values[r][c]
    
    # Add the border
    new_min_row = max(0, min_row - border_thickness)
    new_min_col = max(0, min_col - border_thickness)
    new_max_row = min(rows - 1, max_row + border_thickness)
    new_max_col = min(cols - 1, max_col + border_thickness)
    
    for r in range(new_min_row, new_max_row + 1):
        for c in range(new_min_col, new_max_col + 1):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = inner_color
    
    return output_grid
