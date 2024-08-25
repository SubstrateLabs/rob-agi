from rob_agi.colored_grid import ColoredGrid

def solve_3a301edc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a border around the main shape.
    
    1. Identifies the main shape in the input grid.
    2. Finds the innermost color of the shape.
    3. Determines the border thickness based on the size of the inner color region.
    4. Creates a new grid with the original shape and adds a border of the inner color.
    5. The border thickness is 1 for small inner regions (1-3 cells) and 2 for larger regions.
    6. Preserves the original black space around the shape.
    
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
    inner_color = None
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] != 0:
                if inner_color is None or input_grid.values[r][c] != inner_color:
                    inner_color = input_grid.values[r][c]
    
    # Determine the border thickness
    shape_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    border_thickness = 1 if shape_area <= 9 else 2
    
    # Copy the original shape and add the border
    for r in range(max(0, min_row - border_thickness), min(rows, max_row + border_thickness + 1)):
        for c in range(max(0, min_col - border_thickness), min(cols, max_col + border_thickness + 1)):
            if min_row <= r <= max_row and min_col <= c <= max_col:
                output_grid.values[r][c] = input_grid.values[r][c]
            elif output_grid.values[r][c] == 0:
                output_grid.values[r][c] = inner_color
    
    return output_grid
