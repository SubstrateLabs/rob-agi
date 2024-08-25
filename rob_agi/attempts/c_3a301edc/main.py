from rob_agi.colored_grid import ColoredGrid

def solve_3a301edc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a border around the main shape.
    
    1. Identifies the main shape in the input grid.
    2. Finds the center color of the shape to use as the border color.
    3. Determines the border thickness based on the size of the entire shape.
    4. Creates a new grid with the original shape and adds a border of the center color.
    5. The border thickness is 1 for small shapes (area <= 9), 2 for medium shapes (9 < area <= 49),
       and 3 for large shapes (area > 49).
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
    
    # Find the center color
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    border_color = input_grid.values[center_row][center_col]
    if border_color == 0:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = center_row + dr, center_col + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 0:
                border_color = input_grid.values[nr][nc]
                break
    
    # Determine the border thickness
    shape_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    if shape_area <= 9:
        border_thickness = 1
    elif shape_area <= 49:
        border_thickness = 2
    else:
        border_thickness = 3
    
    # Copy the original shape
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            output_grid.values[r][c] = input_grid.values[r][c]
    
    # Add the border
    for r in range(max(0, min_row - border_thickness), min(rows, max_row + border_thickness + 1)):
        for c in range(max(0, min_col - border_thickness), min(cols, max_col + border_thickness + 1)):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color
    
    return output_grid
