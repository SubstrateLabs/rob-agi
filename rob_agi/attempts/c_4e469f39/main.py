from rob_agi.colored_grid import ColoredGrid

def solve_4e469f39(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by identifying gray (5) shapes and adding a red (2) outline and fill.
    
    The function performs the following steps:
    1. Analyze the grid to find the bounding box of all gray shapes.
    2. Create a continuous top red line above all shapes, extending to the right edge of the grid.
    3. Draw vertical red lines on both sides of the gray shapes and fill the inside with red.
    4. Handle the area below gray shapes by extending red fill downwards.
    
    Args:
    input_grid (ColoredGrid): The input grid containing gray shapes.
    
    Returns:
    ColoredGrid: A new grid with a red outline and fill added around and within all gray shapes.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Find bounding box of all gray shapes
    gray_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5]
    if not gray_cells:
        return output_grid
    
    top = min(r for r, _ in gray_cells)
    left = min(c for _, c in gray_cells)
    right = max(c for _, c in gray_cells)
    
    # Create top red line
    for col in range(left, cols):
        output_grid.values[top-1][col] = 2
    
    # Process columns
    for col in range(left, right + 1):
        inside = False
        top_gray = min(r for r in range(rows) if input_grid.values[r][col] == 5)
        bottom_gray = max(r for r in range(rows) if input_grid.values[r][col] == 5)
        
        for row in range(top - 1, rows):
            if input_grid.values[row][col] == 5:
                inside = not inside
            elif row < bottom_gray:
                if col == left or col == right or inside:
                    output_grid.values[row][col] = 2
            elif row > bottom_gray:
                output_grid.values[row][col] = 2
    
    return output_grid
