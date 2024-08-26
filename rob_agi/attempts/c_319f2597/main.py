from rob_agi.colored_grid import ColoredGrid

def solve_319f2597(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a cross-shaped black region.
    
    The transformation follows these steps:
    1. Analyze the input grid for existing black cells and determine grid dimensions.
    2. Determine the position and width of the vertical stripe.
    3. Determine the position and thickness of the horizontal line.
    4. Create the vertical stripe from top to bottom.
    5. Create the horizontal line from left to right.
    6. Ensure the cross extends to all edges and intersects properly.
    7. Preserve original grid values outside the cross shape.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the cross-shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find existing black cells
    black_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 0]
    
    # Determine vertical stripe position and width
    if black_cells:
        stripe_left = min(c for _, c in black_cells)
        stripe_right = max(c for _, c in black_cells)
    else:
        stripe_left = cols // 3
        stripe_right = stripe_left + 1
    
    # Determine horizontal line position and thickness
    if black_cells:
        horizontal_top = min(r for r, _ in black_cells)
        horizontal_bottom = max(r for r, _ in black_cells)
    else:
        horizontal_top = rows // 2
        horizontal_bottom = horizontal_top + 1
    
    # Create vertical stripe
    for row in range(rows):
        for col in range(stripe_left, stripe_right + 1):
            output_grid.set_cell(row, col, 0)
    
    # Create horizontal line
    for col in range(cols):
        for row in range(horizontal_top, horizontal_bottom + 1):
            output_grid.set_cell(row, col, 0)
    
    # Ensure cross extends to edges
    for row in range(rows):
        output_grid.set_cell(row, 0, 0)
        output_grid.set_cell(row, cols - 1, 0)
    for col in range(cols):
        output_grid.set_cell(0, col, 0)
        output_grid.set_cell(rows - 1, col, 0)
    
    return output_grid
