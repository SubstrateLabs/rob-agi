from rob_agi.colored_grid import ColoredGrid

def solve_319f2597(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a cross-shaped black region.
    
    The transformation follows these steps:
    1. Find the position of the vertical stripe based on existing black squares or set to 1/3 of grid width.
    2. Create a 2-column wide vertical black stripe from top to bottom.
    3. Find the position of the horizontal line based on existing black squares or set to 1/2 of grid height.
    4. Create a 2-row thick horizontal black line across the entire width.
    5. Preserve the original grid in the other areas.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the cross-shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find vertical stripe position
    stripe_left = next((col for col in range(cols) if any(input_grid.get_cell(row, col) == 0 for row in range(rows))), cols // 3)
    stripe_right = stripe_left + 1
    
    # Find horizontal line position
    horizontal_top = next((row for row in range(rows) if any(input_grid.get_cell(row, col) == 0 for col in range(cols))), rows // 2)
    horizontal_bottom = horizontal_top + 1
    
    # Create vertical stripe
    for row in range(rows):
        output_grid.set_cell(row, stripe_left, 0)
        output_grid.set_cell(row, stripe_right, 0)
    
    # Create horizontal line
    for col in range(cols):
        output_grid.set_cell(horizontal_top, col, 0)
        output_grid.set_cell(horizontal_bottom, col, 0)
    
    return output_grid
