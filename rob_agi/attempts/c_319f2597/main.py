from rob_agi.colored_grid import ColoredGrid

def solve_319f2597(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a T-shaped black region and filling the bottom-left quadrant.
    
    The transformation follows these steps:
    1. Identify existing black squares and determine the position of the vertical stripe.
    2. Create a 2-column wide vertical black stripe.
    3. Determine the position of the horizontal line based on existing black squares or grid dimensions.
    4. Create a horizontal black line across the entire width of the grid.
    5. Fill the bottom-left quadrant (below the horizontal line and left of the vertical stripe) with black.
    6. Preserve the original grid in the other quadrants.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the T-shape and filled bottom-left quadrant.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Determine vertical stripe position
    black_cols = [col for col in range(cols) if any(input_grid.get_cell(row, col) == 0 for row in range(rows))]
    if black_cols:
        stripe_left = max(0, min(black_cols) - 1)
    else:
        stripe_left = cols // 2 - 1
    stripe_right = stripe_left + 1
    
    # Create vertical stripe
    for row in range(rows):
        output_grid.set_cell(row, stripe_left, 0)
        output_grid.set_cell(row, stripe_right, 0)
    
    # Determine horizontal line position
    black_rows = [row for row in range(rows) if 0 in input_grid.values[row]]
    if black_rows:
        horizontal_line = min(black_rows)
    else:
        horizontal_line = rows // 3
    
    # Create horizontal line
    for col in range(cols):
        output_grid.set_cell(horizontal_line, col, 0)
    
    # Fill bottom-left quadrant
    for row in range(horizontal_line + 1, rows):
        for col in range(stripe_left):
            output_grid.set_cell(row, col, 0)
    
    return output_grid
