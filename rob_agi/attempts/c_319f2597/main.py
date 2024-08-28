from rob_agi.colored_grid import ColoredGrid

def solve_319f2597(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating an L-shaped black region.
    
    The transformation follows these steps:
    1. Analyze the input grid for existing black cells.
    2. Determine the position of the vertical stripe (2 columns) in the right half.
    3. Determine the position of the horizontal stripe (2 rows) in the upper half.
    4. Create the L-shape by filling in the vertical stripe and horizontal stripe.
    5. Preserve existing black cells from the input grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the L-shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Analyze the input grid
    col_counts = [sum(1 for r in range(rows) if input_grid.get_cell(r, c) == 0) for c in range(cols)]
    row_counts = [sum(1 for c in range(cols//2) if input_grid.get_cell(r, c) == 0) for r in range(rows)]
    
    # Determine vertical stripe
    right_half_cols = list(range(cols//2, cols))
    stripe_cols = sorted(right_half_cols, key=lambda c: (-col_counts[c], -c))[:2]
    stripe_left, stripe_right = min(stripe_cols), max(stripe_cols)
    
    # Determine horizontal stripe
    upper_half_rows = list(range(rows//2))
    stripe_rows = sorted(upper_half_rows, key=lambda r: (-row_counts[r], -r))[:2]
    horizontal_top, horizontal_bottom = min(stripe_rows), max(stripe_rows)
    
    # If no black cells found, use default positions
    if not any(col_counts):
        stripe_left, stripe_right = cols - 2, cols - 1
    if not any(row_counts):
        horizontal_top, horizontal_bottom = rows // 4, rows // 4 + 1
    
    # Create L-shape
    for row in range(rows):
        for col in range(cols):
            if stripe_left <= col <= stripe_right:
                output_grid.set_cell(row, col, 0)
            if horizontal_top <= row <= horizontal_bottom and col <= stripe_right:
                output_grid.set_cell(row, col, 0)
    
    # Preserve existing black cells
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 0 and output_grid.get_cell(r, c) != 0:
                output_grid.set_cell(r, c, 0)
    
    return output_grid
