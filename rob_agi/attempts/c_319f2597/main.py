from rob_agi.colored_grid import ColoredGrid

def solve_319f2597(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating an L-shaped black region.
    
    The transformation follows these steps:
    1. Analyze the input grid for existing black cells.
    2. Determine the position of the vertical stripe (2 columns).
    3. Determine the position of the horizontal line (2 rows).
    4. Create the L-shape by filling in the vertical stripe and horizontal line.
    5. Preserve existing black cells from the input grid.
    6. Extend the L-shape to the edges of the grid.
    7. Ensure the intersection of the vertical stripe and horizontal line is fully black.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the L-shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find existing black cells
    black_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 0]
    
    # Determine vertical stripe
    col_counts = {}
    for _, c in black_cells:
        col_counts[c] = col_counts.get(c, 0) + 1
    stripe_cols = sorted(col_counts, key=lambda x: (-col_counts[x], x))[:2]
    stripe_left, stripe_right = min(stripe_cols), max(stripe_cols)
    
    # Determine horizontal line
    row_counts = {}
    for r, _ in black_cells:
        row_counts[r] = row_counts.get(r, 0) + 1
    line_rows = sorted(row_counts, key=lambda x: (-row_counts[x], x))[:2]
    horizontal_top, horizontal_bottom = min(line_rows), max(line_rows)
    
    # Create L-shape
    for row in range(rows):
        for col in range(cols):
            if stripe_left <= col <= stripe_right:
                output_grid.set_cell(row, col, 0)
            if horizontal_top <= row <= horizontal_bottom and col <= stripe_right:
                output_grid.set_cell(row, col, 0)
    
    # Preserve existing black cells
    for r, c in black_cells:
        output_grid.set_cell(r, c, 0)
    
    # Extend L-shape to edges
    for row in range(rows):
        output_grid.set_cell(row, 0, 0)
    for col in range(cols):
        output_grid.set_cell(0, col, 0)
    
    # Ensure intersection is fully black
    for row in range(horizontal_top, horizontal_bottom + 1):
        for col in range(stripe_left, stripe_right + 1):
            output_grid.set_cell(row, col, 0)
    
    return output_grid
