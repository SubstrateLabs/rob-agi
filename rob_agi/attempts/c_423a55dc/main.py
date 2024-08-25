from rob_agi.colored_grid import ColoredGrid

def solve_423a55dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying a diagonal shift to colored cells.
    
    1. If the shape is not touching the left edge, shift colored cells up-left until they reach the left edge.
    2. Apply a diagonal shift, moving each row one step to the right compared to the row above it.
    3. Preserve the overall dimensions and structure of the shape, including any holes.
    4. Maintain the width of the shape by extending rightward if necessary.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find the leftmost column and topmost row with non-zero cells
    left_edge = min((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)
    top_edge = min((r for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)

    # Calculate the width of the shape
    right_edge = max((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=cols-1)
    shape_width = right_edge - left_edge + 1

    # Perform the transformation
    shift = 0
    for r in range(top_edge, rows):
        row_content = [input_grid.values[r][c] for c in range(left_edge, cols) if input_grid.values[r][c] != 0]
        if not row_content:
            continue
        
        for i, value in enumerate(row_content):
            new_col = max(0, i - shift)
            output_grid.values[r][new_col] = value
        
        # Extend the row to maintain shape width
        last_non_zero = max((c for c in range(cols) if output_grid.values[r][c] != 0), default=-1)
        if last_non_zero >= 0:
            for c in range(last_non_zero + 1, min(cols, last_non_zero + shape_width)):
                output_grid.values[r][c] = output_grid.values[r][last_non_zero]
        
        shift += 1

    return output_grid
