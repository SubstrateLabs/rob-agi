from rob_agi.colored_grid import ColoredGrid

def solve_423a55dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying a diagonal shift and non-linear compression to colored cells.
    
    1. Identify the shape's boundaries.
    2. Apply a diagonal shift that increases for each row and varies across columns.
    3. Apply non-linear compression, with maximum compression in the middle rows.
    4. Preserve the overall structure of the shape, including any holes.
    5. Maintain the bottom row's width while compressing upper rows.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find shape boundaries
    top = min((r for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)
    left = min((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)
    bottom = max((r for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=rows-1)
    right = max((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=cols-1)
    
    shape_height = bottom - top + 1
    shape_width = right - left + 1
    max_shift = shape_width // 2
    max_compression = 0.5

    # Transform the shape
    for r in range(top, bottom + 1):
        row_pos = (r - top) / (bottom - top) if bottom > top else 0
        shift = int(row_pos * max_shift)
        compression = 1 - max_compression * (1 - abs(2 * row_pos - 1))
        
        for c in range(left, right + 1):
            if input_grid.values[r][c] != 0:
                col_pos = (c - left) / (right - left) if right > left else 0
                new_col = int(left + (c - left) * compression + shift * (1 + col_pos))
                if 0 <= new_col < cols:
                    output_grid.values[r][new_col] = input_grid.values[r][c]

    # Post-processing: fill gaps and maintain bottom row width
    for r in range(top, bottom + 1):
        non_zero_cells = [c for c in range(cols) if output_grid.values[r][c] != 0]
        if non_zero_cells:
            left, right = min(non_zero_cells), max(non_zero_cells)
            color = output_grid.values[r][left]
            for c in range(left, right + 1):
                if output_grid.values[r][c] == 0:
                    output_grid.values[r][c] = color

    return output_grid
