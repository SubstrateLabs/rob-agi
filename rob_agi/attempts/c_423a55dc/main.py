from rob_agi.colored_grid import ColoredGrid

def solve_423a55dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying a diagonal shift and compression to colored cells.
    
    1. Find the topmost row and leftmost column of the shape.
    2. Apply a diagonal shift that increases for each row.
    3. Progressively compress the shape horizontally from top to bottom.
    4. Preserve the overall structure of the shape, including any holes.
    5. Maintain the width at the widest point while allowing individual rows to become narrower.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find the topmost row, leftmost column, and shape dimensions
    top_edge = min((r for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)
    left_edge = min((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)
    bottom_edge = max((r for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=rows-1)
    right_edge = max((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=cols-1)
    shape_height = bottom_edge - top_edge + 1
    shape_width = right_edge - left_edge + 1

    # Transform the shape
    for r in range(top_edge, rows):
        shift = r - top_edge
        compression = (r - top_edge) / shape_height if shape_height > 1 else 0

        for c in range(left_edge, cols):
            if input_grid.values[r][c] != 0:
                new_col = int(left_edge + (c - left_edge) * (1 - compression) + shift)
                if 0 <= new_col < cols:
                    output_grid.values[r][new_col] = input_grid.values[r][c]

    # Post-processing: fill gaps
    for r in range(rows):
        non_zero_cells = [c for c in range(cols) if output_grid.values[r][c] != 0]
        if non_zero_cells:
            left, right = min(non_zero_cells), max(non_zero_cells)
            for c in range(left, right + 1):
                if output_grid.values[r][c] == 0:
                    output_grid.values[r][c] = output_grid.values[r][left]

    return output_grid
