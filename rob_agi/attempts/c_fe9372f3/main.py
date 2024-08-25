from rob_agi.colored_grid import ColoredGrid

def solve_fe9372f3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following pattern:
    1. Copy the red (2) cross from the input grid.
    2. Draw blue (1) diagonal lines from corners towards the center.
    3. Place yellow (4) cells on center lines every 4 cells.
    4. Connect yellow cells with sky blue (8) lines, forming a diamond shape.
    5. Fill remaining cells with black (0).
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Copy red cross
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 2:
                output_grid.set_cell(r, c, 2)

    # Draw blue diagonal lines
    for r in range(min(rows, cols)):
        output_grid.set_cell(r, r, 1)
        output_grid.set_cell(r, cols-1-r, 1)
        output_grid.set_cell(rows-1-r, r, 1)
        output_grid.set_cell(rows-1-r, cols-1-r, 1)

    # Place yellow cells and connect with sky blue lines
    center_rows = [rows // 2] if rows % 2 else [rows // 2 - 1, rows // 2]
    center_cols = [cols // 2] if cols % 2 else [cols // 2 - 1, cols // 2]

    for r in center_rows:
        for c in range(0, cols, 4):
            output_grid.set_cell(r, c, 4)
            for cc in range(c+1, cols):
                if output_grid.get_cell(r, cc) != 0:
                    break
                output_grid.set_cell(r, cc, 8)

    for c in center_cols:
        for r in range(0, rows, 4):
            output_grid.set_cell(r, c, 4)
            for rr in range(r+1, rows):
                if output_grid.get_cell(rr, c) != 0:
                    break
                output_grid.set_cell(rr, c, 8)

    return output_grid
