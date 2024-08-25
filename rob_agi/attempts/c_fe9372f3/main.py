from rob_agi.colored_grid import ColoredGrid

def solve_fe9372f3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following pattern:
    1. Locate and copy the red (2) cross from the input grid.
    2. Draw blue (1) diagonal lines from the corners of the red cross outward.
    3. Place yellow (4) cells on the center lines and at regular intervals.
    4. Connect yellow cells with sky blue (8) lines, forming a diamond shape.
    5. Ensure symmetry around the red cross.
    6. Fill remaining cells with black (0).
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find the center of the red cross
    center_row, center_col = None, None
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 2:
                center_row, center_col = r, c
                break
        if center_row is not None:
            break

    # Copy red cross
    for r in range(center_row - 1, center_row + 2):
        for c in range(center_col - 1, center_col + 2):
            if 0 <= r < rows and 0 <= c < cols and input_grid.get_cell(r, c) == 2:
                output_grid.set_cell(r, c, 2)

    # Draw blue diagonal lines
    for offset in range(1, max(rows, cols)):
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = center_row + offset * dr, center_col + offset * dc
            if 0 <= r < rows and 0 <= c < cols:
                output_grid.set_cell(r, c, 1)

    # Place yellow cells and connect with sky blue lines
    for r in range(rows):
        if r == center_row or r % 4 == 0:
            for c in range(0, cols, 4):
                output_grid.set_cell(r, c, 4)
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 8)

    for c in range(cols):
        if c == center_col or c % 4 == 0:
            for r in range(0, rows, 4):
                output_grid.set_cell(r, c, 4)
            for r in range(rows):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 8)

    return output_grid
