from rob_agi.colored_grid import ColoredGrid

def solve_c92b942c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by repeating it in a 3x3 pattern, adding blue crosses
    around non-zero cells within each repetition, and green corners to repetitions
    containing non-zero, non-blue, non-green cells.

    1. Create an output grid that is 3 times larger in each dimension
    2. Repeat the entire input pattern in a 3x3 grid
    3. Add blue crosses around non-zero cells, confined to their pattern repetition
    4. Add green corners to pattern repetitions with non-zero, non-blue, non-green cells
    5. Return the transformed grid
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 3, input_cols * 3
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    def is_within_pattern(row, col, pattern_start_row, pattern_start_col):
        return (pattern_start_row <= row < pattern_start_row + input_rows and
                pattern_start_col <= col < pattern_start_col + input_cols)

    def get_pattern_bounds(row, col):
        pattern_start_row = (row // input_rows) * input_rows
        pattern_start_col = (col // input_cols) * input_cols
        return pattern_start_row, pattern_start_col

    # Repeat the input pattern
    for r in range(output_rows):
        for c in range(output_cols):
            output_grid.set_cell(r, c, input_grid.get_cell(r % input_rows, c % input_cols))

    # Add blue crosses
    for r in range(output_rows):
        for c in range(output_cols):
            if output_grid.get_cell(r, c) not in [0, 1, 3]:
                pattern_start_row, pattern_start_col = get_pattern_bounds(r, c)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    while (0 <= nr < output_rows and 0 <= nc < output_cols and
                           is_within_pattern(nr, nc, pattern_start_row, pattern_start_col)):
                        if output_grid.get_cell(nr, nc) == 0:
                            output_grid.set_cell(nr, nc, 1)
                        nr, nc = nr + dr, nc + dc

    # Add green corners
    for r in range(0, output_rows, input_rows):
        for c in range(0, output_cols, input_cols):
            if any(output_grid.get_cell(r+dr, c+dc) not in [0, 1, 3]
                   for dr in range(input_rows) for dc in range(input_cols)):
                for corner_r, corner_c in [(r, c), (r, c+input_cols-1),
                                           (r+input_rows-1, c), (r+input_rows-1, c+input_cols-1)]:
                    if output_grid.get_cell(corner_r, corner_c) == 0:
                        output_grid.set_cell(corner_r, corner_c, 3)

    return output_grid
