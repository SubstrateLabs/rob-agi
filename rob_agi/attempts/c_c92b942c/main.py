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

    # Repeat the input pattern
    for r in range(output_rows):
        for c in range(output_cols):
            output_grid.set_cell(r, c, input_grid.get_cell(r % input_rows, c % input_cols))

    # Add blue crosses
    for r in range(input_rows):
        for c in range(input_cols):
            if input_grid.get_cell(r, c) not in [0, 1, 3]:
                for i in range(3):
                    for j in range(3):
                        output_r, output_c = r * 3 + i, c * 3 + j
                        if i == 1 or j == 1:
                            if output_grid.get_cell(output_r, output_c) == 0:
                                output_grid.set_cell(output_r, output_c, 1)

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
