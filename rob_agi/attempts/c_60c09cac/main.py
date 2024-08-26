from rob_agi.colored_grid import ColoredGrid

def solve_60c09cac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it.
    Each cell in the input grid becomes a 2x2 square of the same color in the output grid.
    The output grid is exactly twice the size of the input grid in both dimensions.
    """
    rows, cols = input_grid.get_dimensions()
    output_rows = 2 * rows
    output_cols = 2 * cols

    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    for r in range(rows):
        for c in range(cols):
            output_r = 2 * r
            output_c = 2 * c
            color = input_grid.values[r][c]
            output_grid.values[output_r][output_c] = color
            output_grid.values[output_r][output_c+1] = color
            output_grid.values[output_r+1][output_c] = color
            output_grid.values[output_r+1][output_c+1] = color

    return output_grid
