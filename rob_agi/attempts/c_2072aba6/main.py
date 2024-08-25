from rob_agi.colored_grid import ColoredGrid

def solve_2072aba6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid based on the following rules:
    1. The output grid is twice the size of the input in each dimension.
    2. Gray (5) squares in the input are transformed into 2x2 checkerboard patterns of blue (1) and red (2).
    3. Black (0) squares in the input remain as 2x2 black squares in the output.
    4. The overall pattern maintains a larger checkerboard structure across the entire 6x6 grid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(6)] for _ in range(6)])

    def get_start_color(row, col):
        return 1 if (row % 2 == 0 and col % 2 == 0) else 2

    def fill_block(row, col, start_color):
        output_grid.values[row][col] = start_color
        output_grid.values[row][col+1] = 3 - start_color
        output_grid.values[row+1][col] = 3 - start_color
        output_grid.values[row+1][col+1] = start_color

    for i in range(3):
        for j in range(3):
            output_row, output_col = i * 2, j * 2
            if input_grid.values[i][j] == 5:  # Gray square
                start_color = get_start_color(output_row, output_col)
                fill_block(output_row, output_col, start_color)

    return output_grid
