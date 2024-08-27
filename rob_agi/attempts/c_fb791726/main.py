from rob_agi.colored_grid import ColoredGrid

def solve_fb791726(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Expanding its size to (2n+2) x (2n+2), where n is the input grid size
    2. Adding green (3) separator row and column in the middle
    3. Copying non-black cells to their new positions:
       - Top-left quadrant stays in place
       - Top-right quadrant moves to bottom-left
       - Bottom-left quadrant moves to top-right
       - Bottom-right quadrant stays in place
    4. Filling the rest with black (0)
    5. Handling odd-sized grids by treating middle row/column as part of bottom/right quadrants
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2 + 2, input_cols * 2 + 2
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    # Add green separators
    mid_row, mid_col = input_rows, input_cols
    output_grid.values[mid_row] = [3] * output_cols
    for i in range(output_rows):
        output_grid.values[i][mid_col] = 3

    # Copy and transform the input grid
    input_mid_row, input_mid_col = (input_rows - 1) // 2, (input_cols - 1) // 2
    for i in range(input_rows):
        for j in range(input_cols):
            if input_grid.values[i][j] != 0:
                if i <= input_mid_row and j <= input_mid_col:  # Top-left quadrant
                    new_i, new_j = i, j
                elif i <= input_mid_row and j > input_mid_col:  # Top-right quadrant
                    new_i, new_j = i + mid_row + 1, j
                elif i > input_mid_row and j <= input_mid_col:  # Bottom-left quadrant
                    new_i, new_j = i, j + mid_col + 1
                else:  # Bottom-right quadrant
                    new_i, new_j = i + mid_row + 1, j + mid_col + 1
                output_grid.values[new_i][new_j] = input_grid.values[i][j]

    return output_grid
