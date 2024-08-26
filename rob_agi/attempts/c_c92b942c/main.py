from rob_agi.colored_grid import ColoredGrid

def solve_c92b942c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 3x3 expanded pattern with the following rules:
    1. Create an output grid that is 3 times larger in each dimension.
    2. For each non-zero input cell:
       - Place its color in the center of the corresponding 3x3 area.
       - Surround it with blue (1) in a cross pattern, except for blue (1) and green (3).
    3. Apply a global blue (1) grid on every third row and column.
    4. Add green (3) corners in a checkerboard pattern of 3x3 areas.
    5. Repeat the pattern horizontally and vertically to fill the entire output grid.
    6. Never overwrite a non-black color with another color.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 3, input_cols * 3
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    # Process each input cell and repeat the pattern
    for r in range(output_rows):
        for c in range(output_cols):
            input_r, input_c = r // 3 % input_rows, c // 3 % input_cols
            color = input_grid.get_cell(input_r, input_c)
            if color != 0:
                if r % 3 == 1 and c % 3 == 1:
                    output_grid.set_cell(r, c, color)
                elif color not in [1, 3] and (r % 3 == 1 or c % 3 == 1):
                    output_grid.set_cell(r, c, 1)

    # Apply global blue grid
    for r in range(output_rows):
        for c in range(output_cols):
            if (r % 3 == 0 or c % 3 == 0) and output_grid.get_cell(r, c) == 0:
                output_grid.set_cell(r, c, 1)

    # Apply green corners in checkerboard pattern
    for r in range(0, output_rows, 3):
        for c in range(0, output_cols, 3):
            if ((r // 3 + c // 3) % 2 == 0):
                for corner_r, corner_c in [(r, c), (r, c+2), (r+2, c), (r+2, c+2)]:
                    if output_grid.get_cell(corner_r, corner_c) == 0:
                        output_grid.set_cell(corner_r, corner_c, 3)

    return output_grid
