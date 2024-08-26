from rob_agi.colored_grid import ColoredGrid

def solve_59341089(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 3x12 output grid by repeating a specific pattern four times horizontally.
    The pattern for each 3x3 block in the output is:
    1. The first column is the last column of the input grid.
    2. The second column is the first column of the input grid.
    3. The third column is the second column of the input grid.
    This pattern is repeated four times, with a slight modification for the first and last columns of the entire output.

    Args:
        input_grid (ColoredGrid): A 3x3 input grid

    Returns:
        ColoredGrid: A 3x12 output grid
    """
    input_values = input_grid.values
    if len(input_values) != 3 or len(input_values[0]) != 3:
        raise ValueError("Input grid must be 3x3")

    output = [[0 for _ in range(12)] for _ in range(3)]
    first_column = [input_values[r][0] for r in range(3)]
    second_column = [input_values[r][1] for r in range(3)]
    last_column = [input_values[r][2] for r in range(3)]

    for block in range(4):
        start_col = block * 3
        for row in range(3):
            if block == 0:
                output[row][start_col] = last_column[row]
            else:
                output[row][start_col] = output[row][start_col - 1]
            output[row][start_col + 1] = first_column[row]
            output[row][start_col + 2] = second_column[row]

    return ColoredGrid(values=output)
