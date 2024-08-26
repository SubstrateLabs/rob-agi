from rob_agi.colored_grid import ColoredGrid

def solve_59341089(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 3x12 output grid by repeating the input pattern
    four times horizontally, with the first column of each repetition being replaced
    by the last column of the input grid.

    Args:
        input_grid (ColoredGrid): A 3x3 input grid

    Returns:
        ColoredGrid: A 3x12 output grid
    """
    input_values = input_grid.values
    rows, cols = len(input_values), len(input_values[0])
    
    if rows != 3 or cols != 3:
        raise ValueError("Input grid must be 3x3")

    output = [[0 for _ in range(12)] for _ in range(3)]
    last_column = [input_values[r][2] for r in range(3)]

    for block in range(4):
        start_col = block * 3
        for row in range(3):
            output[row][start_col] = last_column[row]
            for col in range(3):
                output[row][start_col + 1 + col] = input_values[row][col]

    return ColoredGrid(values=output)
