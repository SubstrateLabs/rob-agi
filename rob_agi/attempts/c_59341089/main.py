from rob_agi.colored_grid import ColoredGrid

def solve_59341089(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 3x12 output grid by repeating and mirroring the input pattern.
    The output consists of four 3x3 blocks:
    1. The original input grid.
    2. A mirrored version of the input grid (horizontally flipped).
    3. Repeat of the original input grid.
    4. Another mirrored version of the input grid.

    Args:
        input_grid (ColoredGrid): A 3x3 input grid

    Returns:
        ColoredGrid: A 3x12 output grid
    """
    input_values = input_grid.values
    if len(input_values) != 3 or len(input_values[0]) != 3:
        raise ValueError("Input grid must be 3x3")

    output = [[0 for _ in range(12)] for _ in range(3)]

    # Function to mirror a 3x3 grid horizontally
    def mirror_grid(grid):
        return [row[::-1] for row in grid]

    # Fill the output grid
    for block in range(4):
        start_col = block * 3
        source = input_values if block % 2 == 0 else mirror_grid(input_values)
        for row in range(3):
            output[row][start_col:start_col+3] = source[row]

    return ColoredGrid(values=output)
