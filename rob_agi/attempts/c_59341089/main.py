from rob_agi.colored_grid import ColoredGrid

def solve_59341089(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 3x12 output grid by repeating and mirroring the input pattern.
    The output consists of four 3x3 blocks, where:
    1. The top row is repeated 4 times.
    2. The middle row alternates between the original and a horizontally flipped version.
    3. The bottom row alternates between a horizontally flipped version and the original, repeated twice.

    Args:
        input_grid (ColoredGrid): A 3x3 input grid

    Returns:
        ColoredGrid: A 3x12 output grid
    """
    input_values = input_grid.values
    if len(input_values) != 3 or len(input_values[0]) != 3:
        raise ValueError("Input grid must be 3x3")

    output = [[0 for _ in range(12)] for _ in range(3)]

    # Fill the top row (repeat 4 times)
    output[0] = input_values[0] * 4

    # Fill the middle row (alternating original and flipped)
    middle_original = input_values[1]
    middle_flipped = middle_original[::-1]
    output[1] = middle_original + middle_flipped + middle_original + middle_flipped

    # Fill the bottom row (alternating flipped and original, repeated twice)
    bottom_original = input_values[2]
    bottom_flipped = bottom_original[::-1]
    output[2] = (bottom_flipped + bottom_original) * 2

    return ColoredGrid(values=output)
