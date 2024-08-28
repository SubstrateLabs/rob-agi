from rob_agi.colored_grid import ColoredGrid

def solve_59341089(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 3x12 output grid by repeating and mirroring the input pattern.
    The output consists of four 3x3 blocks, where:
    1. The top row is repeated 4 times.
    2. The middle row follows a specific pattern based on its input:
       - If all elements are the same, it alternates between the original and its mirror.
       - If elements differ, it follows the pattern: [last, first, second, second, first, last] repeated twice.
    3. The bottom row alternates between the original and a horizontally flipped version, repeated twice.

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

    # Fill the middle row
    middle = input_values[1]
    if len(set(middle)) == 1:  # All elements are the same
        output[1] = middle + middle[::-1] + middle + middle[::-1]
    else:
        first, second, last = middle
        pattern = [last, first, second, second, first, last]
        output[1] = pattern * 2

    # Fill the bottom row (alternating original and flipped, repeated twice)
    bottom_original = input_values[2]
    bottom_flipped = bottom_original[::-1]
    output[2] = (bottom_original + bottom_flipped) * 2

    return ColoredGrid(values=output)
