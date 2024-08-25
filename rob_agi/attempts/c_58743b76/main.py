from rob_agi.colored_grid import ColoredGrid

def solve_58743b76(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a color sequence to specific target colors.
    
    The function identifies a target color (1 or 2) in the non-border area,
    and replaces them with a repeating color sequence.
    For target color 1 (blue), the sequence is [4, 2, 3] (yellow, red, green).
    For target color 2 (red), the sequence is [4, 6, 1, 2] (yellow, magenta, blue, red).
    The border (first/last row and column) remains unchanged.
    The transformation continues until all target colors are replaced,
    with the sequence wrapping around if necessary.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Define transformation sequences
    sequences = {
        1: [4, 2, 3],
        2: [4, 6, 1, 2]
    }

    # Identify target color
    target_color = None
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if input_grid.values[r][c] in [1, 2]:
                target_color = input_grid.values[r][c]
                break
        if target_color:
            break

    if target_color is None:
        return input_grid  # Return original grid if no target color found

    # Create output grid and apply transformation
    output_grid = input_grid.deep_copy()
    sequence = sequences[target_color]
    seq_index = 0

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if output_grid.values[r][c] == target_color:
                output_grid.values[r][c] = sequence[seq_index]
                seq_index = (seq_index + 1) % len(sequence)

    return output_grid
