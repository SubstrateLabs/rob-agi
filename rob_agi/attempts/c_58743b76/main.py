from rob_agi.colored_grid import ColoredGrid

def solve_58743b76(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a color sequence to specific target colors.
    
    The function identifies a target color (1 or 2) in the non-border area,
    counts its occurrences, and replaces them with a repeating color sequence.
    For target color 1 (blue), the sequence is [4, 2, 3] (yellow, red, green).
    For target color 2 (red), the sequence is [4, 6, 1, 2] (yellow, magenta, blue, red).
    The border, special corner elements, and any other colors remain unchanged.
    """
    rows, cols = input_grid.get_dimensions()
    border_color = input_grid.values[0][0]
    border_width = 2  # Changed to 2 to preserve the special corner

    # Identify target color and count occurrences
    target_color = None
    target_count = 0
    for r in range(border_width, rows - border_width):
        for c in range(border_width, cols - border_width):
            if input_grid.values[r][c] in [1, 2]:
                if target_color is None:
                    target_color = input_grid.values[r][c]
                if input_grid.values[r][c] == target_color:
                    target_count += 1

    # Generate transformation sequence
    if target_color == 1:
        base_sequence = [4, 2, 3]
    elif target_color == 2:
        base_sequence = [4, 6, 1, 2]
    else:
        return input_grid  # Return original grid if no target color found
    
    transform_sequence = (base_sequence * (target_count // len(base_sequence) + 1))[:target_count]

    # Create output grid and apply transformation
    output_grid = input_grid.deep_copy()
    seq_index = 0
    for r in range(border_width, rows - border_width):
        for c in range(border_width, cols - border_width):
            if output_grid.values[r][c] == target_color:
                output_grid.values[r][c] = transform_sequence[seq_index]
                seq_index += 1

    return output_grid
