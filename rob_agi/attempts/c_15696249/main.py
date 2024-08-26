from rob_agi.colored_grid import ColoredGrid

def solve_15696249(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid based on the following rules:
    1. Create horizontal and vertical repetition candidates.
    2. Compare candidates with the input grid to determine the repetition direction.
    3. Create a new 9x9 grid filled with zeros (black).
    4. Apply the chosen repetition:
       - For horizontal repetition: Copy the first 3 rows of the horizontal candidate.
       - For vertical repetition: Copy the first 3 columns of the vertical candidate.
    5. Return the resulting 9x9 grid.
    """
    # Create horizontal and vertical candidates
    horizontal_candidate = [row * 3 for row in input_grid.values]
    vertical_candidate = input_grid.values * 3

    # Check for matches
    horizontal_match = any(row in horizontal_candidate for row in input_grid.values)
    vertical_match = any(col in zip(*vertical_candidate) for col in zip(*input_grid.values))

    # Determine repetition direction
    if horizontal_match and not vertical_match:
        horizontal_repetition = True
    elif vertical_match and not horizontal_match:
        horizontal_repetition = False
    else:
        # If both match or neither match, choose based on unique lines
        unique_rows = len(set(map(tuple, input_grid.values)))
        unique_cols = len(set(zip(*input_grid.values)))
        horizontal_repetition = unique_rows <= unique_cols

    # Create new 9x9 grid
    result = ColoredGrid(values=[[0 for _ in range(9)] for _ in range(9)])

    # Apply repetition
    if horizontal_repetition:
        for r in range(3):
            result.values[r] = horizontal_candidate[r]
    else:
        for r in range(9):
            for c in range(3):
                result.values[r][c] = vertical_candidate[r][c]

    return result
