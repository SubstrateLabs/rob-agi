from rob_agi.colored_grid import ColoredGrid

def solve_15696249(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid based on the following rules:
    1. Compare the first and last rows of the input grid.
    2. Create a new 9x9 grid filled with zeros (black).
    3. If the first and last rows are different:
       - Apply horizontal repetition to all 3 rows, repeating each 3 times.
    4. If the first and last rows are the same:
       - Apply vertical repetition to all 3 columns, repeating each 3 times.
    5. Return the resulting 9x9 grid.
    """
    # Create new 9x9 grid
    result = ColoredGrid(values=[[0 for _ in range(9)] for _ in range(9)])

    # Determine repetition direction
    horizontal_repetition = input_grid.values[0] != input_grid.values[2]

    if horizontal_repetition:
        # Horizontal repetition
        for r in range(3):
            result.values[r] = input_grid.values[r] * 3
    else:
        # Vertical repetition
        for r in range(9):
            for c in range(3):
                result.values[r][c] = input_grid.values[r % 3][c]

    return result
