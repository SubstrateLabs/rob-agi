from rob_agi.colored_grid import ColoredGrid

def solve_15696249(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid based on the following rules:
    1. Check if the input grid can be represented by a single repeated row.
    2. Create a new 9x9 grid filled with zeros (black).
    3. If the input can be represented by a single row:
       - Apply horizontal repetition to the first 3 rows.
    4. If not:
       - Apply vertical repetition to the first 3 columns.
    5. Return the resulting 9x9 grid.
    """
    def can_be_represented_by_single_row(grid):
        return all(row == grid.values[0] for row in grid.values)

    # Create new 9x9 grid
    result = ColoredGrid(values=[[0 for _ in range(9)] for _ in range(9)])

    if can_be_represented_by_single_row(input_grid):
        # Horizontal repetition
        for r in range(3):
            result.values[r] = input_grid.values[0] * 3
    else:
        # Vertical repetition
        for r in range(9):
            for c in range(3):
                result.values[r][c] = input_grid.values[r % 3][c]

    return result
