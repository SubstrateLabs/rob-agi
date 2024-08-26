from rob_agi.colored_grid import ColoredGrid

def solve_85b81ff1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Create a deep copy of the input grid.
    2. For each non-black column:
       a. Set the second row from the top to black (0).
       b. If the second row from the bottom is black, set it to the column's color.
    3. Return the modified grid.

    This approach maintains the structure at the top and bottom while
    redistributing black cells in a consistent pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for col in range(cols):
        # Skip all-black columns
        if all(input_grid.get_cell(row, col) == 0 for row in range(rows)):
            continue

        # Get the color of the column (from the top or bottom row)
        column_color = input_grid.get_cell(0, col) or input_grid.get_cell(rows-1, col)

        # Set the second row from the top to black
        output_grid.set_cell(1, col, 0)

        # Set the second row from the bottom to the column color if it's black
        if input_grid.get_cell(rows-2, col) == 0:
            output_grid.set_cell(rows-2, col, column_color)

    return output_grid
