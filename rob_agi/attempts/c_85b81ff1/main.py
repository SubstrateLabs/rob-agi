from rob_agi.colored_grid import ColoredGrid

def solve_85b81ff1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Create a deep copy of the input grid.
    2. For each non-black column:
       a. Set the second row from the top to black (0).
       b. If the second row from the bottom is black in the input, set it to the column's color.
       c. Leave the top and bottom rows unchanged.
    3. Return the modified grid.

    This approach maintains the structure at the top and bottom while
    redistributing black cells in a consistent pattern, ensuring that isolated
    color cells are moved to the bottom of gaps when possible.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for col in range(cols):
        column_color = input_grid.get_cell(0, col) or input_grid.get_cell(rows-1, col)
        
        if column_color == 0:
            continue  # Skip entirely black columns

        # Set second row to black
        output_grid.set_cell(1, col, 0)

        # Check and potentially modify second-to-last row
        if input_grid.get_cell(rows-2, col) == 0:
            output_grid.set_cell(rows-2, col, column_color)

    return output_grid
