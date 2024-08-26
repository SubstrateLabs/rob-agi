from rob_agi.colored_grid import ColoredGrid

def solve_85b81ff1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Create a deep copy of the input grid.
    2. For each non-black column:
       a. Keep the top row unchanged.
       b. Set the second row to black (0).
       c. For rows 3 to second-to-last:
          - If the cell is non-black in the input, keep it unchanged.
          - If the cell is black in the input:
            * If the cell above is non-black, set it to the column's color.
            * If the cell above is black, keep it black.
       d. Keep the bottom row unchanged.
    3. Return the modified grid.

    This approach maintains the structure at the top and bottom while
    redistributing black cells in a consistent pattern based on the cells above them.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for col in range(cols):
        fill_color = input_grid.get_cell(0, col)
        
        if fill_color == 0:
            continue  # Skip entirely black columns

        # Set second row to black
        output_grid.set_cell(1, col, 0)

        # Process middle rows
        for row in range(2, rows - 1):
            if input_grid.get_cell(row, col) == 0:
                if output_grid.get_cell(row - 1, col) != 0:
                    output_grid.set_cell(row, col, fill_color)

    return output_grid
