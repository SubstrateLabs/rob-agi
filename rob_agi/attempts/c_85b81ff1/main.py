from rob_agi.colored_grid import ColoredGrid

def solve_85b81ff1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Create a deep copy of the input grid.
    2. For each non-black column:
       a. Keep the top row unchanged.
       b. Set the second row to black (0).
       c. For rows 3 to second-to-last:
          - If it's an odd-numbered row:
            * If the cell was colored in the input, keep it colored.
            * If it was black in the input, fill it with the column's color if the cell above in the output grid is colored.
          - If it's an even-numbered row:
            * If both the cell above AND the cell below are colored, fill this cell with the column's color.
            * Otherwise, keep it black (0).
       d. Keep the bottom row unchanged.
    3. Return the modified grid.

    This approach maintains the vertical line structure while strategically connecting them based on the surrounding cells.
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
            current_cell = input_grid.get_cell(row, col)
            cell_above = output_grid.get_cell(row - 1, col)
            cell_below = input_grid.get_cell(row + 1, col)

            if row % 2 == 0:  # Odd-numbered row (0-based index)
                if current_cell != 0:
                    output_grid.set_cell(row, col, current_cell)
                elif cell_above != 0:
                    output_grid.set_cell(row, col, fill_color)
            else:  # Even-numbered row (0-based index)
                if cell_above != 0 and cell_below != 0:
                    output_grid.set_cell(row, col, fill_color)

    return output_grid
