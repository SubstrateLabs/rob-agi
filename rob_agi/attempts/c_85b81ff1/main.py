from rob_agi.colored_grid import ColoredGrid

def solve_85b81ff1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Create a deep copy of the input grid.
    2. For each non-black column:
       a. Keep the top row unchanged.
       b. For the second row:
          - If it was originally colored and the cell below is black, set it to black (0).
          - Otherwise, keep its original color.
       c. For rows 3 to second-to-last:
          - If the cell was originally colored, keep it colored.
          - If the cell was originally black (0):
            * If it's between two colored cells in the same column, fill it with the column's color.
            * If it's below a colored cell and there's no colored cell immediately below it, fill it with the column's color.
            * Otherwise, keep it black (0).
       d. Keep the bottom row unchanged.
    3. Return the modified grid.

    This approach maintains the vertical line structure while strategically connecting them based on the surrounding cells.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for col in range(cols):
        fill_color = next((input_grid.get_cell(r, col) for r in range(rows) if input_grid.get_cell(r, col) != 0), 0)
        
        if fill_color == 0:
            continue  # Skip entirely black columns

        # Process second row
        if input_grid.get_cell(1, col) != 0 and input_grid.get_cell(2, col) == 0:
            output_grid.set_cell(1, col, 0)

        # Process middle rows
        for row in range(2, rows - 1):
            current_cell = input_grid.get_cell(row, col)
            cell_above = output_grid.get_cell(row - 1, col)
            cell_below = input_grid.get_cell(row + 1, col)

            if current_cell != 0:
                output_grid.set_cell(row, col, current_cell)
            elif current_cell == 0:
                if cell_above != 0 and cell_below != 0:
                    output_grid.set_cell(row, col, fill_color)
                elif cell_above != 0 and cell_below == 0:
                    output_grid.set_cell(row, col, fill_color)

    return output_grid
