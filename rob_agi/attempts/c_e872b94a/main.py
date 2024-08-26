from rob_agi.colored_grid import ColoredGrid

def solve_e872b94a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an output grid by counting the number of separated
    levels of gray cells (value 5) and creating a single-column grid with that
    many black cells (value 0).

    The function works as follows:
    1. Scan the input grid from top to bottom, row by row.
    2. Count a new level when a row with gray cells is found after a row without gray cells.
    3. Create a new grid with a single column and a height equal to the number of separated levels.
    4. Fill the new grid with black cells (value 0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: A new grid with a single column of black cells, where the height
                 represents the number of separated levels of gray cells in the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    level_count = 0
    previous_row_had_gray = False

    for r in range(rows):
        current_row_has_gray = False
        for c in range(cols):
            if input_grid.values[r][c] == 5:
                current_row_has_gray = True
                break
        
        if current_row_has_gray and not previous_row_had_gray:
            level_count += 1
        
        previous_row_had_gray = current_row_has_gray

    output_values = [[0] for _ in range(level_count)]
    return ColoredGrid(values=output_values)
