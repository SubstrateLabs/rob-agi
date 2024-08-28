from rob_agi.colored_grid import ColoredGrid

def solve_e872b94a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an output grid by counting the number of distinct
    vertical regions of gray cells (value 5) and creating a single-column grid with
    that many black cells (value 0).

    The function works as follows:
    1. Scan the input grid from top to bottom, column by column.
    2. For each column, count a new level when a gray cell is found after a non-gray cell.
    3. Take the maximum number of levels found across all columns.
    4. Create a new grid with a single column and a height equal to the maximum number of levels.
    5. Fill the new grid with black cells (value 0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: A new grid with a single column of black cells, where the height
                 represents the maximum number of distinct vertical regions of gray cells
                 found in any column of the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    max_level_count = 0

    for c in range(cols):
        level_count = 0
        previous_cell_was_gray = False
        for r in range(rows):
            current_cell_is_gray = input_grid.values[r][c] == 5
            if current_cell_is_gray and not previous_cell_was_gray:
                level_count += 1
            previous_cell_was_gray = current_cell_is_gray
        max_level_count = max(max_level_count, level_count)

    output_values = [[0] for _ in range(max_level_count)]
    return ColoredGrid(values=output_values)
