from rob_agi.colored_grid import ColoredGrid

def solve_e872b94a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an output grid by counting the number of rows
    containing gray cells (value 5) and creating a single-column grid with that
    many black cells (value 0).

    The function works as follows:
    1. Count the number of rows in the input grid that contain at least one gray cell.
    2. Create a new grid with a single column and a height equal to the count from step 1.
    3. Fill the new grid with black cells (value 0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: A new grid with a single column of black cells, where the height
                 represents the number of rows with gray cells in the input grid.
    """
    level_count = 0
    rows, cols = input_grid.get_dimensions()
    
    for row in range(rows):
        has_gray = False
        for col in range(cols):
            if input_grid.values[row][col] == 5:  # 5 represents gray
                has_gray = True
                break
        if has_gray:
            level_count += 1
    
    output_values = [[0] for _ in range(level_count)]
    output_grid = ColoredGrid(values=output_values)
    
    return output_grid
