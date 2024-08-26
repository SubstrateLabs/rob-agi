from rob_agi.colored_grid import ColoredGrid

def solve_e872b94a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an output grid by counting the number of distinct
    levels of gray cells (value 5) and creating a single-column grid with that
    many black cells (value 0).

    The function works as follows:
    1. Scan the input grid from top to bottom, tracking the highest gray cell in each column.
    2. Count a new level when a gray cell appears higher than any previous gray cell in its column.
    3. Create a new grid with a single column and a height equal to the number of distinct levels.
    4. Fill the new grid with black cells (value 0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: A new grid with a single column of black cells, where the height
                 represents the number of distinct levels of gray cells in the input grid.
    """
    level_count = 0
    rows, cols = input_grid.get_dimensions()
    highest_gray = [-1] * cols
    
    for row in range(rows):
        new_level = False
        for col in range(cols):
            if input_grid.values[row][col] == 5 and row > highest_gray[col]:
                new_level = True
                highest_gray[col] = row
        if new_level:
            level_count += 1
    
    output_values = [[0] for _ in range(level_count)]
    output_grid = ColoredGrid(values=output_values)
    
    return output_grid
