from rob_agi.colored_grid import ColoredGrid

def solve_e872b94a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an output grid by counting the number of distinct
    levels of gray cells (value 5) and creating a single-column grid with that
    many black cells (value 0).

    The function works as follows:
    1. Scan the input grid from top to bottom, tracking the highest gray cell in each column.
    2. Count a new level when a gray cell appears at a height not seen before in any column.
    3. Remove the bottom level if no gray cells were found in a column.
    4. Create a new grid with a single column and a height equal to the number of distinct levels.
    5. Fill the new grid with black cells (value 0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: A new grid with a single column of black cells, where the height
                 represents the number of distinct levels of gray cells in the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    highest_gray = [rows] * cols  # Initialize with bottom of grid
    distinct_levels = set()

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5 and r < highest_gray[c]:
                highest_gray[c] = r
                distinct_levels.add(r)

    # Remove the bottom level if it's still present (represents columns with no gray cells)
    distinct_levels.discard(rows)

    level_count = len(distinct_levels)
    output_values = [[0] for _ in range(level_count)]
    return ColoredGrid(values=output_values)
