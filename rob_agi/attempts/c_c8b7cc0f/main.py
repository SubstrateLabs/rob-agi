from rob_agi.colored_grid import ColoredGrid

def solve_c8b7cc0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x3 output grid based on the following rules:
    1. Ignores black (0) and blue (1) colors.
    2. Finds the most frequent non-black, non-blue color in the input grid.
    3. Creates a 3x3 output grid with the top-left cells filled with the most frequent color.
    4. The number of filled cells is equal to the count of the most frequent color, up to a maximum of 5.
    5. If no valid colors are found, returns an all-black 3x3 grid.
    """
    # Analyze the input grid
    color_counts = {}
    for row in input_grid.values:
        for cell in row:
            if cell != 0 and cell != 1:  # Ignore black (0) and blue (1)
                color_counts[cell] = color_counts.get(cell, 0) + 1

    # Find the target color
    if not color_counts:
        return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    target_color = max(color_counts, key=color_counts.get)

    # Count occurrences of target color
    target_count = min(sum(row.count(target_color) for row in input_grid.values), 5)

    # Create the output grid
    output_values = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(target_count):
        output_values[i // 3][i % 3] = target_color

    # Return the output grid
    return ColoredGrid(values=output_values)
