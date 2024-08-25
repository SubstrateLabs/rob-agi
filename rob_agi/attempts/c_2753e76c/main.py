from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_2753e76c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a compact representation of the most frequent colors.
    
    The solution follows these steps:
    1. Count the frequency of each non-black color in the input grid.
    2. Select the top 4 most frequent colors (or fewer if there aren't 4).
    3. Determine the width of the output grid based on the most frequent color.
    4. Create an output grid with 4 rows and the determined width.
    5. Fill the grid from bottom to top, with more frequent colors occupying more cells.
    6. Align colors to the right in each row, filling unused cells with black (0).
    
    This approach captures the essence of the input grid by showing the relative
    frequencies of the most prominent colors in a compact form.
    """
    # Count color frequencies (excluding black)
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell != 0)

    # Select top colors (up to 4)
    top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:4]

    # Determine output grid width based on the most frequent color
    grid_width = color_counts[top_colors[0]]
    grid_height = 4

    # Calculate color proportions
    total_count = sum(color_counts[color] for color in top_colors)
    color_proportions = {color: max(1, int((color_counts[color] / total_count) * grid_width)) for color in top_colors}

    # Create the output grid
    output_grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]

    # Fill the grid
    for i, color in enumerate(reversed(top_colors)):
        cells_to_fill = color_proportions[color]
        output_grid[grid_height - 1 - i][-cells_to_fill:] = [color] * cells_to_fill

    return ColoredGrid(values=output_grid)
