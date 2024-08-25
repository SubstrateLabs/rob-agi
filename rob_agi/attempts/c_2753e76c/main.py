from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_2753e76c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a compact representation of the most frequent colors.
    
    The solution follows these steps:
    1. Count the frequency of each non-black color in the input grid.
    2. Select the top 4 most frequent colors (or fewer if there aren't 4).
    3. Create an output grid where each row represents one of the top colors.
    4. The width of each color in its row is proportional to its frequency.
    5. The output grid is 5x4 if there are 4 or more colors, otherwise 3x3.
    
    This approach captures the essence of the input grid by showing the relative
    frequencies of the most prominent colors in a compact form.
    """
    # Count color frequencies (excluding black)
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell != 0)

    # Select top colors (up to 4)
    top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:4]

    # Determine output grid size
    grid_height = 5 if len(top_colors) >= 4 else 3
    grid_width = 4 if len(top_colors) >= 4 else 3

    # Calculate color proportions
    max_count = color_counts[top_colors[0]]
    color_cells = [int((color_counts[color] / max_count) * grid_width) for color in top_colors]

    # Create and fill the output grid
    output_grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    for i, color in enumerate(top_colors):
        output_grid[i][:color_cells[i]] = [color] * color_cells[i]

    return ColoredGrid(values=output_grid)
