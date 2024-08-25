from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_2753e76c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a compact representation of the most frequent colors.
    
    The solution follows these steps:
    1. Count the frequency of each non-black color in the input grid.
    2. Select the top 4 most frequent colors (or fewer if there aren't 4).
    3. Calculate the relative frequency of each color.
    4. Create an output grid with 4 rows and width equal to the sum of relative frequencies.
    5. Fill each row from right to left with the corresponding color.
    6. Align colors to the right in each row, filling unused cells with black (0).
    
    This approach captures the essence of the input grid by showing the relative
    frequencies of the most prominent colors in a compact form.
    """
    # Count color frequencies (excluding black)
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell != 0)

    # Select top colors (up to 4)
    top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:4]

    if not top_colors:
        return ColoredGrid(values=[[0]])

    # Calculate relative frequencies
    max_count = max(color_counts[color] for color in top_colors)
    relative_frequencies = {color: round((count / max_count) * 4) for color, count in color_counts.items() if color in top_colors}

    # Determine output grid width
    width = sum(relative_frequencies.values())

    # Create the output grid
    output_grid = [[0 for _ in range(width)] for _ in range(4)]

    # Fill the grid
    current_col = width
    for row, color in enumerate(top_colors):
        cells_to_fill = relative_frequencies[color]
        start = max(0, current_col - cells_to_fill)
        output_grid[row][start:current_col] = [color] * (current_col - start)
        current_col = start

    return ColoredGrid(values=output_grid)
