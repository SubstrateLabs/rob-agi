from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_f823c43c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the background color and the pattern color,
    then creates a new grid with a regular pattern based on these colors.
    
    1. Analyzes the input grid to find the background color (most common) and pattern color (second most common, excluding 6).
    2. Creates a new grid filled with the background color.
    3. Applies the pattern color in a regular grid:
       - On every other row, starting from the second row (index 1 if 0-indexed)
       - In every other column, starting from the second column (index 1 if 0-indexed)
    4. Returns the new grid as a ColoredGrid object.
    """
    # Analyze the input grid
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell != 6)
    background_color = color_counts.most_common(1)[0][0]
    
    # Find the pattern color (second most common, excluding 6)
    pattern_color = next(color for color, _ in color_counts.most_common() if color != background_color)

    # Create a new grid
    rows, cols = input_grid.get_dimensions()
    new_grid = [[background_color for _ in range(cols)] for _ in range(rows)]

    # Apply the pattern color
    for row in range(1, rows, 2):
        for col in range(1, cols, 2):
            new_grid[row][col] = pattern_color

    # Convert to ColoredGrid and return
    return ColoredGrid(values=new_grid)
