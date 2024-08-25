from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_b94a9452(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying a pattern and creating a new grid with inverted colors.
    
    The solution follows these steps:
    1. Find the bounding box of the non-zero region in the input grid.
    2. Determine the output grid size (square, based on the larger dimension of the pattern).
    3. Identify the two most frequent colors in the pattern.
    4. Create a new grid with the more frequent color as the background.
    5. Transfer the pattern to the new grid, inverting the colors.
    6. Center the pattern in the output grid if it's smaller than the output size.
    
    Edge cases:
    - If the input is empty, return a 1x1 grid with color 0.
    - If there's only one color in the pattern, use that color for the entire output grid.
    """
    grid = input_grid.values
    height, width = len(grid), len(grid[0])

    # Find the bounding box of the non-zero region
    min_row, max_row, min_col, max_col = height, -1, width, -1
    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0:
                min_row, max_row = min(min_row, r), max(max_row, r)
                min_col, max_col = min(min_col, c), max(max_col, c)

    # Handle empty input
    if min_row > max_row or min_col > max_col:
        return ColoredGrid(values=[[0]])

    # Determine the output size
    pattern_height, pattern_width = max_row - min_row + 1, max_col - min_col + 1
    output_size = max(pattern_height, pattern_width)

    # Identify colors used in the pattern
    color_freq = Counter(grid[r][c] for r in range(min_row, max_row + 1) 
                         for c in range(min_col, max_col + 1) if grid[r][c] != 0)
    
    # Handle single-color patterns
    if len(color_freq) == 1:
        pattern_color = list(color_freq.keys())[0]
        return ColoredGrid(values=[[pattern_color] * output_size for _ in range(output_size)])

    # Determine outer and inner colors
    outer_color, inner_color = color_freq.most_common(2)
    outer_color, inner_color = outer_color[0], inner_color[0]

    # Create the output grid with the outer color
    output = [[outer_color] * output_size for _ in range(output_size)]

    # Transfer the pattern with inverted colors
    start_row = (output_size - pattern_height) // 2
    start_col = (output_size - pattern_width) // 2
    for r in range(pattern_height):
        for c in range(pattern_width):
            if grid[min_row + r][min_col + c] == outer_color:
                output[start_row + r][start_col + c] = inner_color

    return ColoredGrid(values=output)
