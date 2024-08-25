from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict

def solve_9af7a82c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an output grid based on color frequencies and first appearances.
    
    The function:
    1. Analyzes the input grid to count color frequencies and record their first appearances.
    2. Determines the output grid dimensions based on the number of unique colors and the maximum frequency.
    3. Sorts the unique colors by frequency (descending) and then by first appearance.
    4. Initializes the output grid with the determined dimensions.
    5. Fills the output grid with colors, placing each color in its respective column according to its frequency.
    6. Returns the completed output grid as a ColoredGrid object.
    """
    # Step 1: Analyze the input grid
    color_count = defaultdict(int)
    first_appearance = {}
    
    for r, row in enumerate(input_grid.values):
        for c, color in enumerate(row):
            if color != 0:  # Ignore black (empty space)
                color_count[color] += 1
                if color not in first_appearance:
                    first_appearance[color] = (r, c)
    
    # Step 2: Determine output grid dimensions
    unique_colors = list(color_count.keys())
    width = len(unique_colors)
    height = max(color_count.values())
    
    # Step 3: Sort the unique colors
    sorted_colors = sorted(unique_colors, key=lambda color: (-color_count[color], first_appearance[color]))
    
    # Step 4 & 5: Initialize and fill the output grid
    output_grid = [[0] * width for _ in range(height)]
    for col, color in enumerate(sorted_colors):
        for row in range(color_count[color]):
            output_grid[row][col] = color
    
    # Step 6: Return the completed output grid
    return ColoredGrid(values=output_grid)
