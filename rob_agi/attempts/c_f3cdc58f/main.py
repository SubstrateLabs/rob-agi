from rob_agi.colored_grid import ColoredGrid

def solve_f3cdc58f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern of colored columns in the bottom-left corner.
    
    The solution follows these steps:
    1. Analyze the input grid to count occurrences of colors 1 (blue), 2 (red), 3 (green), and 4 (yellow).
    2. Calculate initial column heights based on color proportions.
    3. Adjust column heights to ensure they fit within the grid and maintain minimum visibility.
    4. Create the new grid with colored columns in the order blue, red, green, yellow from left to right.
    5. Fill the columns from bottom to top, leaving the rest of the grid black (0).

    This approach creates a pattern of colored columns that represent the relative proportions
    of colors from the input grid, while ensuring all present colors are visible.
    """
    height, width = input_grid.get_dimensions()
    
    # Step 1: Count color occurrences
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for row in input_grid.values:
        for color in row:
            if color in color_counts:
                color_counts[color] += 1
    
    total_colored = sum(color_counts.values())
    if total_colored == 0:
        return ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Step 2: Calculate initial column heights
    heights = {color: max(1, int((count / total_colored) * height)) for color, count in color_counts.items()}
    
    # Step 3: Adjust column heights
    while sum(heights.values()) > height:
        tallest = max(heights, key=heights.get)
        if heights[tallest] > 1:
            heights[tallest] -= 1
        else:
            # If we can't reduce further, break to avoid infinite loop
            break
    
    # Ensure all present colors have at least one row
    for color, count in color_counts.items():
        if count > 0 and heights[color] == 0:
            heights[color] = 1
            # Reduce the tallest column to make room
            tallest = max(heights, key=heights.get)
            if heights[tallest] > 1:
                heights[tallest] -= 1
    
    # Step 4 & 5: Create the new grid and fill the columns
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    current_row = height - 1
    for color in [1, 2, 3, 4]:  # Blue, Red, Green, Yellow
        for _ in range(heights[color]):
            if current_row >= 0:
                new_grid[current_row][color - 1] = color
                current_row -= 1
    
    return ColoredGrid(values=new_grid)
