from rob_agi.colored_grid import ColoredGrid

def solve_f3cdc58f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern of colored columns in the bottom-left corner.
    
    The solution follows these steps:
    1. Count the occurrences of colors 1 (blue), 2 (red), 3 (green), and 4 (yellow) in the input grid.
    2. Calculate column heights based on color proportions and grid height.
    3. Determine the position of each column, creating a "staircase" pattern.
    4. Create the new grid with the calculated column positions and heights.
    5. The rest of the grid is filled with 0 (black/empty).

    This approach creates a stair-step pattern of colored columns in the order blue, red, green, yellow
    from left to right, while maintaining the relative proportions of colors from the input grid.
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
    
    # Step 2: Calculate column heights
    max_height = height - 1  # Leave room for rounding
    heights = {color: max(1, int((count / total_colored) * max_height)) for color, count in color_counts.items()}
    
    # Adjust heights if they exceed the grid height
    while sum(heights.values()) > height:
        max_color = max(heights, key=heights.get)
        heights[max_color] -= 1
    
    # Step 3: Determine column positions
    positions = {}
    current_bottom = height
    for color in [4, 3, 2, 1]:  # Yellow, Green, Red, Blue
        top = current_bottom
        bottom = max(top - heights[color], 0)
        positions[color] = (bottom, top)
        current_bottom = bottom
    
    # Step 4: Create the new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    for col, color in enumerate([1, 2, 3, 4]):
        bottom, top = positions[color]
        for row in range(bottom, top):
            new_grid[row][col] = color
    
    # Step 5: Return the new grid
    return ColoredGrid(values=new_grid)
