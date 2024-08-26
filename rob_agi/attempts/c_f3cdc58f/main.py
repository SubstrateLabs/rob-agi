from rob_agi.colored_grid import ColoredGrid

def solve_f3cdc58f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern of colored columns in the bottom-left corner.
    
    The solution follows these steps:
    1. Count the occurrences of colors 1 (blue), 2 (red), 3 (green), and 4 (yellow) in the input grid.
    2. Calculate initial column heights based on color proportions and grid height.
    3. Adjust column heights to fit within the grid while maintaining relative proportions.
    4. Determine the position of each column, creating a "staircase" pattern.
    5. Create the new grid with the calculated column positions and heights.
    6. The rest of the grid is filled with 0 (black/empty).

    This approach creates a stair-step pattern of colored columns while maintaining
    the relative proportions of colors from the input grid.
    """
    height, width = input_grid.get_dimensions()
    
    # Step 1: Count color occurrences
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for row in input_grid.values:
        for color in row:
            if color in color_counts:
                color_counts[color] += 1
    
    # Step 2: Calculate initial column heights
    total_colored = sum(color_counts.values())
    if total_colored > 0:
        initial_heights = {color: int((count / total_colored) * height) for color, count in color_counts.items()}
    else:
        initial_heights = {1: 0, 2: 0, 3: 0, 4: 0}
    
    # Step 3: Adjust column heights
    max_height = max(initial_heights.values())
    if max_height > height:
        scale_factor = height / max_height
        heights = {color: max(1, int(h * scale_factor)) if count > 0 else 0 
                   for color, h, count in zip(initial_heights.keys(), initial_heights.values(), color_counts.values())}
    else:
        heights = initial_heights
    
    # Step 4: Determine column positions
    positions = {4: (height - heights[4], height)}  # Yellow starts at the bottom
    for color in [3, 2, 1]:  # Green, Red, Blue
        bottom = max(height - heights[color], positions[color + 1][0])
        top = bottom + heights[color]
        positions[color] = (bottom, top)
    
    # Step 5: Create the new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    for col, color in enumerate([1, 2, 3, 4]):
        bottom, top = positions[color]
        for row in range(bottom, top):
            new_grid[row][col] = color
    
    # Step 6: Return the new grid
    return ColoredGrid(values=new_grid)
