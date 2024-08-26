from rob_agi.colored_grid import ColoredGrid

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting non-black shapes towards the center.
    
    The function calculates the center of mass of non-black pixels, determines the target center,
    and shifts all non-black pixels to balance the composition while keeping shapes intact and
    within grid boundaries.
    """
    height, width = input_grid.get_dimensions()
    
    # Calculate current center of mass
    total_x, total_y, count = 0, 0, 0
    for y, row in enumerate(input_grid.values):
        for x, color in enumerate(row):
            if color != 0:
                total_x += x
                total_y += y
                count += 1
    
    if count == 0:
        return input_grid  # No non-black pixels, return original grid
    
    current_center = (total_x / count, total_y / count)
    
    # Determine target center
    target_center = (width / 2 - 0.5, height / 2 - 0.5)
    
    # Calculate shift
    shift_x = round(target_center[0] - current_center[0])
    shift_y = round(target_center[1] - current_center[1])
    
    # Adjust shift to keep shapes within grid
    min_x, max_x, min_y, max_y = width, 0, height, 0
    for y, row in enumerate(input_grid.values):
        for x, color in enumerate(row):
            if color != 0:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    shift_x = max(-min_x, min(width - 1 - max_x, shift_x))
    shift_y = max(-min_y, min(height - 1 - max_y, shift_y))
    
    # Create new grid and apply shift
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    for y, row in enumerate(input_grid.values):
        for x, color in enumerate(row):
            if color != 0:
                new_x, new_y = x + shift_x, y + shift_y
                if 0 <= new_x < width and 0 <= new_y < height:
                    new_grid.values[new_y][new_x] = color
    
    return new_grid
