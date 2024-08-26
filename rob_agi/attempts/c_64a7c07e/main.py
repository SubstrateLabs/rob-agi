from rob_agi.colored_grid import ColoredGrid

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting non-black shapes horizontally towards the center.
    
    The function finds the leftmost and rightmost non-black pixels, calculates the target position
    for the leftmost pixel, and shifts all non-black pixels horizontally to center the composition
    while keeping shapes intact and within grid boundaries. Vertical positions are maintained.
    """
    height, width = input_grid.get_dimensions()
    
    # Find leftmost and rightmost non-black pixels
    left_x, right_x = width, -1
    for y in range(height):
        for x in range(width):
            if input_grid.values[y][x] != 0:
                left_x = min(left_x, x)
                right_x = max(right_x, x)
    
    if left_x == width:  # No non-black pixels found
        return input_grid
    
    # Calculate target position and shift
    target_x = width // 2
    shift = target_x - left_x
    
    # Adjust shift if it would push pixels out of bounds
    if right_x + shift >= width:
        shift = width - 1 - right_x
    
    # Create new grid and apply shift
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    for y in range(height):
        for x in range(width):
            if input_grid.values[y][x] != 0:
                new_x = x + shift
                new_grid.values[y][new_x] = input_grid.values[y][x]
    
    return new_grid
