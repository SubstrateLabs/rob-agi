from rob_agi.colored_grid import ColoredGrid

def solve_c35c1b4c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c35c1b4c challenge by expanding the largest contiguous region of color.
    
    The function identifies the largest connected region of a single color,
    then expands this region by changing adjacent cells to the same color
    if they are mostly surrounded by the expanding color.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the largest region expanded.
    """
    # Step 1: Identify largest region
    largest_region = None
    largest_size = 0
    expanding_color = None
    for color in range(10):  # Assuming colors are 0-9
        regions = input_grid.find_connected_regions(color)
        if regions and len(regions[0]) > largest_size:
            largest_region = regions[0]
            largest_size = len(regions[0])
            expanding_color = color

    # Step 2: Create a deep copy
    grid = input_grid.deep_copy()

    # Step 3: Define neighbor counting function
    def count_color_neighbors(x, y, color):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid.values) and 0 <= ny < len(grid.values[0]):
                    if grid.values[nx][ny] == color:
                        count += 1
        return count

    # Step 4: Expansion process
    queue = largest_region.copy()
    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid.values) and 0 <= ny < len(grid.values[0]):
                if grid.values[nx][ny] != expanding_color:
                    if count_color_neighbors(nx, ny, expanding_color) >= 5:
                        grid.values[nx][ny] = expanding_color
                        queue.append((nx, ny))

    # Step 5: Return modified grid
    return grid
