from rob_agi.colored_grid import ColoredGrid

def find_connected_region(grid, start_x, start_y):
    color = grid[start_x][start_y]
    visited = set()
    stack = [(start_x, start_y)]
    while stack:
        x, y = stack.pop()
        if (x, y) not in visited:
            visited.add((x, y))
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == color:
                    stack.append((nx, ny))
    return list(visited)

def check_adjacent_color(grid, region):
    for x, y in region:
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] not in [0, 8]:
                    return grid[nx][ny]
    return None

def solve_37d3e8b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying connected regions of sky color (8)
    and assigning them new colors based on size, adjacency, and a specific color sequence.
    
    The solution follows these steps:
    1. Create a copy of the input grid.
    2. Define a color sequence [1, 2, 3, 4, 5, 6, 7].
    3. Process large regions first, assigning primary colors (1, 2, 3).
    4. Process smaller regions, using adjacent colors or the next in the sequence.
    5. Handle any remaining unprocessed regions.
    6. Return the transformed grid.
    """
    grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 4, 5, 6, 7]
    primary_colors = [1, 2, 3]
    processed = set()

    # First pass: Handle large regions and assign primary colors
    color_index = 0
    for x in range(len(grid.values)):
        for y in range(len(grid.values[0])):
            if grid.values[x][y] == 8 and (x, y) not in processed:
                region = find_connected_region(grid.values, x, y)
                if len(region) > 10:
                    color = check_adjacent_color(grid.values, region)
                    if color is None:
                        color = primary_colors[color_index]
                        color_index = (color_index + 1) % len(primary_colors)
                    for rx, ry in region:
                        grid.values[rx][ry] = color
                        processed.add((rx, ry))

    # Second pass: Handle smaller regions
    color_index = 0
    for x in range(len(grid.values)):
        for y in range(len(grid.values[0])):
            if grid.values[x][y] == 8 and (x, y) not in processed:
                region = find_connected_region(grid.values, x, y)
                color = check_adjacent_color(grid.values, region)
                if color is None:
                    color = color_sequence[color_index]
                    color_index = (color_index + 1) % len(color_sequence)
                for rx, ry in region:
                    grid.values[rx][ry] = color
                    processed.add((rx, ry))

    # Final pass: Handle any remaining regions
    for x in range(len(grid.values)):
        for y in range(len(grid.values[0])):
            if grid.values[x][y] == 8:
                region = find_connected_region(grid.values, x, y)
                color = check_adjacent_color(grid.values, region) or 7
                for rx, ry in region:
                    grid.values[rx][ry] = color

    return grid
