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

def solve_37d3e8b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying connected regions of sky color (8)
    and assigning them new colors based on a specific sequence and connectivity rules.
    
    The solution follows these steps:
    1. Create a copy of the input grid.
    2. Initialize a color sequence [1, 2, 3, 4, 5, 6, 7].
    3. Scan the grid from top-left to bottom-right.
    4. For each sky-colored (8) cell:
       a. Find the connected region.
       b. If the region is connected to a previously colored region, use that color.
       c. Otherwise, use the next color in the sequence.
    5. Color all cells in the region with the chosen color.
    6. Return the transformed grid.
    """
    grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 4, 5, 6, 7]
    color_index = 0
    processed = set()

    for x in range(len(grid.values)):
        for y in range(len(grid.values[0])):
            if grid.values[x][y] == 8 and (x, y) not in processed:
                region = find_connected_region(grid.values, x, y)
                color = None
                for rx, ry in region:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nx, ny = rx + dx, ry + dy
                        if (nx, ny) in processed and grid.values[nx][ny] != 0:
                            color = grid.values[nx][ny]
                            break
                    if color:
                        break
                if color is None:
                    color = color_sequence[color_index]
                    color_index = (color_index + 1) % len(color_sequence)
                for rx, ry in region:
                    grid.values[rx][ry] = color
                    processed.add((rx, ry))

    return grid
