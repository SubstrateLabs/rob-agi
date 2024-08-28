from rob_agi.colored_grid import ColoredGrid

def find_connected_regions(grid):
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r, c, color, region):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] != color:
            return
        visited.add((r, c))
        region.append((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                dfs(r + dr, c + dc, color, region)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                region = []
                dfs(r, c, grid.values[r][c], region)
                regions.append(region)

    return regions

def update_region(grid, region):
    rows, cols = grid.get_dimensions()
    original_color = grid.values[region[0][0]][region[0][1]]
    neighbor_colors = {original_color}
    
    for x, y in region:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    neighbor_colors.add(grid.values[nx][ny])
    
    max_color = max(neighbor_colors)
    
    for x, y in region:
        grid.values[x][y] = max_color

    return max_color != original_color

def solve_step_by_step(input_grid):
    output_grid = input_grid.deep_copy()
    step = 0
    changed = True

    while changed:
        changed = False
        regions = find_connected_regions(output_grid)
        for region in regions:
            if update_region(output_grid, region):
                changed = True
        
        print(f"Step {step}:")
        print(output_grid)
        print()
        step += 1

    return output_grid

# Example case
input_grid = ColoredGrid(values=[
    [0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0],
    [0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 8, 0]
])

print("Input Grid:")
print(input_grid)
print()

solve_step_by_step(input_grid)
