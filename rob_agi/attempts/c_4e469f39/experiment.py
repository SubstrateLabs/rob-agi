from rob_agi.colored_grid import ColoredGrid

def find_connected_regions(grid: ColoredGrid, color: int) -> list[list[tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r: int, c: int) -> list[tuple[int, int]]:
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] != color:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc))
        return region

    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == color and (r, c) not in visited:
                regions.append(dfs(r, c))

    return regions

# Test case with multiple disconnected gray shapes
test_grid = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 0],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 0],
    [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 5, 0],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 0],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

gray_regions = find_connected_regions(test_grid, 5)
print("Connected gray regions:")
for i, region in enumerate(gray_regions):
    print(f"Region {i + 1}: {region}")

# Find overall bounding box
min_row = min(min(r for r, _ in region) for region in gray_regions)
max_row = max(max(r for r, _ in region) for region in gray_regions)
min_col = min(min(c for _, c in region) for region in gray_regions)
max_col = max(max(c for _, c in region) for region in gray_regions)

print(f"\nOverall bounding box: ({min_row}, {min_col}) to ({max_row}, {max_col})")

# Create a new grid with red outline and fill
result_grid = test_grid.deep_copy()

# Add top red line
for c in range(min_col, len(result_grid.values[0])):
    result_grid.values[min_row - 1][c] = 2

# Process columns
for c in range(min_col, max_col + 1):
    inside = False
    for r in range(min_row - 1, len(result_grid.values)):
        if test_grid.values[r][c] == 5:
            inside = not inside
        elif r > max_row or inside:
            result_grid.values[r][c] = 2
        elif c == min_col or c == max_col:
            result_grid.values[r][c] = 2

print("\nResult grid:")
for row in result_grid.values:
    print(" ".join(str(cell) for cell in row))
