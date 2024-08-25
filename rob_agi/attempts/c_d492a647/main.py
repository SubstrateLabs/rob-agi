from rob_agi.colored_grid import ColoredGrid

def find_color_to_add(grid):
    for row in grid.values:
        for cell in row:
            if cell in [1, 3]:  # Blue or Green
                return cell
    return 3  # Default to Green if no color found

def find_enclosed_regions(grid):
    def is_enclosed(r, c):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                if grid.values[nr][nc] == 0:
                    return False
            else:
                return False
        return True

    def flood_fill(r, c, region):
        if (r, c) in region or grid.values[r][c] != 0:
            return
        region.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                flood_fill(nr, nc, region)

    enclosed_regions = []
    visited = set()
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if grid.values[r][c] == 0 and (r, c) not in visited and is_enclosed(r, c):
                region = set()
                flood_fill(r, c, region)
                enclosed_regions.append(region)
                visited.update(region)

    return enclosed_regions

def solve_d492a647(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Identifies the color to be added (green or blue) based on the input grid.
    2. Finds all enclosed black regions in the grid.
    3. For each enclosed region, applies a checkerboard pattern of the identified color,
       starting from the top-left corner of the region.
    4. Preserves all non-black squares and the overall structure of the grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the pattern applied.
    """
    color_to_add = find_color_to_add(input_grid)
    output_grid = input_grid.deep_copy()
    enclosed_regions = find_enclosed_regions(input_grid)

    for region in enclosed_regions:
        top_left = min(region)
        for r, c in region:
            if (r - top_left[0] + 1) % 2 == 1 and (c - top_left[1] + 1) % 2 == 1:
                output_grid.values[r][c] = color_to_add

    return output_grid
