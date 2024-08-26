from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_184a9768(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Identifies the largest contiguous region and expands it to its bounding box
    2. Places other color regions within the largest region, maintaining their relative positions
    3. Processes sky blue (8) regions specially, placing them near yellow (4) if possible
    4. Removes isolated cells and gray (5) cells
    5. Fills in surrounded empty cells
    6. Ensures all cells outside the largest region's bounding box are empty (0)
    
    The transformation reorganizes color regions while preserving their general shapes and relationships.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

        regions = {color: [] for color in range(1, 10)}
        visited = set()
        for r in range(rows):
            for c in range(cols):
                color = input_grid.values[r][c]
                if color != 0 and (r, c) not in visited:
                    region = []
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            queue.extend(get_neighbors(curr_r, curr_c))
                    regions[color].append(region)
        return regions

    def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        if not region:
            return (0, 0, 0, 0)
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        return (min_r, min_c, max_r, max_c)

    def fill_region(region: List[Tuple[int, int]], color: int, grid: ColoredGrid):
        for r, c in region:
            grid.values[r][c] = color

    def can_place_region(region: List[Tuple[int, int]], start_r: int, start_c: int, grid: ColoredGrid) -> bool:
        for r, c in region:
            new_r, new_c = r - region[0][0] + start_r, c - region[0][1] + start_c
            if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols or grid.values[new_r][new_c] != 0:
                return False
        return True

    def place_region(region: List[Tuple[int, int]], color: int, bbox: Tuple[int, int, int, int], grid: ColoredGrid) -> bool:
        min_r, min_c, max_r, max_c = bbox
        for start_r in range(min_r, max_r + 1):
            for start_c in range(min_c, max_c + 1):
                if can_place_region(region, start_r, start_c, grid):
                    for r, c in region:
                        new_r, new_c = r - region[0][0] + start_r, c - region[0][1] + start_c
                        grid.values[new_r][new_c] = color
                    return True
        return False

    def process_sky_blue(region: List[Tuple[int, int]], bbox: Tuple[int, int, int, int], grid: ColoredGrid):
        if place_region(region, 8, bbox, grid):
            for r, c in region:
                for nr, nc in get_neighbors(r, c):
                    if grid.values[nr][nc] == 0:
                        grid.values[nr][nc] = 4

    def remove_isolated_cells(grid: ColoredGrid):
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] not in [0, 8]:
                    neighbors = get_neighbors(r, c)
                    if all(grid.values[nr][nc] != grid.values[r][c] for nr, nc in neighbors):
                        grid.values[r][c] = 0

    def fill_surrounded_cells(grid: ColoredGrid):
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 0:
                    neighbors = get_neighbors(r, c)
                    neighbor_colors = [grid.values[nr][nc] for nr, nc in neighbors if grid.values[nr][nc] != 0]
                    if neighbor_colors and all(color == neighbor_colors[0] for color in neighbor_colors):
                        grid.values[r][c] = neighbor_colors[0]

    # Find all connected regions
    all_regions = find_connected_regions()

    # Find the largest region of any color
    largest_region = max((region for color_regions in all_regions.values() for region in color_regions), key=len)
    largest_color = input_grid.values[largest_region[0][0]][largest_region[0][1]]
    largest_bbox = get_bounding_box(largest_region)

    # Fill the largest region in the output grid
    fill_region(largest_region, largest_color, output_grid)

    # Process other colors in order: Red (2), Yellow (4), Sky Blue (8), Green (3), Magenta (6), Orange (7), Brown (9)
    color_order = [2, 4, 8, 3, 6, 7, 9]
    for color in color_order:
        if color in all_regions:
            for region in sorted(all_regions[color], key=len, reverse=True):
                if color == 8:
                    process_sky_blue(region, largest_bbox, output_grid)
                else:
                    place_region(region, color, largest_bbox, output_grid)

    # Clean up the grid
    remove_isolated_cells(output_grid)

    # Fill surrounded cells (two passes)
    fill_surrounded_cells(output_grid)
    fill_surrounded_cells(output_grid)

    # Ensure all cells outside the largest region's bounding box are empty
    min_r, min_c, max_r, max_c = largest_bbox
    for r in range(rows):
        for c in range(cols):
            if r < min_r or r > max_r or c < min_c or c > max_c:
                output_grid.values[r][c] = 0

    return output_grid
