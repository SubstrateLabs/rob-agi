from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_184a9768(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Processes each color (except sky blue) by expanding the largest region to its bounding box
    2. Processes sky blue (8) regions specially, filling empty spaces with yellow (4)
    3. Removes isolated cells and gray (5) cells
    4. Fills in surrounded empty cells
    
    The transformation processes colors in order of their first appearance, with sky blue (8) processed last.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def find_connected_regions(color: int) -> List[List[Tuple[int, int]]]:
        regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == color and (r, c) not in visited:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and output_grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            stack.extend(get_neighbors(curr_r, curr_c))
                    regions.append(region)
        return regions

    def get_largest_region(regions: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        return max(regions, key=len) if regions else []

    def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        if not region:
            return (0, 0, 0, 0)
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        return (min_r, min_c, max_r, max_c)

    def fill_bounding_box(color: int, bbox: Tuple[int, int, int, int]):
        min_r, min_c, max_r, max_c = bbox
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                output_grid.values[r][c] = color

    def process_color(color: int):
        regions = find_connected_regions(color)
        largest_region = get_largest_region(regions)
        bbox = get_bounding_box(largest_region)
        fill_bounding_box(color, bbox)

    def process_sky_blue():
        regions = find_connected_regions(8)
        largest_region = get_largest_region(regions)
        min_r, min_c, max_r, max_c = get_bounding_box(largest_region)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if output_grid.values[r][c] == 0:
                    output_grid.values[r][c] = 4
                elif output_grid.values[r][c] != 8:
                    output_grid.values[r][c] = 8

    def remove_isolated_cells():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] not in [0, 8]:
                    neighbors = get_neighbors(r, c)
                    if all(output_grid.values[nr][nc] != output_grid.values[r][c] for nr, nc in neighbors):
                        output_grid.values[r][c] = 0

    def remove_gray_cells():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == 5:
                    output_grid.values[r][c] = 0

    def fill_surrounded_cells():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == 0:
                    neighbors = get_neighbors(r, c)
                    neighbor_colors = [output_grid.values[nr][nc] for nr, nc in neighbors if output_grid.values[nr][nc] != 0]
                    if neighbor_colors and all(color == neighbor_colors[0] for color in neighbor_colors):
                        output_grid.values[r][c] = neighbor_colors[0]

    def get_color_order() -> List[int]:
        color_order = []
        for r in range(rows):
            for c in range(cols):
                color = input_grid.values[r][c]
                if color not in [0, 8, 5] and color not in color_order:
                    color_order.append(color)
        return color_order

    colors = get_color_order()
    for color in colors:
        process_color(color)
    
    process_sky_blue()
    remove_isolated_cells()
    remove_gray_cells()
    fill_surrounded_cells()
    fill_surrounded_cells()  # Second pass to ensure completeness

    return output_grid
