from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_50aad11f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying magenta shapes, assigning them colors
    based on single-pixel color indicators, and arranging them in a new grid.
    
    1. Identifies connected magenta regions in the input grid.
    2. Finds single-pixel color indicators.
    3. Assigns colors to magenta regions based on the indicators.
    4. Creates a new grid with the colored shapes arranged horizontally.
    5. Optimizes the grid size by removing trailing black columns.
    
    Returns a new ColoredGrid with the transformed arrangement.
    """
    def flood_fill(start: Tuple[int, int], color: int) -> List[Tuple[int, int]]:
        stack = [start]
        region = set()
        rows, cols = input_grid.get_dimensions()
        while stack:
            r, c = stack.pop()
            if (r, c) not in region and 0 <= r < rows and 0 <= c < cols and input_grid.get_cell(r, c) == color:
                region.add((r, c))
                stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        return sorted(region)

    def find_magenta_regions() -> List[List[Tuple[int, int]]]:
        regions = []
        visited = set()
        rows, cols = input_grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and input_grid.get_cell(r, c) == 6:
                    region = flood_fill((r, c), 6)
                    regions.append(region)
                    visited.update(region)
        return sorted(regions, key=lambda x: (x[0][0], x[0][1]))

    def find_color_indicators() -> List[Tuple[int, int, int]]:
        indicators = []
        rows, cols = input_grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                color = input_grid.get_cell(r, c)
                if color not in [0, 6]:
                    indicators.append((color, r, c))
        return sorted(indicators, key=lambda x: (x[1], x[2]))

    def assign_colors(regions: List[List[Tuple[int, int]]], indicators: List[Tuple[int, int, int]]) -> List[Tuple[int, List[Tuple[int, int]]]]:
        colored_regions = []
        for i, region in enumerate(regions):
            color = indicators[i % len(indicators)][0]
            colored_regions.append((color, region))
        return colored_regions

    magenta_regions = find_magenta_regions()
    color_indicators = find_color_indicators()
    colored_regions = assign_colors(magenta_regions, color_indicators)

    max_height = 4
    total_width = sum(max(c for _, c in region) - min(c for _, c in region) + 1 for _, region in colored_regions) + len(colored_regions) - 1

    output_grid = ColoredGrid(values=[[0 for _ in range(total_width)] for _ in range(max_height)])

    current_col = 0
    for color, region in colored_regions:
        min_row = min(r for r, _ in region)
        min_col = min(c for _, c in region)
        for r, c in region:
            output_grid.set_cell(r - min_row, current_col + c - min_col, color)
        current_col += max(c for _, c in region) - min_col + 2

    # Remove trailing black columns
    while all(output_grid.get_cell(r, -1) == 0 for r in range(max_height)):
        for row in output_grid.values:
            row.pop()

    return output_grid
