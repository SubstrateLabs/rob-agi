from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_50aad11f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying magenta shapes, assigning them colors
    based on adjacent single-pixel color indicators, and arranging them
    in a new grid.
    
    1. Identifies connected magenta regions in the input grid.
    2. Finds single-pixel color indicators adjacent to each magenta region.
    3. Assigns colors to magenta regions based on the indicators.
    4. Creates a new grid with the colored shapes arranged horizontally.
    5. Compresses shapes vertically if needed to fit in 4 rows.
    6. Optimizes the grid size by removing trailing black columns.
    
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
        return sorted(regions, key=lambda x: (min(c for _, c in x), min(r for r, _ in x)))

    def find_color_indicators(regions: List[List[Tuple[int, int]]]) -> List[int]:
        indicators = []
        rows, cols = input_grid.get_dimensions()
        for region in regions:
            indicator = 0  # Default color (black) if no indicator found
            min_x = min(x for x, _ in region)
            max_x = max(x for x, _ in region)
            min_y = min(y for _, y in region)
            max_y = max(y for _, y in region)
        
            for x in range(max(0, min_x - 1), min(rows, max_x + 2)):
                for y in range(max(0, min_y - 1), min(cols, max_y + 2)):
                    cell_color = input_grid.get_cell(x, y)
                    if cell_color not in [0, 6]:
                        indicator = cell_color
                        break
                if indicator != 0:
                    break
        
            if indicator == 0 and indicators:  # If no indicator found, use the previous color
                indicator = indicators[-1]
            indicators.append(indicator)
    
        return indicators

    def assign_colors(regions: List[List[Tuple[int, int]]], indicators: List[int]) -> List[Tuple[int, List[Tuple[int, int]]]]:
        colored_regions = []
        for region, color in zip(regions, indicators):
            colored_regions.append((color, region))
        return colored_regions

    def compress_shape(shape: List[Tuple[int, int]], max_height: int) -> List[Tuple[int, int]]:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        height = max_r - min_r + 1
        if height <= max_height:
            return [(r - min_r, c - min_c) for r, c in shape]
        
        compression_ratio = max_height / height
        compressed_shape = []
        for r, c in shape:
            new_r = int((r - min_r) * compression_ratio)
            if new_r < max_height:
                compressed_shape.append((new_r, c - min_c))
        return compressed_shape

    magenta_regions = find_magenta_regions()
    color_indicators = find_color_indicators(magenta_regions)
    colored_regions = assign_colors(magenta_regions, color_indicators)

    max_height = 4
    total_width = sum(max(c for _, c in region) - min(c for _, c in region) + 1 for _, region in colored_regions) + len(colored_regions) - 1

    output_grid = ColoredGrid(values=[[0 for _ in range(total_width)] for _ in range(max_height)])

    current_col = 0
    for color, region in colored_regions:
        compressed_region = compress_shape(region, max_height)
        width = max(c for _, c in compressed_region) + 1
        for r, c in compressed_region:
            output_grid.set_cell(r, current_col + c, color)
        current_col += width + 1

    # Remove trailing black columns
    while all(output_grid.get_cell(r, -1) == 0 for r in range(max_height)):
        for row in output_grid.values:
            row.pop()

    return output_grid
