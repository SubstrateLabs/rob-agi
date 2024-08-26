from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_50aad11f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying magenta shapes, assigning them colors
    based on adjacent color indicators, and arranging them in a new grid.
    
    1. Identifies and sorts connected magenta regions in the input grid.
    2. Assigns colors to magenta regions based on nearby non-black, non-magenta colors.
    3. Creates a new grid with the colored shapes arranged horizontally.
    4. Compresses shapes vertically to fit in 4 rows while preserving key features.
    5. Adds spacing between shapes and optimizes the grid size.
    
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
            min_r = max(0, min(r for r, _ in region) - 1)
            max_r = min(rows - 1, max(r for r, _ in region) + 1)
            min_c = max(0, min(c for _, c in region) - 1)
            max_c = min(cols - 1, max(c for _, c in region) + 1)
            
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    cell_color = input_grid.get_cell(r, c)
                    if cell_color not in [0, 6]:
                        indicator = cell_color
                        break
                if indicator != 0:
                    break
            
            indicators.append(indicator)
        
        return indicators

    def compress_shape(shape: List[Tuple[int, int]], max_height: int) -> List[Tuple[int, int]]:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        height = max_r - min_r + 1
        width = max(c for _, c in shape) - min_c + 1
        
        if height <= max_height:
            return [(r - min_r, c - min_c) for r, c in shape]
        
        compressed_shape = set()
        for r, c in shape:
            new_r = min(max_height - 1, int((r - min_r) * (max_height - 1) / (height - 1)))
            compressed_shape.add((new_r, c - min_c))
        
        # Ensure key features are preserved
        if len(compressed_shape) < width:
            columns = set(c for _, c in shape)
            compressed_shape = set((r % max_height, c - min_c) for r, c in shape if c in columns)
        
        return sorted(compressed_shape)

    magenta_regions = find_magenta_regions()
    if not magenta_regions:
        return ColoredGrid(values=[[0] for _ in range(4)])

    color_indicators = find_color_indicators(magenta_regions)
    colored_regions = list(zip(color_indicators, magenta_regions))

    max_height = 4
    total_width = sum(max(c for _, c in region) - min(c for _, c in region) + 1 for _, region in colored_regions) + len(colored_regions) - 1

    output_grid = ColoredGrid(values=[[0 for _ in range(max(1, total_width))] for _ in range(max_height)])

    current_col = 0
    for color, region in colored_regions:
        compressed_region = compress_shape(region, max_height)
        width = max(c for _, c in compressed_region) + 1
        for r, c in compressed_region:
            output_grid.set_cell(r, current_col + c, color)
        current_col += width + 1

    # Remove trailing black columns
    while output_grid.get_dimensions()[1] > 1 and all(output_grid.get_cell(r, -1) == 0 for r in range(max_height)):
        for row in output_grid.values:
            row.pop()

    return output_grid
