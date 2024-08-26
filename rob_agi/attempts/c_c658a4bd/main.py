from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c658a4bd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a square output grid with nested color frames.
    
    The solution analyzes the input grid to identify distinct colored regions,
    determines the order of colors from outside to inside based on their position and size,
    and creates a new grid with concentric frames of these colors.
    
    The output grid size is determined by the number of distinct colors,
    and each color forms a complete frame around the inner colors. The innermost
    color is handled specially:
    - If there's only one color, it fills the center cell.
    - If there are two colors, it fills the center 2x2 square.
    - If there are three or more colors, it fills a 2x2 square in the center if possible,
      or just the center cell if the grid size is even.
    
    The algorithm considers the largest connected region for each color, prioritizing colors
    based on their distance from the edge, size of the region, and position. The background
    color (black/0) is ignored in calculations.
    """
    def analyze_grid(grid: ColoredGrid) -> List[Dict]:
        regions = []
        for color in range(1, 10):  # Assuming colors are 1-9
            color_regions = grid.find_connected_regions(color)
            if color_regions:
                region = max(color_regions, key=len)  # Use the largest region for each color
                min_x = min(c for _, c in region)
                max_x = max(c for _, c in region)
                min_y = min(r for r, _ in region)
                max_y = max(r for r, _ in region)
                distance_from_edge = min(min_x, min_y, grid.num_cols - max_x - 1, grid.num_rows - max_y - 1)
                regions.append({
                    'color': color,
                    'size': len(region),
                    'distance_from_edge': distance_from_edge,
                    'top': min_y,
                    'left': min_x
                })
        return regions

    def order_colors(regions: List[Dict]) -> List[int]:
        return [r['color'] for r in sorted(regions, key=lambda x: (x['distance_from_edge'], -x['size'], x['top'], x['left']))]

    def create_output_grid(ordered_colors: List[int]) -> ColoredGrid:
        size = 2 * len(ordered_colors) - 1
        output = ColoredGrid(values=[[0 for _ in range(size)] for _ in range(size)])
        
        for i, color in enumerate(ordered_colors):
            frame_size = size - 2 * i
            for x in range(frame_size):
                for y in range(frame_size):
                    if x == 0 or x == frame_size - 1 or y == 0 or y == frame_size - 1:
                        output.values[i + y][i + x] = color
        
        # Handle the innermost color
        if len(ordered_colors) == 1:
            center = len(ordered_colors) - 1
            output.values[center][center] = ordered_colors[-1]
        elif len(ordered_colors) >= 2:
            center = len(ordered_colors) - 1
            output.values[center-1][center-1] = ordered_colors[-1]
            output.values[center-1][center] = ordered_colors[-1]
            output.values[center][center-1] = ordered_colors[-1]
            output.values[center][center] = ordered_colors[-1]
        
        return output

    regions = analyze_grid(input_grid)
    ordered_colors = order_colors(regions)
    return create_output_grid(ordered_colors)
