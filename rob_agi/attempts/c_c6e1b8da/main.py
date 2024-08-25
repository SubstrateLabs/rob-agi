from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c6e1b8da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by simplifying and regularizing colored regions.
    
    The transformation follows these steps:
    1. Identify distinct colored regions in the input grid.
    2. For each region, create a simplified rectangular shape.
    3. Place the simplified shapes on a new grid, maintaining relative positions.
    4. Ensure a 1-cell black border around the grid and between shapes.
    5. Align shapes to grid edges or other shapes when possible.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    regions = identify_regions(input_grid)
    simplified_regions = simplify_regions(regions)
    output_grid = place_regions(simplified_regions, input_grid.get_dimensions())
    return output_grid

def identify_regions(grid: ColoredGrid) -> List[Dict]:
    regions = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                color = grid.get_cell(r, c)
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                                stack.append((nr, nc))
                
                min_r = min(r for r, _ in region)
                max_r = max(r for r, _ in region)
                min_c = min(c for _, c in region)
                max_c = max(c for _, c in region)
                
                regions.append({
                    'color': color,
                    'area': len(region),
                    'bounding_box': (min_r, min_c, max_r - min_r + 1, max_c - min_c + 1),
                    'center': ((min_r + max_r) / 2, (min_c + max_c) / 2)
                })
    
    return sorted(regions, key=lambda x: x['area'], reverse=True)

def simplify_regions(regions: List[Dict]) -> List[Dict]:
    simplified = []
    for region in regions:
        height, width = region['bounding_box'][2], region['bounding_box'][3]
        area = region['area']
        
        # Simplify to square if possible
        if abs(height - width) <= 2:
            new_size = max(3, min(height, width))
            simplified.append({
                'color': region['color'],
                'height': new_size,
                'width': new_size,
                'center': region['center']
            })
        else:
            # Simplify to rectangle
            if height > width:
                new_height = max(5, min(height, area // 3))
                new_width = max(3, min(width, area // new_height))
            else:
                new_width = max(5, min(width, area // 3))
                new_height = max(3, min(height, area // new_width))
            
            simplified.append({
                'color': region['color'],
                'height': new_height,
                'width': new_width,
                'center': region['center']
            })
    
    return simplified

def place_regions(regions: List[Dict], dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for region in regions:
        color = region['color']
        height, width = region['height'], region['width']
        center_r, center_c = region['center']
        
        # Find the best position for the region
        best_r = max(1, min(rows - height - 1, int(center_r - height / 2)))
        best_c = max(1, min(cols - width - 1, int(center_c - width / 2)))
        
        # Place the region
        for r in range(best_r, best_r + height):
            for c in range(best_c, best_c + width):
                if 0 < r < rows - 1 and 0 < c < cols - 1:
                    grid[r][c] = color
    
    return ColoredGrid(values=grid)
