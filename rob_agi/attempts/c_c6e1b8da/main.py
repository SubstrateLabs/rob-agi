from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c6e1b8da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by simplifying and regularizing colored regions.
    
    The transformation follows these steps:
    1. Identify distinct colored regions in the input grid.
    2. Simplify each region into a rectangular or square shape.
    3. Place simplified regions on a new grid, maintaining relative positions.
    4. Optimize layout by aligning regions vertically and horizontally.
    5. Ensure a 1-cell black border around the grid and between regions.
    6. Adjust regions to touch grid edges when possible.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    regions = identify_regions(input_grid)
    simplified_regions = simplify_regions(regions)
    output_grid = place_and_optimize_regions(simplified_regions, input_grid.get_dimensions())
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
        
        if abs(height - width) <= 2:
            new_size = max(3, min(height, width))
            simplified.append({
                'color': region['color'],
                'height': new_size,
                'width': new_size,
                'center': region['center']
            })
        else:
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

def place_and_optimize_regions(regions: List[Dict], dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initial placement
    for region in regions:
        color = region['color']
        height, width = region['height'], region['width']
        center_r, center_c = region['center']
        
        best_r = max(1, min(rows - height - 1, int(center_r - height / 2)))
        best_c = max(1, min(cols - width - 1, int(center_c - width / 2)))
        
        # Try to attach to edges
        if best_r < rows // 2:
            best_r = 1
        elif best_r > rows // 2:
            best_r = rows - height - 1
        
        if best_c < cols // 2:
            best_c = 1
        elif best_c > cols // 2:
            best_c = cols - width - 1
        
        # Place the region
        for r in range(best_r, best_r + height):
            for c in range(best_c, best_c + width):
                if 0 < r < rows - 1 and 0 < c < cols - 1:
                    grid[r][c] = color
    
    # Optimize layout
    optimize_layout(grid)
    
    return ColoredGrid(values=grid)

def optimize_layout(grid: List[List[int]]):
    rows, cols = len(grid), len(grid[0])
    
    # Vertical alignment
    for c in range(1, cols - 1):
        align_column(grid, c)
    
    # Horizontal grouping
    for r in range(1, rows - 1):
        group_row(grid, r)
    
    # Ensure borders
    ensure_borders(grid)

def align_column(grid: List[List[int]], col: int):
    rows = len(grid)
    current_color = 0
    start_row = 0
    
    for r in range(1, rows - 1):
        if grid[r][col] != current_color:
            if current_color != 0:
                center = (start_row + r) // 2
                fill_vertical(grid, col, start_row, r - 1, current_color)
            current_color = grid[r][col]
            start_row = r

    if current_color != 0:
        fill_vertical(grid, col, start_row, rows - 2, current_color)

def group_row(grid: List[List[int]], row: int):
    cols = len(grid[0])
    current_color = 0
    start_col = 0
    
    for c in range(1, cols - 1):
        if grid[row][c] != current_color:
            if current_color != 0:
                center = (start_col + c) // 2
                fill_horizontal(grid, row, start_col, c - 1, current_color)
            current_color = grid[row][c]
            start_col = c

    if current_color != 0:
        fill_horizontal(grid, row, start_col, cols - 2, current_color)

def fill_vertical(grid: List[List[int]], col: int, start: int, end: int, color: int):
    for r in range(start, end + 1):
        grid[r][col] = color

def fill_horizontal(grid: List[List[int]], row: int, start: int, end: int, color: int):
    for c in range(start, end + 1):
        grid[row][c] = color

def ensure_borders(grid: List[List[int]]):
    rows, cols = len(grid), len(grid[0])
    
    # Ensure top and bottom borders
    for c in range(cols):
        grid[0][c] = 0
        grid[rows-1][c] = 0
    
    # Ensure left and right borders
    for r in range(rows):
        grid[r][0] = 0
        grid[r][cols-1] = 0
    
    # Ensure borders between regions
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != 0:
                if grid[r-1][c] != grid[r][c] and grid[r-1][c] != 0:
                    grid[r-1][c] = 0
                if grid[r+1][c] != grid[r][c] and grid[r+1][c] != 0:
                    grid[r+1][c] = 0
                if grid[r][c-1] != grid[r][c] and grid[r][c-1] != 0:
                    grid[r][c-1] = 0
                if grid[r][c+1] != grid[r][c] and grid[r][c+1] != 0:
                    grid[r][c+1] = 0
