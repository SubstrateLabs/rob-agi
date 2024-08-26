from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_e9b4f6fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the largest non-black region,
    extracting it, and transforming its colors based on their frequency.
    
    The transformation includes:
    1. Identifying and extracting the largest non-black region
    2. Identifying the border color (most frequent on the edges)
    3. Ordering interior colors based on their frequency in the extracted region
    4. Transforming colors: border color remains unchanged, interior colors mapped from 1 to n-1
    5. Adjusting the shape by removing unnecessary border-colored rows/columns
    """
    # Step 1: Identify and extract the main colored region
    main_region = find_largest_region(input_grid)
    extracted_grid = extract_region(input_grid, main_region)
    
    # Step 2: Identify border color
    border_color = identify_border_color(extracted_grid)
    
    # Step 3: Order interior colors based on frequency
    interior_colors = get_interior_colors(extracted_grid, border_color)
    ordered_colors = order_colors_by_frequency(extracted_grid, interior_colors)
    
    # Step 4: Create color transformation mapping
    color_map = create_color_map(ordered_colors, border_color)
    
    # Step 5: Transform colors
    transformed_grid = transform_colors(extracted_grid, color_map)
    
    # Step 6: Adjust shape (remove unnecessary border-colored rows/columns)
    final_grid = adjust_shape(transformed_grid, border_color)
    
    return final_grid

def find_largest_region(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    largest_region = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0 and (r, c) not in visited:
                region = grid.find_connected_regions(grid.get_cell(r, c))[0]
                if len(region) > len(largest_region):
                    largest_region = region
                visited.update(region)
    
    return largest_region

def extract_region(grid: ColoredGrid, region: List[Tuple[int, int]]) -> ColoredGrid:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    return grid.extract_subgrid(min_r, min_c, max_r - min_r + 1, max_c - min_c + 1)

def identify_border_color(grid: ColoredGrid) -> int:
    rows, cols = grid.get_dimensions()
    border_cells = (
        [(0, c) for c in range(cols)] +
        [(rows-1, c) for c in range(cols)] +
        [(r, 0) for r in range(1, rows-1)] +
        [(r, cols-1) for r in range(1, rows-1)]
    )
    border_colors = [grid.get_cell(r, c) for r, c in border_cells]
    return max(set(border_colors), key=border_colors.count)

def get_interior_colors(grid: ColoredGrid, border_color: int) -> Set[int]:
    return set(cell for row in grid.values for cell in row if cell != border_color)

def order_colors_by_frequency(grid: ColoredGrid, colors: Set[int]) -> List[int]:
    color_freq = {color: 0 for color in colors}
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color in colors:
                color_freq[color] += 1
    
    return sorted(colors, key=lambda color: (-color_freq[color], color))

def create_color_map(ordered_colors: List[int], border_color: int) -> Dict[int, int]:
    color_map = {border_color: border_color}  # Keep border color unchanged
    for new_color, old_color in enumerate(ordered_colors, start=1):
        color_map[old_color] = new_color
    return color_map

def transform_colors(grid: ColoredGrid, color_map: Dict[int, int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[color_map[grid.get_cell(r, c)] for c in range(cols)] for r in range(rows)]
    return ColoredGrid(values=new_values)

def adjust_shape(grid: ColoredGrid, border_color: int) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    top = next(r for r in range(rows) if any(grid.get_cell(r, c) != border_color for c in range(cols)))
    bottom = next(r for r in range(rows-1, -1, -1) if any(grid.get_cell(r, c) != border_color for c in range(cols)))
    left = next(c for c in range(cols) if any(grid.get_cell(r, c) != border_color for r in range(rows)))
    right = next(c for c in range(cols-1, -1, -1) if any(grid.get_cell(r, c) != border_color for r in range(rows)))
    
    new_values = [row[left:right+1] for row in grid.values[top:bottom+1]]
    return ColoredGrid(values=new_values)
