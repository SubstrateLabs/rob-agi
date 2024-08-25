from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_e9b4f6fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the largest non-black region,
    extracting it, and transforming its colors based on their first appearance.
    
    The transformation includes:
    1. Identifying and extracting the largest non-black region
    2. Identifying the border color and interior colors
    3. Ordering interior colors based on their first appearance in the original grid
    4. Transforming colors: border color to highest value, interior colors from 1 to n-1
    5. Adjusting the shape by removing unnecessary border-colored rows/columns
    """
    # Step 1: Identify and extract the main colored region
    main_region = find_largest_region(input_grid)
    extracted_grid = extract_region(input_grid, main_region)
    
    # Step 2: Identify border color and interior colors
    border_color = identify_border_color(extracted_grid)
    interior_colors = get_unique_colors(extracted_grid) - {border_color}
    
    # Step 3: Order interior colors based on first appearance
    ordered_colors = order_colors(extracted_grid, interior_colors)
    
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

def get_unique_colors(grid: ColoredGrid) -> Set[int]:
    return set(cell for row in grid.values for cell in row if cell != 0)

def order_colors(grid: ColoredGrid, interior_colors: Set[int]) -> List[int]:
    color_positions = {color: (float('inf'), float('inf')) for color in interior_colors}
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color in interior_colors:
                color_positions[color] = min(color_positions[color], (r, c))
    
    return sorted(interior_colors, key=lambda color: color_positions[color])

def create_color_map(ordered_colors: List[int], border_color: int) -> Dict[int, int]:
    new_colors = range(1, len(ordered_colors) + 1)
    color_map = dict(zip(ordered_colors, new_colors))
    color_map[border_color] = len(ordered_colors) + 1
    return color_map

def transform_colors(grid: ColoredGrid, color_map: Dict[int, int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[color_map[grid.get_cell(r, c)] for c in range(cols)] for r in range(rows)]
    return ColoredGrid(values=new_values)

def adjust_shape(grid: ColoredGrid, border_color: int) -> ColoredGrid:
    new_values = remove_border_rows(grid.values, border_color)
    new_values = remove_border_columns(new_values, border_color)
    return ColoredGrid(values=new_values)

def remove_border_rows(values: List[List[int]], border_color: int) -> List[List[int]]:
    return [row for row in values if any(cell != border_color for cell in row)]

def remove_border_columns(values: List[List[int]], border_color: int) -> List[List[int]]:
    if not values:
        return values
    cols = len(values[0])
    return [[row[c] for c in range(cols) if any(values[r][c] != border_color for r in range(len(values)))] for row in values]
