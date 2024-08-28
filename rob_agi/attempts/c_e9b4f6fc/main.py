from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_e9b4f6fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the largest non-black region,
    expanding it to include adjacent colored cells, extracting it,
    and transforming its colors based on their frequency in the entire input grid.
    
    The transformation includes:
    1. Identifying the largest non-black region and expanding it
    2. Extracting the expanded region
    3. Identifying the border color (most frequent on the edges)
    4. Ordering all non-black colors based on their frequency in the entire input grid
    5. Transforming colors: border color remains unchanged, other colors mapped from 1 to n-1
    6. Preserving the shape and border of the extracted region
    """
    # Step 1: Identify and expand the main colored region
    main_region = find_and_expand_largest_region(input_grid)
    
    # Step 2: Extract the expanded region
    extracted_grid = extract_region(input_grid, main_region)
    
    # Step 3: Identify border color
    border_color = identify_border_color(extracted_grid)
    
    # Step 4: Order all non-black colors based on frequency in the entire input grid
    all_colors = get_all_colors(input_grid)
    ordered_colors = order_colors_by_frequency(input_grid, all_colors - {0, border_color})
    
    # Step 5: Create color transformation mapping
    color_map = create_color_map(ordered_colors, border_color)
    
    # Step 6: Transform colors while preserving shape and border
    final_grid = transform_colors_preserve_shape(extracted_grid, color_map, border_color)
    
    return final_grid

def find_and_expand_largest_region(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    largest_region = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0 and (r, c) not in visited:
                region = find_connected_region(grid, r, c)
                if len(region) > len(largest_region):
                    largest_region = region
                visited.update(region)
    
    # Expand the largest region
    expanded_region = set(largest_region)
    for r, c in largest_region:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
                expanded_region.add((nr, nc))
    
    return list(expanded_region)

def find_connected_region(grid: ColoredGrid, start_r: int, start_c: int) -> List[Tuple[int, int]]:
    color = grid.get_cell(start_r, start_c)
    rows, cols = grid.get_dimensions()
    region = []
    stack = [(start_r, start_c)]
    visited = set()
    
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited:
            visited.add((r, c))
            if grid.get_cell(r, c) == color:
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
    
    return region

def extract_region(grid: ColoredGrid, region: List[Tuple[int, int]]) -> ColoredGrid:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    new_values = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            if (r, c) in region:
                row.append(grid.get_cell(r, c))
            else:
                row.append(0)  # Fill with black if not in the region
        new_values.append(row)
    
    return ColoredGrid(values=new_values)

def identify_border_color(grid: ColoredGrid) -> int:
    rows, cols = grid.get_dimensions()
    border_cells = (
        [(0, c) for c in range(cols)] +
        [(rows-1, c) for c in range(cols)] +
        [(r, 0) for r in range(1, rows-1)] +
        [(r, cols-1) for r in range(1, rows-1)]
    )
    border_colors = [grid.get_cell(r, c) for r, c in border_cells if grid.get_cell(r, c) != 0]
    return max(set(border_colors), key=border_colors.count)

def get_all_colors(grid: ColoredGrid) -> Set[int]:
    return set(cell for row in grid.values for cell in row)

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
    color_map = {0: 0, border_color: border_color}  # Keep black and border color unchanged
    for new_color, old_color in enumerate(ordered_colors, start=1):
        color_map[old_color] = new_color
    return color_map

def transform_colors_preserve_shape(grid: ColoredGrid, color_map: Dict[int, int], border_color: int) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            cell = grid.get_cell(r, c)
            new_row.append(color_map[cell])
        new_values.append(new_row)
    return ColoredGrid(values=new_values)
