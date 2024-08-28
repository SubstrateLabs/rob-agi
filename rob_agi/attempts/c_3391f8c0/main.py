from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_3391f8c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Vertical flip of the entire grid
    2. Color transformation based on the presence of specific colors
    3. Pattern identification, transformation, and relocation
    4. Adjustment of patterns to fit available space while maintaining their general shape and concept

    Color transformation rules:
    - 1 (blue) <-> 7 (orange) if 7 is present, otherwise 1 <-> 8 (sky)
    - 2 (red) <-> 3 (green)
    - Other colors remain unchanged
    - Black (0) is treated as empty space

    Patterns are identified, transformed, relocated to opposite corners or sides, and adjusted to maintain 
    their essential shape and concept while adapting to the new grid configuration.
    """
    height, width = input_grid.get_dimensions()
    unique_colors = input_grid.get_unique_colors()

    # Step 1: Create color transformation map
    color_map = create_color_map(unique_colors)

    # Step 2: Vertical flip and color transformation
    output_grid = vertical_flip_and_transform(input_grid, color_map)

    # Step 3: Identify and transform patterns
    patterns = []
    for color in unique_colors:
        if color != 0:
            regions = output_grid.find_connected_regions(color)
            patterns.extend([(color, region) for region in regions])

    # Step 4: Transform and relocate patterns
    transform_and_relocate_patterns(output_grid, patterns)

    return output_grid

def create_color_map(unique_colors: set) -> dict:
    color_map = {}
    if 1 in unique_colors:
        if 7 in unique_colors:
            color_map[1], color_map[7] = 7, 1
        elif 8 in unique_colors:
            color_map[1], color_map[8] = 8, 1
    if 2 in unique_colors and 3 in unique_colors:
        color_map[2], color_map[3] = 3, 2
    return color_map

def vertical_flip_and_transform(grid: ColoredGrid, color_map: dict) -> ColoredGrid:
    height, width = grid.get_dimensions()
    new_values = []
    for i in range(height - 1, -1, -1):
        row = [color_map.get(grid.get_cell(i, j), grid.get_cell(i, j)) for j in range(width)]
        new_values.append(row)
    return ColoredGrid(values=new_values)

def transform_and_relocate_patterns(grid: ColoredGrid, patterns: List[Tuple[int, List[Tuple[int, int]]]]):
    """Transforms and relocates all patterns in the grid."""
    grid_center = (grid.num_rows // 2, grid.num_cols // 2)
    
    for color, region in patterns:
        if not region:
            continue
        
        # Calculate pattern center
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        pattern_center = ((min_row + max_row) // 2, (min_col + max_col) // 2)
        
        # Calculate new position (opposite side of the grid center)
        new_center = (
            2 * grid_center[0] - pattern_center[0],
            2 * grid_center[1] - pattern_center[1]
        )
        
        # Clear the original pattern
        for r, c in region:
            grid.set_cell(r, c, 0)
        
        # Move and adjust the pattern
        new_region = move_and_adjust_pattern(grid, region, new_center, color)
        
        # Fill the new pattern
        for r, c in new_region:
            if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                grid.set_cell(r, c, color)

def move_and_adjust_pattern(grid: ColoredGrid, region: List[Tuple[int, int]], new_center: Tuple[int, int], color: int) -> List[Tuple[int, int]]:
    """Moves and adjusts a pattern to fit in the new location."""
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    new_min_row = new_center[0] - height // 2
    new_min_col = new_center[1] - width // 2
    
    new_region = []
    for r, c in region:
        new_r = new_min_row + (r - min_row)
        new_c = new_min_col + (c - min_col)
        if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
            new_region.append((new_r, new_c))
    
    # Adjust pattern if it doesn't fit
    while not all(0 <= r < grid.num_rows and 0 <= c < grid.num_cols for r, c in new_region):
        new_region = contract_pattern(new_region)
    
    return new_region

def contract_pattern(region: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Contracts a pattern by removing outer cells."""
    if not region:
        return []
    
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    
    return [
        (r, c) for r, c in region
        if min_row < r < max_row and min_col < c < max_col
    ]

def can_expand(grid: ColoredGrid, row: int, col: int, height: int, width: int) -> bool:
    return all(grid.get_cell(r, c) == 0 
               for r in range(row, min(row + height, grid.num_rows)) 
               for c in range(col, min(col + width, grid.num_cols)))

def needs_contraction(grid: ColoredGrid, row: int, col: int, height: int, width: int) -> bool:
    return row + height > grid.num_rows or col + width > grid.num_cols

def expand_pattern(grid: ColoredGrid, region: List[Tuple[int, int]], new_row: int, new_col: int) -> List[Tuple[int, int]]:
    color = grid.get_cell(region[0][0], region[0][1])
    expanded_region = []
    for r, c in sorted(region, key=lambda x: (x[0] - new_row) ** 2 + (x[1] - new_col) ** 2):
        new_r, new_c = 2 * new_row - r, 2 * new_col - c
        if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
            grid.set_cell(new_r, new_c, color)
            expanded_region.append((new_r, new_c))
    return expanded_region

def contract_pattern(grid: ColoredGrid, region: List[Tuple[int, int]], new_row: int, new_col: int) -> List[Tuple[int, int]]:
    color = grid.get_cell(region[0][0], region[0][1])
    contracted_region = []
    for r, c in sorted(region, key=lambda x: (x[0] - new_row) ** 2 + (x[1] - new_col) ** 2, reverse=True):
        new_r, new_c = (r + new_row) // 2, (c + new_col) // 2
        if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
            grid.set_cell(new_r, new_c, color)
            contracted_region.append((new_r, new_c))
    return contracted_region

def move_pattern(grid: ColoredGrid, region: List[Tuple[int, int]], row_offset: int, col_offset: int) -> List[Tuple[int, int]]:
    color = grid.get_cell(region[0][0], region[0][1])
    moved_region = []
    for r, c in sorted(region, key=lambda x: (x[0] + row_offset) ** 2 + (x[1] + col_offset) ** 2, reverse=True):
        new_r, new_c = r + row_offset, c + col_offset
        if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
            grid.set_cell(new_r, new_c, color)
            moved_region.append((new_r, new_c))
    return moved_region
    """Adjusts a pattern to fit available space while maintaining its general shape."""
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # Check if pattern can be expanded vertically
    if min_row > 0 and all(grid.get_cell(min_row - 1, c) == 0 for c in range(min_col, max_col + 1)):
        for r, c in sorted(region):
            grid.set_cell(r - 1, c, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)
    elif max_row < grid.num_rows - 1 and all(grid.get_cell(max_row + 1, c) == 0 for c in range(min_col, max_col + 1)):
        for r, c in sorted(region, reverse=True):
            grid.set_cell(r + 1, c, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)

    # Check if pattern can be expanded horizontally
    if min_col > 0 and all(grid.get_cell(r, min_col - 1) == 0 for r in range(min_row, max_row + 1)):
        for r, c in sorted(region, key=lambda x: x[1]):
            grid.set_cell(r, c - 1, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)
    elif max_col < grid.num_cols - 1 and all(grid.get_cell(r, max_col + 1) == 0 for r in range(min_row, max_row + 1)):
        for r, c in sorted(region, key=lambda x: x[1], reverse=True):
            grid.set_cell(r, c + 1, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)
def adjust_pattern_position(grid: ColoredGrid, region: List[Tuple[int, int]]):
    """Adjusts the pattern position to better utilize space and avoid overlaps."""
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    
    color = grid.get_cell(region[0][0], region[0][1])
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Try to move the pattern towards the center if possible
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in directions:
        new_min_row = min_row + dr
        new_min_col = min_col + dc
        if (0 <= new_min_row < grid.num_rows - height + 1 and
            0 <= new_min_col < grid.num_cols - width + 1 and
            all(grid.get_cell(new_min_row + r, new_min_col + c) == 0
                for r in range(height) for c in range(width))):
            # Move the pattern
            for r, c in sorted(region, reverse=True):
                grid.set_cell(r, c, 0)
                grid.set_cell(r + dr, c + dc, color)
            return  # Stop after one successful move
