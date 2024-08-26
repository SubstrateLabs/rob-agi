from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_5b692c0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Enhances shapes in the input grid while preserving their unique characteristics.
    
    The function performs the following steps:
    1. Identifies connected regions (shapes) in the input grid.
    2. For each shape:
       a. Analyzes its structure, orientation, and unique features.
       b. Creates a template based on the shape's most characteristic part.
       c. Enhances the shape using the template, applying symmetry selectively.
       d. Preserves unique features and color patterns.
       e. Refines the shape by smoothing edges and filling gaps.
    3. Places the enhanced shapes onto a new grid, maintaining their relative positions.
    
    This results in idealized versions of the original shapes that maintain their
    essential characteristics while improving their overall form and symmetry.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    regions = find_connected_regions(input_grid)
    for region in regions:
        enhanced_shape = enhance_shape(region, input_grid)
        place_shape(output_grid, enhanced_shape)
    
    return output_grid

def enhance_shape(region: List[Tuple[int, int, int]], input_grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    top, left, bottom, right = get_bounding_box(region)
    template = create_template(region, top, left, bottom, right)
    enhanced = apply_symmetry(region, template, top, left, bottom, right)
    enhanced = preserve_features(enhanced, region)
    enhanced = refine_shape(enhanced, top, left, bottom, right, input_grid)
    return enhanced

def get_bounding_box(region: List[Tuple[int, int, int]]) -> Tuple[int, int, int, int]:
    top = min(r for r, _, _ in region)
    bottom = max(r for r, _, _ in region)
    left = min(c for _, c, _ in region)
    right = max(c for _, c, _ in region)
    return top, left, bottom, right

def create_template(region: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> Dict[Tuple[int, int], int]:
    template = {}
    mid_row, mid_col = (top + bottom) // 2, (left + right) // 2
    for r, c, color in region:
        if r <= mid_row and c <= mid_col:
            template[(r - top, c - left)] = color
    return template

def apply_symmetry(region: List[Tuple[int, int, int]], template: Dict[Tuple[int, int], int], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    enhanced = region.copy()
    height, width = bottom - top + 1, right - left + 1
    for r in range(height):
        for c in range(width):
            if (r, c) in template:
                enhanced.extend([
                    (top + r, left + c, template[(r, c)]),
                    (bottom - r, left + c, template[(r, c)]),
                    (top + r, right - c, template[(r, c)]),
                    (bottom - r, right - c, template[(r, c)])
                ])
    return list(set(enhanced))

def preserve_features(enhanced: List[Tuple[int, int, int]], original: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    original_set = set((r, c) for r, c, _ in original)
    preserved = enhanced.copy()
    for r, c, color in original:
        if (r, c) not in original_set:
            preserved.append((r, c, color))
    return preserved

def refine_shape(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int, input_grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    refined = shape.copy()
    shape_dict = {(r, c): color for r, c, color in shape}
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) not in shape_dict:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc) in shape_dict]
                if len(neighbors) >= 5:
                    color_counts = defaultdict(int)
                    for nr, nc in neighbors:
                        color_counts[shape_dict[(nr, nc)]] += 1
                    most_common_color = max(color_counts, key=color_counts.get)
                    refined.append((r, c, most_common_color))
    return refined

def find_connected_regions(grid: ColoredGrid) -> List[List[Tuple[int, int, int]]]:
    regions = []
    visited = set()
    
    def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int, int]]:
        region = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and 0 <= curr_r < grid.num_rows and 0 <= curr_c < grid.num_cols and grid.get_cell(curr_r, curr_c) == color:
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c, color))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((curr_r + dr, curr_c + dc))
        return region

    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                regions.append(flood_fill(r, c, grid.get_cell(r, c)))
    
    return regions

def transform_shape(region: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    if not region:
        return []
    
    # Determine bounding box
    top = min(r for r, _, _ in region)
    bottom = max(r for r, _, _ in region)
    left = min(c for _, c, _ in region)
    right = max(c for _, c, _ in region)
    
    # Create a template based on the most characteristic part
    template = create_template(region, top, left, bottom, right)
    
    # Enhance the shape using the template
    enhanced_shape = enhance_shape(region, template, top, left, bottom, right)
    
    # Preserve unique features
    preserved_shape = preserve_features(enhanced_shape, region)
    
    # Refine the shape
    refined_shape = refine_shape(preserved_shape, top, left, bottom, right)
    
    return refined_shape

def create_template(region: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> Dict[Tuple[int, int], int]:
    template = {}
    mid_row = (top + bottom) // 2
    mid_col = (left + right) // 2
    
    for r, c, color in region:
        if r <= mid_row and c <= mid_col:
            template[(r - top, c - left)] = color
    
    return template

def enhance_shape(region: List[Tuple[int, int, int]], template: Dict[Tuple[int, int], int], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    enhanced = region.copy()
    height = bottom - top + 1
    width = right - left + 1
    
    for r in range(height):
        for c in range(width):
            if (r, c) in template:
                enhanced.append((top + r, left + c, template[(r, c)]))
                enhanced.append((bottom - r, left + c, template[(r, c)]))
                enhanced.append((top + r, right - c, template[(r, c)]))
                enhanced.append((bottom - r, right - c, template[(r, c)]))
    
    return list(set(enhanced))  # Remove duplicates

def preserve_features(enhanced_shape: List[Tuple[int, int, int]], original_shape: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    preserved = enhanced_shape.copy()
    original_set = set((r, c) for r, c, _ in original_shape)
    
    for r, c, color in original_shape:
        if (r, c) not in original_set:
            preserved.append((r, c, color))
    
    return preserved

def refine_shape(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    refined = shape.copy()
    shape_dict = {(r, c): color for r, c, color in shape}
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) not in shape_dict:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc) in shape_dict]
                if len(neighbors) >= 5:
                    color_counts = defaultdict(int)
                    for nr, nc in neighbors:
                        color_counts[shape_dict[(nr, nc)]] += 1
                    most_common_color = max(color_counts, key=color_counts.get)
                    refined.append((r, c, most_common_color))
    
    return refined

def mirror_horizontal(shape: List[Tuple[int, int, int]], top: int, bottom: int) -> List[Tuple[int, int, int]]:
    midline = (top + bottom) // 2
    mirrored = shape.copy()
    for r, c, color in shape:
        if r <= midline:
            mirrored.append((2 * midline - r, c, color))
    return mirrored

def mirror_vertical(shape: List[Tuple[int, int, int]], left: int, right: int) -> List[Tuple[int, int, int]]:
    midline = (left + right) // 2
    mirrored = shape.copy()
    for r, c, color in shape:
        if c <= midline:
            mirrored.append((r, 2 * midline - c, color))
    return mirrored

def expand_shape(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    expanded = shape.copy()
    shape_set = set((r, c) for r, c, _ in shape)
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) not in shape_set:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc) in shape_set]
                if neighbors:
                    color_counts = defaultdict(int)
                    for nr, nc in neighbors:
                        color = next(color for x, y, color in shape if x == nr and y == nc)
                        color_counts[color] += 1
                    most_common_color = max(color_counts, key=color_counts.get)
                    expanded.append((r, c, most_common_color))
    
    return expanded

def smooth_edges(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    smoothed = shape.copy()
    shape_dict = {(r, c): color for r, c, color in shape}
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) not in shape_dict:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc) in shape_dict]
                if len(neighbors) >= 5:
                    color_counts = defaultdict(int)
                    for nr, nc in neighbors:
                        color_counts[shape_dict[(nr, nc)]] += 1
                    most_common_color = max(color_counts, key=color_counts.get)
                    smoothed.append((r, c, most_common_color))
    
    return smoothed

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int, int]]):
    for r, c, color in shape:
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
            grid.set_cell(r, c, color)
