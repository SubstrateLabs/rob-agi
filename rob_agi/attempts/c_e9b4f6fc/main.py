from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict

def solve_e9b4f6fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the main colored region,
    extracting it, and transforming its colors based on frequency.
    
    The transformation includes:
    1. Identifying and extracting the largest non-black region
    2. Analyzing color frequencies in the extracted region
    3. Transforming colors based on their frequency (most frequent -> 1, second -> 2, etc.)
    4. Adjusting the shape if necessary (removing unnecessary rows/columns)
    
    No border is added in this implementation.
    """
    # Step 1: Identify and extract the main colored region
    main_region = find_largest_region(input_grid)
    extracted_grid = extract_region(input_grid, main_region)
    
    # Step 2: Analyze color frequencies
    color_freq = get_color_frequencies(extracted_grid)
    
    # Step 3: Create color transformation mapping
    color_map = create_color_map(color_freq)
    
    # Step 4: Transform colors
    transformed_grid = transform_colors(extracted_grid, color_map)
    
    # Step 5: Adjust shape (remove unnecessary rows/columns)
    final_grid = adjust_shape(transformed_grid)
    
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

def get_color_frequencies(grid: ColoredGrid) -> Dict[int, int]:
    return Counter(cell for row in grid.values for cell in row if cell != 0)

def create_color_map(color_freq: Dict[int, int]) -> Dict[int, int]:
    sorted_colors = sorted(color_freq.keys(), key=lambda x: (-color_freq[x], x))
    new_colors = [1, 2, 3, 4]  # blue, red, green, yellow
    return {old: new for old, new in zip(sorted_colors, new_colors)}

def transform_colors(grid: ColoredGrid, color_map: Dict[int, int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[color_map.get(grid.get_cell(r, c), grid.get_cell(r, c)) for c in range(cols)] for r in range(rows)]
    return ColoredGrid(values=new_values)

def adjust_shape(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    
    # Remove empty rows
    new_values = [row for row in grid.values if any(cell != 0 for cell in row)]
    
    # Remove empty columns
    if new_values:
        new_values = [[row[c] for c in range(cols) if any(new_values[r][c] != 0 for r in range(len(new_values)))] 
                      for row in new_values]
    
    return ColoredGrid(values=new_values)
