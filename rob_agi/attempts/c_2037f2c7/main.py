from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting key features from shapes,
    merging them, and creating a simplified horizontal representation.
    
    1. Identifies non-zero regions in the input grid
    2. Extracts features from these regions (edges, symmetry, extensions)
    3. Merges features, prioritizing those near corners
    4. Transforms vertical features to horizontal
    5. Simplifies the merged features
    6. Generates a small output grid with sky blue (8) elements
    7. Applies symmetry and balances the pattern
    """
    # Step 1: Identify non-zero regions
    regions = find_non_zero_regions(input_grid)
    
    # Step 2 & 3: Extract and merge features
    merged_features = merge_features(regions, input_grid)
    
    # Step 4 & 5: Transform and simplify features
    simplified_features = simplify_features(merged_features)
    
    # Step 6 & 7: Generate output grid and apply symmetry
    output_grid = generate_output_grid(simplified_features)
    
    return output_grid

def find_non_zero_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    return grid.find_connected_regions(lambda x: x != 0)

def merge_features(regions: List[List[Tuple[int, int]]], grid: ColoredGrid) -> List[Tuple[int, int]]:
    merged = []
    for region in regions:
        top = min(r for r, _ in region)
        bottom = max(r for r, _ in region)
        left = min(c for _, c in region)
        right = max(c for _, c in region)
        
        # Prioritize corner regions
        weight = 1 / min(top + 1, left + 1, grid.num_rows - bottom, grid.num_cols - right)
        
        merged.extend([(r, c, weight) for r, c in region])
    
    return sorted(merged, key=lambda x: -x[2])[:50]  # Keep top 50 weighted points

def simplify_features(features: List[Tuple[int, int, float]]) -> List[Tuple[int, int]]:
    # Convert vertical to horizontal and normalize
    max_row = max(r for r, _, _ in features)
    max_col = max(c for _, c, _ in features)
    
    normalized = [(c / max_col * 7, (max_row - r) / max_row * 3) for r, c, _ in features]
    return [(round(r), round(c)) for r, c in normalized]

def generate_output_grid(features: List[Tuple[int, int]]) -> ColoredGrid:
    output = [[0 for _ in range(8)] for _ in range(4)]
    
    for r, c in features:
        if 0 <= r < 4 and 0 <= c < 8:
            output[r][c] = 8
    
    # Apply horizontal symmetry
    for r in range(4):
        for c in range(4):
            if output[r][c] == 8:
                output[r][7-c] = 8
    
    return ColoredGrid(values=output)
