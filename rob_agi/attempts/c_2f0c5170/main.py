from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2f0c5170(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most significant pattern from black regions in the input grid,
    centers it, and returns it as a new ColoredGrid.
    
    1. Scans the input grid for black regions
    2. Analyzes each black region for significance
    3. Selects the most significant region
    4. Extracts the chosen region
    5. Centers the pattern in a new grid
    6. Returns the result as a ColoredGrid
    """
    
    def find_black_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
        rows, cols = grid.get_dimensions()
        visited = set()
        regions = []
        
        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            if (r, c) in visited or grid.get_cell(r, c) != 0:
                return []
            visited.add((r, c))
            region = [(r, c)]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    region.extend(dfs(nr, nc))
            return region
        
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0 and (r, c) not in visited:
                    regions.append(dfs(r, c))
        
        return regions

    def calculate_significance(grid: ColoredGrid, region: List[Tuple[int, int]]) -> int:
        non_black = set()
        unique_colors = set()
        for r, c in region:
            color = grid.get_cell(r, c)
            if color != 0:
                non_black.add((r, c))
                unique_colors.add(color)
        return len(non_black) * 2 + len(unique_colors) * 3

    def extract_region(grid: ColoredGrid, region: List[Tuple[int, int]]) -> ColoredGrid:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        return grid.extract_subgrid(min_r, min_c, max_r - min_r + 1, max_c - min_c + 1)

    def center_pattern(pattern: ColoredGrid) -> ColoredGrid:
        pattern_rows, pattern_cols = pattern.get_dimensions()
        new_rows = max(pattern_rows, 5)  # Ensure minimum size of 5x5
        new_cols = max(pattern_cols, 5)
        vertical_padding = (new_rows - pattern_rows) // 2
        horizontal_padding = (new_cols - pattern_cols) // 2
        
        centered = ColoredGrid(values=[[0 for _ in range(new_cols)] for _ in range(new_rows)])
        for r in range(pattern_rows):
            for c in range(pattern_cols):
                centered.set_cell(r + vertical_padding, c + horizontal_padding, pattern.get_cell(r, c))
        
        return centered

    # Main logic
    black_regions = find_black_regions(input_grid)
    if not black_regions:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if no regions found
    
    most_significant_region = max(black_regions, key=lambda region: calculate_significance(input_grid, region))
    extracted_pattern = extract_region(input_grid, most_significant_region)
    centered_pattern = center_pattern(extracted_pattern)
    
    return centered_pattern
