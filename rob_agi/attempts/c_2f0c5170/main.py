from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2f0c5170(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most complex pattern from black regions in the input grid,
    centers it in a reasonably sized output grid, and returns it as a new ColoredGrid.
    
    1. Identifies all black regions in the input grid
    2. Analyzes each region for complexity based on non-black cells and unique colors
    3. Selects the most complex region
    4. Extracts and trims the chosen region
    5. Determines an appropriate output grid size
    6. Centers the pattern in the new grid
    7. Returns the result as a ColoredGrid
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

    def calculate_complexity(grid: ColoredGrid, region: List[Tuple[int, int]]) -> int:
        non_black = set()
        unique_colors = set()
        for r, c in region:
            color = grid.get_cell(r, c)
            if color != 0:
                non_black.add((r, c))
                unique_colors.add(color)
        return len(non_black) * 2 + len(unique_colors) * 3

    def extract_and_trim_region(grid: ColoredGrid, region: List[Tuple[int, int]]) -> ColoredGrid:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        subgrid = grid.extract_subgrid(min_r, min_c, max_r - min_r + 1, max_c - min_c + 1)
        
        # Trim black edges
        rows, cols = subgrid.get_dimensions()
        top = next(r for r in range(rows) if any(subgrid.get_cell(r, c) != 0 for c in range(cols)))
        bottom = next(r for r in range(rows - 1, -1, -1) if any(subgrid.get_cell(r, c) != 0 for c in range(cols)))
        left = next(c for c in range(cols) if any(subgrid.get_cell(r, c) != 0 for r in range(rows)))
        right = next(c for c in range(cols - 1, -1, -1) if any(subgrid.get_cell(r, c) != 0 for r in range(rows)))
        
        return subgrid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)

    def determine_output_size(pattern: ColoredGrid) -> Tuple[int, int]:
        rows, cols = pattern.get_dimensions()
        max_dim = max(rows, cols)
        min_dim = min(rows, cols)
        target_max = min(max_dim + 2, int(min_dim * 1.5) + 2)  # Add padding and limit aspect ratio
        if rows > cols:
            return target_max, max(cols + 2, (target_max * cols) // rows)
        else:
            return max(rows + 2, (target_max * rows) // cols), target_max

    def center_pattern(pattern: ColoredGrid, output_size: Tuple[int, int]) -> ColoredGrid:
        output_rows, output_cols = output_size
        pattern_rows, pattern_cols = pattern.get_dimensions()
        vertical_padding = (output_rows - pattern_rows) // 2
        horizontal_padding = (output_cols - pattern_cols) // 2
        
        centered = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])
        for r in range(pattern_rows):
            for c in range(pattern_cols):
                centered.set_cell(r + vertical_padding, c + horizontal_padding, pattern.get_cell(r, c))
        
        return centered

    # Main logic
    black_regions = find_black_regions(input_grid)
    if not black_regions:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if no regions found
    
    most_complex_region = max(black_regions, key=lambda region: calculate_complexity(input_grid, region))
    extracted_pattern = extract_and_trim_region(input_grid, most_complex_region)
    output_size = determine_output_size(extracted_pattern)
    centered_pattern = center_pattern(extracted_pattern, output_size)
    
    return centered_pattern
