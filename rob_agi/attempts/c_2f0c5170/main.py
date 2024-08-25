from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2f0c5170(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most complex pattern from two black regions in the input grid,
    centers it in a reasonably sized output grid, and returns it as a new ColoredGrid.
    
    1. Identifies the two black regions in the input grid
    2. Extracts patterns from both black regions
    3. Compares the complexity of the two patterns based on non-black cells and color variety
    4. Selects the most complex pattern
    5. Trims the chosen pattern and ensures a minimal black border
    6. Determines an appropriate output grid size
    7. Centers the pattern in the new grid
    8. Returns the result as a ColoredGrid
    """
    
    def find_two_black_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
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
                    if len(regions) == 2:
                        return regions
        
        return regions

    def extract_pattern(grid: ColoredGrid, region: List[Tuple[int, int]]) -> ColoredGrid:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        
        pattern = ColoredGrid(values=[[0 for _ in range(max_c - min_c + 3)] for _ in range(max_r - min_r + 3)])
        non_black_cells = 0
        unique_colors = set()
        
        for r in range(min_r - 1, max_r + 2):
            for c in range(min_c - 1, max_c + 2):
                if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                    color = grid.get_cell(r, c)
                    pattern.set_cell(r - min_r + 1, c - min_c + 1, color)
                    if color != 0:
                        non_black_cells += 1
                        unique_colors.add(color)
        
        return pattern, non_black_cells, len(unique_colors)

    def trim_pattern(pattern: ColoredGrid) -> ColoredGrid:
        rows, cols = pattern.get_dimensions()
        top = next(r for r in range(rows) if any(pattern.get_cell(r, c) != 0 for c in range(cols)))
        bottom = next(r for r in range(rows - 1, -1, -1) if any(pattern.get_cell(r, c) != 0 for c in range(cols)))
        left = next(c for c in range(cols) if any(pattern.get_cell(r, c) != 0 for r in range(rows)))
        right = next(c for c in range(cols - 1, -1, -1) if any(pattern.get_cell(r, c) != 0 for r in range(rows)))
        
        return pattern.extract_subgrid(top, left, bottom - top + 1, right - left + 1)

    def determine_output_size(pattern: ColoredGrid) -> Tuple[int, int]:
        rows, cols = pattern.get_dimensions()
        max_dim = max(rows, cols)
        min_dim = min(rows, cols)
        
        # Ensure odd dimensions for perfect centering
        rows = rows + 1 if rows % 2 == 0 else rows
        cols = cols + 1 if cols % 2 == 0 else cols
        
        # Ensure minimum size of 5x5
        rows = max(rows, 5)
        cols = max(cols, 5)
        
        # Adjust aspect ratio if necessary
        if rows > cols * 1.5:
            cols = max(cols, rows * 2 // 3)
        elif cols > rows * 1.5:
            rows = max(rows, cols * 2 // 3)
        
        return rows, cols

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
    black_regions = find_two_black_regions(input_grid)
    if len(black_regions) < 2:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if fewer than 2 regions found
    
    patterns = []
    for region in black_regions[:2]:
        pattern, non_black_cells, unique_colors = extract_pattern(input_grid, region)
        patterns.append((pattern, non_black_cells, unique_colors))
    
    # Choose the most complex pattern
    chosen_pattern = max(patterns, key=lambda x: (x[1], x[2]))[0]
    
    trimmed_pattern = trim_pattern(chosen_pattern)
    output_size = determine_output_size(trimmed_pattern)
    centered_pattern = center_pattern(trimmed_pattern, output_size)
    
    return centered_pattern
