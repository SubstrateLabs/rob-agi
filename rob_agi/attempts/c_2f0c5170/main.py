from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2f0c5170(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most complex pattern from black regions in the input grid,
    centers it in an optimally sized output grid, and returns it as a new ColoredGrid.
    
    1. Identifies all black regions in the input grid
    2. Extracts patterns from each black region, focusing on non-black cells
    3. Analyzes complexity of patterns based on non-black cell count, color variety, and spread
    4. Selects the most complex pattern
    5. Determines optimal output grid size based on the pattern
    6. Centers the pattern in the new grid with a minimal black border
    7. Optimizes the final grid size
    8. Returns the result as a ColoredGrid, or a 5x5 black grid if no pattern is found
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

    def extract_pattern(grid: ColoredGrid, region: List[Tuple[int, int]]) -> Tuple[ColoredGrid, int, int, int]:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        
        pattern = ColoredGrid(values=[[0 for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)])
        non_black_cells = 0
        unique_colors = set()
        max_distance = 0
        
        for r, c in region:
            color = grid.get_cell(r, c)
            pattern.set_cell(r - min_r, c - min_c, color)
            if color != 0:
                non_black_cells += 1
                unique_colors.add(color)
                max_distance = max(max_distance, r - min_r, c - min_c)
        
        return pattern, non_black_cells, len(unique_colors), max_distance

    def determine_output_size(pattern: ColoredGrid, max_distance: int) -> int:
        rows, cols = pattern.get_dimensions()
        size = max(rows, cols, max_distance * 2 + 3, 5)  # Ensure minimum size of 5x5 and space for pattern
        return size + 1 if size % 2 == 0 else size  # Ensure odd dimensions for perfect centering

    def center_pattern(pattern: ColoredGrid, output_size: int) -> ColoredGrid:
        pattern_rows, pattern_cols = pattern.get_dimensions()
        vertical_padding = (output_size - pattern_rows) // 2
        horizontal_padding = (output_size - pattern_cols) // 2
        
        centered = ColoredGrid(values=[[0 for _ in range(output_size)] for _ in range(output_size)])
        for r in range(pattern_rows):
            for c in range(pattern_cols):
                color = pattern.get_cell(r, c)
                centered.set_cell(r + vertical_padding, c + horizontal_padding, color)
        
        return centered

    def optimize_grid_size(grid: ColoredGrid) -> ColoredGrid:
        rows, cols = grid.get_dimensions()
        top = next((r for r in range(rows) if any(grid.get_cell(r, c) != 0 for c in range(cols))), 0)
        bottom = next((r for r in range(rows - 1, -1, -1) if any(grid.get_cell(r, c) != 0 for c in range(cols))), rows - 1)
        left = next((c for c in range(cols) if any(grid.get_cell(r, c) != 0 for r in range(rows))), 0)
        right = next((c for c in range(cols - 1, -1, -1) if any(grid.get_cell(r, c) != 0 for r in range(rows))), cols - 1)
        
        # Ensure at least one cell of black border
        top, left = max(0, top - 1), max(0, left - 1)
        bottom, right = min(rows - 1, bottom + 1), min(cols - 1, right + 1)
        
        optimized = grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
        
        # Ensure odd dimensions and minimum size of 5x5
        new_rows, new_cols = max(5, optimized.num_rows), max(5, optimized.num_cols)
        if new_rows % 2 == 0:
            new_rows += 1
        if new_cols % 2 == 0:
            new_cols += 1
        
        return optimized.expand((new_rows - optimized.num_rows) // 2,
                                (new_cols - optimized.num_cols) // 2,
                                (new_rows - optimized.num_rows + 1) // 2,
                                (new_cols - optimized.num_cols + 1) // 2)

    # Main logic
    black_regions = find_black_regions(input_grid)
    if not black_regions:
        return ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])  # Return a 5x5 black grid if no regions found
    
    patterns = []
    for region in black_regions:
        pattern, non_black_cells, unique_colors, max_distance = extract_pattern(input_grid, region)
        complexity_score = non_black_cells * unique_colors * max_distance
        patterns.append((pattern, complexity_score, max_distance))
    
    if not patterns:
        return ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])  # Return a 5x5 black grid if no patterns found
    
    # Choose the most complex pattern
    chosen_pattern, _, max_distance = max(patterns, key=lambda x: x[1])
    
    output_size = determine_output_size(chosen_pattern, max_distance)
    centered_pattern = center_pattern(chosen_pattern, output_size)
    optimized_pattern = optimize_grid_size(centered_pattern)
    
    return optimized_pattern
    
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

    def extract_pattern(grid: ColoredGrid, region: List[Tuple[int, int]]) -> Tuple[ColoredGrid, int, int, int]:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        
        pattern = ColoredGrid(values=[[0 for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)])
        non_black_cells = 0
        unique_colors = set()
        max_distance = 0
        
        for r, c in region:
            color = grid.get_cell(r, c)
            if color != 0:
                pattern.set_cell(r - min_r, c - min_c, color)
                non_black_cells += 1
                unique_colors.add(color)
                max_distance = max(max_distance, r - min_r, c - min_c)
        
        return pattern, non_black_cells, len(unique_colors), max_distance

    def determine_output_size(pattern: ColoredGrid, max_distance: int) -> Tuple[int, int]:
        rows, cols = pattern.get_dimensions()
        size = max(rows, cols, max_distance * 2 + 3, 5)  # Ensure minimum size of 5x5 and space for pattern
        
        # Ensure odd dimensions for perfect centering
        size = size + 1 if size % 2 == 0 else size
        
        return size, size

    def center_pattern(pattern: ColoredGrid, output_size: int) -> ColoredGrid:
        pattern_rows, pattern_cols = pattern.get_dimensions()
        vertical_padding = (output_size - pattern_rows) // 2
        horizontal_padding = (output_size - pattern_cols) // 2
        
        centered = ColoredGrid(values=[[0 for _ in range(output_size)] for _ in range(output_size)])
        for r in range(pattern_rows):
            for c in range(pattern_cols):
                color = pattern.get_cell(r, c)
                if color != 0:
                    centered.set_cell(r + vertical_padding, c + horizontal_padding, color)
        
        return centered

    def optimize_grid_size(grid: ColoredGrid) -> ColoredGrid:
        rows, cols = grid.get_dimensions()
        try:
            top = next(r for r in range(rows) if any(grid.get_cell(r, c) != 0 for c in range(cols)))
            bottom = next(r for r in range(rows - 1, -1, -1) if any(grid.get_cell(r, c) != 0 for c in range(cols)))
            left = next(c for c in range(cols) if any(grid.get_cell(r, c) != 0 for r in range(rows)))
            right = next(c for c in range(cols - 1, -1, -1) if any(grid.get_cell(r, c) != 0 for r in range(rows)))
        except StopIteration:
            # If no non-black cells are found, return a 5x5 black grid
            return ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])
        
        # Ensure at least one cell of black border
        top, left = max(0, top - 1), max(0, left - 1)
        bottom, right = min(rows - 1, bottom + 1), min(cols - 1, right + 1)
        
        optimized = grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
        
        # Ensure odd dimensions and minimum size of 5x5
        new_rows, new_cols = max(5, optimized.num_rows), max(5, optimized.num_cols)
        if new_rows % 2 == 0:
            new_rows += 1
        if new_cols % 2 == 0:
            new_cols += 1
        
        return optimized.expand((new_rows - optimized.num_rows) // 2,
                                (new_cols - optimized.num_cols) // 2,
                                (new_rows - optimized.num_rows + 1) // 2,
                                (new_cols - optimized.num_cols + 1) // 2)

    # Main logic
    black_regions = find_black_regions(input_grid)
    if not black_regions:
        return ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])  # Return a 5x5 black grid if no regions found
    
    patterns = []
    for region in black_regions:
        pattern, non_black_cells, unique_colors, max_distance = extract_pattern(input_grid, region)
        complexity_score = non_black_cells * unique_colors * max_distance
        patterns.append((pattern, complexity_score, max_distance))
    
    if not patterns:
        return ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])  # Return a 5x5 black grid if no patterns found
    
    # Choose the most complex pattern
    chosen_pattern, _, max_distance = max(patterns, key=lambda x: x[1])
    
    output_size = determine_output_size(chosen_pattern, max_distance)
    centered_pattern = center_pattern(chosen_pattern, output_size[0])
    optimized_pattern = optimize_grid_size(centered_pattern)
    
    return optimized_pattern
