from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_ea959feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the correct repeating pattern
    and applying it consistently across the entire grid, while preserving significant blue regions.
    
    The solution works as follows:
    1. Identifies the correct pattern by analyzing non-blue cells in the input grid.
    2. Detects significant blue regions using a flood-fill algorithm.
    3. Creates a mask to determine which cells should follow the pattern and which should remain blue.
    4. Applies the pattern to non-masked cells and preserves blue cells in masked regions.
    5. Returns a new ColoredGrid with the correct pattern applied and significant blue regions preserved.
    
    This approach works for all cases by identifying the underlying pattern
    and replicating it across the grid while maintaining important blue structures.
    """
    def identify_pattern(grid: List[List[int]]) -> Tuple[List[List[int]], int, int]:
        rows, cols = len(grid), len(grid[0])
        for pattern_height in range(1, rows + 1):
            for pattern_width in range(1, cols + 1):
                pattern = [row[:pattern_width] for row in grid[:pattern_height]]
                if all(grid[r][c] == pattern[r % pattern_height][c % pattern_width]
                       for r in range(rows) for c in range(cols)
                       if grid[r][c] != 1):  # Ignore blue cells (1) when checking pattern
                    return pattern, pattern_height, pattern_width
        raise ValueError("Could not identify a consistent pattern")

    def find_blue_regions(grid: List[List[int]], threshold: int) -> Set[Tuple[int, int]]:
        rows, cols = len(grid), len(grid[0])
        visited = set()
        blue_regions = set()

        def dfs(r: int, c: int) -> Set[Tuple[int, int]]:
            stack = [(r, c)]
            region = set()
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and grid[curr_r][curr_c] == 1:
                    visited.add((curr_r, curr_c))
                    region.add((curr_r, curr_c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_r, new_c = curr_r + dr, curr_c + dc
                        if 0 <= new_r < rows and 0 <= new_c < cols:
                            stack.append((new_r, new_c))
            return region if len(region) >= threshold else set()

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visited:
                    blue_regions.update(dfs(r, c))

        return blue_regions

    pattern, pattern_height, pattern_width = identify_pattern(input_grid.values)
    blue_regions = find_blue_regions(input_grid.values, threshold=4)
    
    def apply_pattern(row: int, col: int) -> int:
        if (row, col) in blue_regions:
            return 1  # Preserve blue cells in significant regions
        return pattern[row % pattern_height][col % pattern_width]
    
    rows, cols = input_grid.get_dimensions()
    corrected_values = [
        [apply_pattern(i, j) for j in range(cols)]
        for i in range(rows)
    ]
    
    return ColoredGrid(values=corrected_values)
