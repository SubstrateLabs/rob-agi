from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b4c17c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red (2) regions to the right edge in blue (1) zones
    and to the left edge in sky blue (8) zones, while maintaining their vertical position and shape.
    
    1. Identifies vertical blue and sky blue zones.
    2. Locates red regions within each zone.
    3. Moves red regions to the appropriate edge based on the zone color.
    4. Preserves the vertical position and shape of red regions.
    5. Maintains proper spacing between regions and zone edges.
    6. Handles cases where regions might already be at the correct edge.
    7. Ensures that no red region extends beyond the zone boundaries.
    8. Processes each vertical zone independently.
    9. Maintains the relative order of multiple regions within a zone.
    10. Ensures regions move as close to the edge as possible while maintaining proper spacing.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def find_vertical_zones() -> List[Tuple[int, int, int]]:
        zones = []
        start_col = 0
        current_color = input_grid.values[0][0]
        for c in range(1, cols):
            if input_grid.values[0][c] != current_color:
                zones.append((current_color, start_col, c - 1))
                start_col = c
                current_color = input_grid.values[0][c]
        zones.append((current_color, start_col, cols - 1))
        return zones

    def find_red_regions(start_col: int, end_col: int) -> List[List[Tuple[int, int]]]:
        regions = []
        visited = set()

        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            if (r, c) in visited or r < 0 or r >= rows or c < start_col or c > end_col or input_grid.values[r][c] != 2:
                return []
            visited.add((r, c))
            region = [(r, c)]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                region.extend(dfs(r + dr, c + dc))
            return region

        for r in range(rows):
            for c in range(start_col, end_col + 1):
                if input_grid.values[r][c] == 2 and (r, c) not in visited:
                    regions.append(dfs(r, c))

        return regions

    def move_regions(regions: List[List[Tuple[int, int]]], start_col: int, end_col: int, to_right: bool):
        regions.sort(key=lambda r: min(y for y, _ in r))  # Sort from top to bottom

        if to_right:
            new_start = end_col
            for region in regions:
                region_width = max(c for _, c in region) - min(c for _, c in region) + 1
                new_start = min(new_start, end_col - region_width + 1)  # Ensure region fits within zone
                offset = new_start - min(c for _, c in region)
                for r, c in region:
                    output_grid.values[r][c] = input_grid.values[r][start_col]  # Restore original background
                    output_grid.values[r][c + offset] = 2
                new_start = new_start - 1  # Move to the next available position
        else:
            new_start = start_col
            for region in regions:
                region_width = max(c for _, c in region) - min(c for _, c in region) + 1
                offset = new_start - min(c for _, c in region)
                for r, c in region:
                    output_grid.values[r][c] = input_grid.values[r][start_col]  # Restore original background
                    output_grid.values[r][c + offset] = 2
                new_start = new_start + region_width  # Move to the next available position

    vertical_zones = find_vertical_zones()
    for color, start_col, end_col in vertical_zones:
        if color in [1, 8]:  # Only process blue and sky blue zones
            regions = find_red_regions(start_col, end_col)
            move_regions(regions, start_col, end_col, color == 1)

    return output_grid
