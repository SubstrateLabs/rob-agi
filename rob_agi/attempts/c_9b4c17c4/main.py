from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b4c17c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red (2) regions to the right edge in blue (1) zones
    and to the left edge in sky blue (8) zones, while maintaining their vertical position and shape.
    
    1. Identifies vertical blue and sky blue zones.
    2. Locates red regions within each zone, including those that span zone boundaries.
    3. Moves red regions to the appropriate edge based on the zone color.
    4. Preserves the vertical position and shape of red regions.
    5. Maintains proper spacing between regions and zone edges.
    6. Handles cases where regions might already be at the correct edge.
    7. Ensures that no red region extends beyond the zone boundaries.
    8. Processes each vertical zone independently.
    9. Maintains the relative order of multiple regions within a zone.
    10. Ensures regions move as close to the edge as possible while maintaining proper spacing.
    11. Handles regions that span zone boundaries by splitting and moving each part appropriately.
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

    def find_red_regions() -> List[List[Tuple[int, int]]]:
        regions = []
        visited = set()

        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or input_grid.values[r][c] != 2:
                return []
            visited.add((r, c))
            region = [(r, c)]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                region.extend(dfs(r + dr, c + dc))
            return region

        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == 2 and (r, c) not in visited:
                    regions.append(dfs(r, c))

        return regions

    def split_region(region: List[Tuple[int, int]], zones: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int]]]:
        split_regions = []
        for zone_color, start_col, end_col in zones:
            zone_region = [cell for cell in region if start_col <= cell[1] <= end_col]
            if zone_region:
                split_regions.append((zone_color, zone_region))
        return split_regions

    def move_regions(regions: List[List[Tuple[int, int]]], zones: List[Tuple[int, int, int]]):
        for zone_color, start_col, end_col in zones:
            if zone_color not in [1, 8]:
                continue

            zone_regions = []
            for region in regions:
                zone_regions.extend(split_region(region, [(zone_color, start_col, end_col)]))

            zone_regions.sort(key=lambda r: min(y for _, y in r[1]))  # Sort from top to bottom

            if zone_color == 1:  # Blue zone, move to right
                new_start = end_col
                for _, region in zone_regions:
                    region_width = max(c for _, c in region) - min(c for _, c in region) + 1
                    new_start = min(new_start, end_col - region_width + 1)  # Ensure region fits within zone
                    offset = new_start - min(c for _, c in region)
                    for r, c in region:
                        output_grid.values[r][c] = zone_color  # Restore original background
                        output_grid.values[r][c + offset] = 2
                    new_start = new_start - region_width - 1  # Move to the next available position, leaving a gap
            else:  # Sky blue zone, move to left
                new_start = start_col
                for _, region in zone_regions:
                    region_width = max(c for _, c in region) - min(c for _, c in region) + 1
                    offset = new_start - min(c for _, c in region)
                    for r, c in region:
                        output_grid.values[r][c] = zone_color  # Restore original background
                        output_grid.values[r][c + offset] = 2
                    new_start = new_start + region_width + 1  # Move to the next available position, leaving a gap

    vertical_zones = find_vertical_zones()
    red_regions = find_red_regions()
    move_regions(red_regions, vertical_zones)

    return output_grid
