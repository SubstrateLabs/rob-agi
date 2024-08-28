from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b4c17c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red (2) regions to the right edge in blue (1) zones
    and to the left edge in sky blue (8) zones, while maintaining their vertical position and shape.
    
    1. Identifies horizontal blue and sky blue zones.
    2. Locates red regions within each zone.
    3. Moves red regions to the appropriate edge based on the zone color.
    4. Preserves the vertical position and shape of red regions.
    5. Maintains proper spacing between regions and zone edges.
    6. Processes each horizontal zone independently.
    7. Maintains the relative order of multiple regions within a zone.
    8. Ensures regions move as close to the edge as possible while maintaining proper spacing.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def find_horizontal_zones() -> List[Tuple[int, int, int]]:
        zones = []
        start_row = 0
        current_color = input_grid.values[0][0]
        for r in range(1, rows):
            if input_grid.values[r][0] != current_color:
                zones.append((current_color, start_row, r - 1))
                start_row = r
                current_color = input_grid.values[r][0]
        zones.append((current_color, start_row, rows - 1))
        return zones

    def find_red_regions(start_row: int, end_row: int) -> List[List[Tuple[int, int]]]:
        regions = []
        for r in range(start_row, end_row + 1):
            region = []
            for c in range(cols):
                if output_grid.values[r][c] == 2:
                    region.append((r, c))
                elif region:
                    regions.append(region)
                    region = []
            if region:
                regions.append(region)
        return regions

    def move_regions(zone_color: int, start_row: int, end_row: int):
        regions = find_red_regions(start_row, end_row)
        if zone_color == 1:  # Blue zone, move to right
            new_start = cols - 1
            for region in reversed(regions):
                region_width = len(region)
                new_start = min(new_start, cols - region_width)
                for r, c in region:
                    output_grid.values[r][c] = zone_color
                    output_grid.values[r][new_start + (c - region[0][1])] = 2
                new_start -= 1  # Leave a gap
        else:  # Sky blue zone, move to left
            new_start = 0
            for region in regions:
                region_width = len(region)
                for r, c in region:
                    output_grid.values[r][c] = zone_color
                    output_grid.values[r][new_start + (c - region[0][1])] = 2
                new_start += region_width + 1  # Leave a gap

    horizontal_zones = find_horizontal_zones()
    for zone_color, start_row, end_row in horizontal_zones:
        if zone_color in [1, 8]:
            move_regions(zone_color, start_row, end_row)

    return output_grid
