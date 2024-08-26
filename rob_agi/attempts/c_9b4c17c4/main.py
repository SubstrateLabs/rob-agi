from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b4c17c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red (2) regions to the right edge in blue (1) zones
    and to the left edge in sky blue (8) zones, while maintaining their vertical spacing and shape.
    
    1. Identifies blue and sky blue zones.
    2. Locates red regions within each zone.
    3. Moves red regions to the appropriate edge based on the zone color.
    4. Preserves vertical spacing between red regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def find_zones() -> List[Tuple[int, int, int]]:
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

    def find_red_regions(start_row: int, end_row: int) -> List[Tuple[int, int, int, int]]:
        regions = []
        for r in range(start_row, end_row + 1):
            c = 0
            while c < cols:
                if output_grid.values[r][c] == 2:
                    width = 1
                    while c + width < cols and output_grid.values[r][c + width] == 2:
                        width += 1
                    height = 1
                    while r + height <= end_row and all(output_grid.values[r + height][c + w] == 2 for w in range(width)):
                        height += 1
                    regions.append((r, c, width, height))
                    c += width
                else:
                    c += 1
        return regions

    def move_regions(regions: List[Tuple[int, int, int, int]], zone_start: int, zone_end: int, to_right: bool):
        target_col = cols - 1 if to_right else 0
        zone_height = zone_end - zone_start + 1
        total_red_height = sum(height for _, _, _, height in regions)
        spacing = max(0, (zone_height - total_red_height) // (len(regions) + 1))

        current_row = zone_start + spacing
        for top, _, width, height in regions:
            if to_right:
                for r in range(current_row, current_row + height):
                    for c in range(cols - width, cols):
                        output_grid.values[r][c] = 2
                    for c in range(cols - width):
                        output_grid.values[r][c] = 1
            else:
                for r in range(current_row, current_row + height):
                    for c in range(width):
                        output_grid.values[r][c] = 2
                    for c in range(width, cols):
                        output_grid.values[r][c] = 8
            
            current_row += height + spacing

    zones = find_zones()
    for color, start, end in zones:
        # Clear the zone
        for r in range(start, end + 1):
            for c in range(cols):
                output_grid.values[r][c] = color
        
        regions = find_red_regions(start, end)
        move_regions(regions, start, end, color == 1)

    return output_grid
