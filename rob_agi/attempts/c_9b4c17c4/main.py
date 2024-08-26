from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b4c17c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red (2) regions to the right edge in blue (1) zones
    and to the left edge in sky blue (8) zones, while maintaining their vertical order and shape.
    
    1. Identifies blue and sky blue zones.
    2. Locates red regions within each zone.
    3. Moves red regions to the appropriate edge based on the zone color.
    4. Stacks overlapping regions vertically.
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
        regions.sort(key=lambda x: x[0])  # Sort by top coordinate
        target_col = cols - 1 if to_right else 0
        current_row = zone_start

        for top, _, width, height in regions:
            new_top = max(current_row, top)
            new_bottom = min(new_top + height, zone_end + 1)
            actual_height = new_bottom - new_top

            if to_right:
                for r in range(new_top, new_bottom):
                    for c in range(target_col - width + 1, target_col + 1):
                        output_grid.values[r][c] = 2
                    for c in range(cols):
                        if c < target_col - width + 1 or c > target_col:
                            output_grid.values[r][c] = output_grid.values[r][c] if output_grid.values[r][c] != 2 else 1
            else:
                for r in range(new_top, new_bottom):
                    for c in range(target_col, target_col + width):
                        output_grid.values[r][c] = 2
                    for c in range(width, cols):
                        output_grid.values[r][c] = output_grid.values[r][c] if output_grid.values[r][c] != 2 else 8

            current_row = new_bottom

    zones = find_zones()
    for color, start, end in zones:
        regions = find_red_regions(start, end)
        move_regions(regions, start, end, color == 1)

    return output_grid
