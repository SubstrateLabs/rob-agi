from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b4c17c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red (2) regions to the right edge in blue (1) zones
    and to the left edge in sky blue (8) zones, while maintaining their vertical spacing and shape.
    
    1. Identifies vertical blue and sky blue zones.
    2. Locates red regions within each zone.
    3. Moves red regions to the appropriate edge based on the zone color.
    4. Preserves vertical spacing between red regions within each column.
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

    def find_red_regions_in_column(col: int) -> List[Tuple[int, int]]:
        regions = []
        start_row = None
        for r in range(rows):
            if output_grid.values[r][col] == 2:
                if start_row is None:
                    start_row = r
            elif start_row is not None:
                regions.append((start_row, r - 1))
                start_row = None
        if start_row is not None:
            regions.append((start_row, rows - 1))
        return regions

    def move_regions_in_column(regions: List[Tuple[int, int]], col: int, to_bottom: bool):
        zone_color = 1 if to_bottom else 8
        total_red_height = sum(end - start + 1 for start, end in regions)
        free_space = rows - total_red_height
        spacing = free_space // (len(regions) + 1)

        new_positions = []
        current_row = 0 if not to_bottom else rows - 1

        for start, end in regions:
            height = end - start + 1
            if to_bottom:
                new_start = current_row - height + 1
                new_positions.append((new_start, current_row))
                current_row = new_start - spacing - 1
            else:
                new_positions.append((current_row, current_row + height - 1))
                current_row = current_row + height + spacing

        # Clear the column and place red regions
        for r in range(rows):
            output_grid.values[r][col] = zone_color

        for (old_start, old_end), (new_start, new_end) in zip(regions, new_positions):
            for r in range(new_start, new_end + 1):
                output_grid.values[r][col] = 2

    vertical_zones = find_vertical_zones()
    for color, start_col, end_col in vertical_zones:
        for col in range(start_col, end_col + 1):
            regions = find_red_regions_in_column(col)
            move_regions_in_column(regions, col, color == 1)

    return output_grid
