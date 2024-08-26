from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9bb6954(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. For each color from 1 to 9:
       a. Find all connected regions of the current color.
       b. Identify the largest region.
    2. For the largest region of each color:
       a. Calculate the center of mass.
       b. Determine whether to draw a horizontal or vertical line based on the region's shape.
       c. Draw the line through the center of mass, respecting color precedence.
    3. Process colors in ascending order to ensure proper precedence.
    4. Return the transformed grid with these lines drawn.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def find_largest_region(color: int) -> List[Tuple[int, int]]:
        regions = input_grid.find_connected_regions(color)
        return max(regions, key=len) if regions else []

    def calculate_center_of_mass(region: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not region:
            return (0, 0)
        avg_row = sum(r for r, _ in region) // len(region)
        avg_col = sum(c for _, c in region) // len(region)
        return (avg_row, avg_col)

    def determine_line_direction(region: List[Tuple[int, int]]) -> str:
        if not region:
            return "horizontal"
        min_row, max_row = min(r for r, _ in region), max(r for r, _ in region)
        min_col, max_col = min(c for _, c in region), max(c for _, c in region)
        height = max_row - min_row
        width = max_col - min_col
        return "vertical" if height > width else "horizontal"

    def draw_line(color: int, center: Tuple[int, int], direction: str):
        if direction == "horizontal":
            for c in range(cols):
                if output_grid.get_cell(center[0], c) < color:
                    output_grid.set_cell(center[0], c, color)
        else:
            for r in range(rows):
                if output_grid.get_cell(r, center[1]) < color:
                    output_grid.set_cell(r, center[1], color)

    for color in range(1, 10):  # Colors 1 to 9
        largest_region = find_largest_region(color)
        if largest_region:
            center = calculate_center_of_mass(largest_region)
            direction = determine_line_direction(largest_region)
            draw_line(color, center, direction)

    return output_grid
