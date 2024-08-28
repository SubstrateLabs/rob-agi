from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating an inner diagonal skeleton of sky blue (8) regions with red (2).
    
    This function applies the following steps:
    1. Identify all sky blue (8) regions in the input grid.
    2. For each region, determine its type (single cell, straight line, 2x2 square, L-shape, or complex shape).
    3. Process each region according to its type, marking appropriate cells as red (2) in the output grid.
    4. Ensure the resulting red markings form a minimal inner diagonal skeleton of the original sky blue regions.
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions on a black background.
    
    Returns:
    ColoredGrid: A new grid with red markings representing the inner diagonal skeleton of the original sky blue regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def is_inner_corner(r: int, c: int) -> bool:
        if input_grid.values[r][c] != 8:
            return False
        adjacent_black = sum(1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols and input_grid.values[r+dr][c+dc] == 0)
        return adjacent_black >= 2

    def find_inner_corners(region: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [cell for cell in region if is_inner_corner(*cell)]

    def process_single_cell(cell: Tuple[int, int]) -> None:
        output_grid.values[cell[0]][cell[1]] = 2

    def process_straight_line(region: List[Tuple[int, int]]) -> None:
        if len(region) <= 3:
            mid = len(region) // 2
            output_grid.values[region[mid][0]][region[mid][1]] = 2
        else:
            third = len(region) // 3
            output_grid.values[region[third][0]][region[third][1]] = 2
            output_grid.values[region[-third-1][0]][region[-third-1][1]] = 2

    def process_2x2_square(region: List[Tuple[int, int]]) -> None:
        bottom_right = max(region, key=lambda x: x[0] + x[1])
        output_grid.values[bottom_right[0]][bottom_right[1]] = 2

    def process_l_shape(region: List[Tuple[int, int]]) -> None:
        corners = find_inner_corners(region)
        if corners:
            r, c = corners[0]
            output_grid.values[r][c] = 2
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                if 0 <= r+dr < rows and 0 <= c+dc < cols and input_grid.values[r+dr][c+dc] == 8:
                    third = len([cell for cell in region if cell[0] == r+dr or cell[1] == c+dc]) // 3
                    cells = [cell for cell in region if cell[0] == r+dr or cell[1] == c+dc]
                    if cells:
                        output_grid.values[cells[third][0]][cells[third][1]] = 2

    def process_complex_shape(region: List[Tuple[int, int]]) -> None:
        corners = find_inner_corners(region)
        if len(corners) == 2:
            r1, c1 = corners[0]
            r2, c2 = corners[1]
            output_grid.values[r1][c1] = output_grid.values[r2][c2] = 2
            if abs(r1 - r2) > 1 or abs(c1 - c2) > 1:
                mid_r, mid_c = (r1 + r2) // 2, (c1 + c2) // 2
                if input_grid.values[mid_r][mid_c] == 8:
                    output_grid.values[mid_r][mid_c] = 2
        elif len(corners) > 2:
            for i in range(len(corners)):
                r1, c1 = corners[i]
                r2, c2 = corners[(i + 1) % len(corners)]
                output_grid.values[r1][c1] = 2
                if abs(r1 - r2) > 1 or abs(c1 - c2) > 1:
                    mid_r, mid_c = (r1 + r2) // 2, (c1 + c2) // 2
                    if input_grid.values[mid_r][mid_c] == 8:
                        output_grid.values[mid_r][mid_c] = 2

    sky_blue_regions = input_grid.find_connected_regions(8)
    
    for region in sky_blue_regions:
        if len(region) == 1:
            process_single_cell(region[0])
        elif all(cell[0] == region[0][0] for cell in region) or all(cell[1] == region[0][1] for cell in region):
            process_straight_line(region)
        elif len(region) == 4 and len(set(cell[0] for cell in region)) == len(set(cell[1] for cell in region)) == 2:
            process_2x2_square(region)
        elif len(find_inner_corners(region)) == 1:
            process_l_shape(region)
        else:
            process_complex_shape(region)
    
    return output_grid
