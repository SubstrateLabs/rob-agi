from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_84f2aca1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the centers of shapes with specific colors.
    
    The function identifies all shapes in the grid and fills their centers based on the following rules:
    - For shapes with odd dimensions (including 3x3): Fill the single center cell with gray (5)
    - For shapes with even dimensions in either width or height: Fill a 2x2 or 1x2 or 2x1 area in the center with orange (7)
    
    The function uses a flood fill algorithm to identify connected regions of the same color,
    calculates the dimensions of each shape, and applies the appropriate fill.
    Only cells that were part of the original shape are filled.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    processed: Set[Tuple[int, int]] = set()

    def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        shape = []
        queue = [(r, c)]
        while queue:
            curr_r, curr_c = queue.pop(0)
            if (curr_r, curr_c) in processed:
                continue
            if 0 <= curr_r < rows and 0 <= curr_c < cols and output_grid.get_cell(curr_r, curr_c) == color:
                shape.append((curr_r, curr_c))
                processed.add((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((curr_r + dr, curr_c + dc))
        return shape

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in processed and output_grid.get_cell(r, c) != 0:
                shape = flood_fill(r, c, output_grid.get_cell(r, c))
                if shape:
                    min_r = min(coord[0] for coord in shape)
                    max_r = max(coord[0] for coord in shape)
                    min_c = min(coord[1] for coord in shape)
                    max_c = max(coord[1] for coord in shape)
                    
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1
                    
                    center_r = (min_r + max_r) // 2
                    center_c = (min_c + max_c) // 2
                    
                    if height % 2 == 1 and width % 2 == 1:
                        # Odd dimensions: fill single center with gray
                        if (center_r, center_c) in shape:
                            output_grid.set_cell(center_r, center_c, 5)
                    else:
                        # Even dimensions in at least one direction: fill with orange
                        fill_height = 2 if height % 2 == 0 else 1
                        fill_width = 2 if width % 2 == 0 else 1
                        fill_start_r = center_r - (fill_height // 2)
                        fill_start_c = center_c - (fill_width // 2)
                        for dr in range(fill_height):
                            for dc in range(fill_width):
                                if (fill_start_r + dr, fill_start_c + dc) in shape:
                                    output_grid.set_cell(fill_start_r + dr, fill_start_c + dc, 7)

    return output_grid
