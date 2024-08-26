from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_84f2aca1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the holes in shapes with specific colors.
    
    The function identifies all shapes in the grid and fills their holes based on the following rules:
    - For shapes with a single-cell hole: Fill the hole with gray (5)
    - For shapes with larger holes (2x2, 2x1, or 1x2): Fill the hole with orange (7)
    
    The function uses a flood fill algorithm to identify connected regions of the same color,
    calculates the dimensions of each shape, identifies holes, and applies the appropriate fill.
    Only cells that were originally empty (color 0) and surrounded by the shape are filled.
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

    def find_hole(shape: List[Tuple[int, int]], min_r: int, max_r: int, min_c: int, max_c: int) -> List[Tuple[int, int]]:
        hole = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in shape and output_grid.get_cell(r, c) == 0:
                    hole.append((r, c))
        return hole

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in processed and output_grid.get_cell(r, c) != 0:
                shape = flood_fill(r, c, output_grid.get_cell(r, c))
                if shape:
                    min_r = min(coord[0] for coord in shape)
                    max_r = max(coord[0] for coord in shape)
                    min_c = min(coord[1] for coord in shape)
                    max_c = max(coord[1] for coord in shape)
                    
                    hole = find_hole(shape, min_r, max_r, min_c, max_c)
                    
                    if len(hole) == 1:
                        # Single-cell hole: fill with gray
                        hr, hc = hole[0]
                        output_grid.set_cell(hr, hc, 5)
                    elif len(hole) > 1:
                        # Larger hole: fill with orange
                        for hr, hc in hole:
                            output_grid.set_cell(hr, hc, 7)

    return output_grid
