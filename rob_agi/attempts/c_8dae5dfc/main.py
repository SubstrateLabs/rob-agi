from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_8dae5dfc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rotating colors within connected regions.
    
    The transformation process:
    1. Identifies connected regions of non-black cells.
    2. For each region, determines the unique colors from outer to inner.
    3. Rotates the colors of each region:
       - The second-to-outermost color becomes the new outermost color.
       - The outermost color moves to the second-to-innermost position.
       - All other colors shift outward by one position.
    4. Applies the rotated colors to the corresponding cells in the new grid.
    
    Black (0) cells, representing empty space, remain unchanged.
    The overall structure and position of shapes are maintained.
    """
    def get_adjacent_cells(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def find_connected_region(start_r: int, start_c: int) -> List[Tuple[int, int]]:
        color = input_grid.values[start_r][start_c]
        region = []
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and input_grid.values[r][c] == color:
                visited.add((r, c))
                region.append((r, c))
                for nr, nc in get_adjacent_cells(r, c):
                    if is_valid_cell(nr, nc) and (nr, nc) not in visited and input_grid.values[nr][nc] != 0:
                        stack.append((nr, nc))
        return region

    def determine_unique_colors(region: List[Tuple[int, int]]) -> List[int]:
        color_order = []
        seen_colors = set()
        for r, c in region:
            color = input_grid.values[r][c]
            if color not in seen_colors:
                color_order.append(color)
                seen_colors.add(color)
        return color_order

    def rotate_colors(colors: List[int]) -> Dict[int, int]:
        n = len(colors)
        if n <= 1:
            return {colors[0]: colors[0]}
        rotated = colors[1:] + [colors[0]]
        return dict(zip(colors, rotated))

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    visited: Set[Tuple[int, int]] = set()

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and (r, c) not in visited:
                region = find_connected_region(r, c)
                unique_colors = determine_unique_colors(region)
                color_mapping = rotate_colors(unique_colors)
                for cell_r, cell_c in region:
                    new_grid[cell_r][cell_c] = color_mapping[input_grid.values[cell_r][cell_c]]

    return ColoredGrid(values=new_grid)
