from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_8dae5dfc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by shifting colors within connected regions.
    
    The transformation process:
    1. Identifies connected regions of non-black cells.
    2. For each region, determines the color layers from outer to inner.
    3. Shifts the colors of each region:
       - The innermost color moves to the outermost layer.
       - All other colors shift inward by one layer.
    4. Applies the shifted colors to the corresponding cells in the new grid.
    
    Black (0) cells, representing empty space, remain unchanged.
    The overall structure and position of shapes are maintained.
    """
    def get_adjacent_cells(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def is_edge_cell(r: int, c: int) -> bool:
        return any(not is_valid_cell(nr, nc) or input_grid.values[nr][nc] == 0
                   for nr, nc in get_adjacent_cells(r, c))

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
                    if is_valid_cell(nr, nc) and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return region

    def determine_color_layers(region: List[Tuple[int, int]]) -> List[int]:
        layers = []
        remaining = set(region)
        while remaining:
            layer_color = input_grid.values[region[0][0]][region[0][1]]
            layer = [cell for cell in remaining if is_edge_cell(*cell) and input_grid.values[cell[0]][cell[1]] == layer_color]
            layers.append(layer_color)
            remaining -= set(layer)
            region = list(remaining)
        return layers

    def shift_colors(layers: List[int]) -> List[int]:
        return [layers[-1]] + layers[:-1] if layers else []

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    visited: Set[Tuple[int, int]] = set()

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and (r, c) not in visited:
                region = find_connected_region(r, c)
                color_layers = determine_color_layers(region)
                shifted_colors = shift_colors(color_layers)
                for (cell_r, cell_c), color in zip(region, shifted_colors):
                    new_grid[cell_r][cell_c] = color

    return ColoredGrid(values=new_grid)
