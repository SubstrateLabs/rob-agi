from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_8dae5dfc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by shifting colors within connected regions.
    
    The transformation process:
    1. Identifies distinct structures (connected regions of non-black cells).
    2. For each structure, determines the color layers from outer to inner.
    3. Creates a color mapping that shifts colors:
       - The innermost color becomes the new outermost color.
       - All other colors shift inward by one position.
    4. Applies the color mapping to each structure in the new grid.
    
    Black (0) cells, representing empty space, remain unchanged.
    The overall structure and position of shapes are maintained.
    """
    def get_adjacent_cells(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def find_structure(start_r: int, start_c: int) -> Set[Tuple[int, int]]:
        structure = set()
        stack = [(start_r, start_c)]
        start_color = input_grid.values[start_r][start_c]
        while stack:
            r, c = stack.pop()
            if (r, c) not in structure and input_grid.values[r][c] != 0:
                structure.add((r, c))
                for nr, nc in get_adjacent_cells(r, c):
                    if is_valid_cell(nr, nc) and (nr, nc) not in structure and input_grid.values[nr][nc] != 0:
                        stack.append((nr, nc))
        return structure

    def determine_color_layers(structure: Set[Tuple[int, int]]) -> List[int]:
        color_layers = []
        seen_colors = set()
        queue = deque(structure)
        visited = set()
        while queue:
            layer_size = len(queue)
            layer_colors = set()
            for _ in range(layer_size):
                r, c = queue.popleft()
                if (r, c) not in visited:
                    visited.add((r, c))
                    color = input_grid.values[r][c]
                    if color not in seen_colors:
                        layer_colors.add(color)
                        seen_colors.add(color)
                    for nr, nc in get_adjacent_cells(r, c):
                        if (nr, nc) in structure and (nr, nc) not in visited:
                            queue.append((nr, nc))
            if layer_colors:
                color_layers.extend(sorted(layer_colors))
        return color_layers

    def create_color_mapping(color_layers: List[int]) -> Dict[int, int]:
        if len(color_layers) <= 1:
            return {color_layers[0]: color_layers[0]} if color_layers else {}
        return {old: new for old, new in zip(color_layers, [color_layers[-1]] + color_layers[:-1])}

    def apply_color_transformation(structure: Set[Tuple[int, int]], color_mapping: Dict[int, int], new_grid: List[List[int]]):
        for r, c in structure:
            new_grid[r][c] = color_mapping[input_grid.values[r][c]]

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    processed = set()

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and (r, c) not in processed:
                structure = find_structure(r, c)
                processed.update(structure)
                color_layers = determine_color_layers(structure)
                color_mapping = create_color_mapping(color_layers)
                apply_color_transformation(structure, color_mapping, new_grid)

    return ColoredGrid(values=new_grid)
