from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_8dae5dfc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying distinct shapes and inverting their color layers.
    
    The algorithm works as follows:
    1. Identify distinct shapes in the grid using flood fill, excluding black (0) pixels.
    2. For each shape, determine the color layers from outermost to innermost.
    3. Create a new color mapping for each shape by reversing the order of colors.
    4. Apply the new color mapping to each shape, starting from the outermost layer and moving inward.
    5. Preserve any black (0) pixels from the original grid.
    
    This process effectively "inverts" each shape's color layers while maintaining its overall structure and position.
    The outermost color of each shape becomes the innermost color in the transformed grid.
    """
    def find_shapes(grid: List[List[int]]) -> List[Set[Tuple[int, int]]]:
        shapes = []
        visited = set()
        rows, cols = len(grid), len(grid[0])
        
        def flood_fill(r: int, c: int, color: int) -> Set[Tuple[int, int]]:
            shape = set()
            stack = [(r, c)]
            while stack:
                r, c = stack.pop()
                if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid[r][c] == color:
                    visited.add((r, c))
                    shape.add((r, c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return shape
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r][c] != 0:
                    shapes.append(flood_fill(r, c, grid[r][c]))
        
        return shapes

    def get_color_layers(grid: List[List[int]], shape: Set[Tuple[int, int]]) -> List[int]:
        color_layers = []
        remaining = shape.copy()
        while remaining:
            current_layer = set()
            for r, c in remaining:
                if any((r+dr, c+dc) not in shape for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    current_layer.add((r, c))
            if current_layer:
                color = grid[next(iter(current_layer))[0]][next(iter(current_layer))[1]]
                color_layers.append(color)
                remaining -= current_layer
            else:
                # If no boundary found, add the remaining color
                color = grid[next(iter(remaining))[0]][next(iter(remaining))[1]]
                color_layers.append(color)
                break
        return color_layers

    def create_color_mapping(color_layers: List[int]) -> Dict[int, int]:
        return {old: new for old, new in zip(color_layers, reversed(color_layers))}

    def apply_inverted_colors(new_grid: List[List[int]], shape: Set[Tuple[int, int]], color_layers: List[int], color_mapping: Dict[int, int]) -> None:
        remaining = shape.copy()
        for color in color_layers:
            current_layer = set()
            for r, c in remaining:
                if not remaining or any((r+dr, c+dc) not in remaining for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    current_layer.add((r, c))
            for r, c in current_layer:
                new_grid[r][c] = color_mapping[color]
            remaining -= current_layer

    # Main algorithm
    shapes = find_shapes(input_grid.values)
    new_grid = [[0 for _ in range(len(input_grid.values[0]))] for _ in range(len(input_grid.values))]
    
    for shape in shapes:
        color_layers = get_color_layers(input_grid.values, shape)
        color_mapping = create_color_mapping(color_layers)
        apply_inverted_colors(new_grid, shape, color_layers, color_mapping)
    
    return ColoredGrid(values=new_grid)
