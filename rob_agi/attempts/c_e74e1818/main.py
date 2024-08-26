from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selectively flipping shapes vertically within their bounds
    to achieve a more balanced and symmetrical composition.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Analyze each shape's characteristics (weight distribution, bounds, position)
    3. Determine if each shape should be flipped based on its position relative to the grid center
    4. Flip the shapes that improve the overall composition
    5. Reconstruct the grid with the flipped shapes while maintaining vertical order and horizontal positions
    
    This function improves vertical symmetry and balance of the image while preserving
    the vertical ordering and horizontal positions of shapes.
    """
    shapes = identify_shapes(input_grid)
    flipped_shapes = determine_flips(shapes, input_grid.num_rows)
    return reconstruct_grid(shapes, flipped_shapes, input_grid)

def identify_shapes(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    shapes = {}
    visited = set()
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                color = grid.values[r][c]
                shape = []
                stack = [(r, c)]
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == color:
                        visited.add((curr_r, curr_c))
                        shape.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                                stack.append((nr, nc))
                
                if color not in shapes:
                    shapes[color] = []
                shapes[color].append(shape)
    
    return shapes

def determine_flips(shapes: Dict[int, List[List[Tuple[int, int]]]], grid_height: int) -> Dict[int, List[bool]]:
    flipped_shapes = {}
    grid_center = grid_height / 2
    
    for color, color_shapes in shapes.items():
        flipped_shapes[color] = []
        for shape in color_shapes:
            min_r = min(r for r, _ in shape)
            max_r = max(r for r, _ in shape)
            shape_center = (min_r + max_r) / 2
            
            if shape_center < grid_center:
                flipped_shapes[color].append(True)
            else:
                flipped_shapes[color].append(False)
    
    return flipped_shapes

def flip_shape_vertically(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    return [(max_r - (r - min_r), c) for r, c in shape]

def reconstruct_grid(shapes: Dict[int, List[List[Tuple[int, int]]]], flipped_shapes: Dict[int, List[bool]], original_grid: ColoredGrid) -> ColoredGrid:
    new_grid = [[0 for _ in range(original_grid.num_cols)] for _ in range(original_grid.num_rows)]
    
    for color, color_shapes in shapes.items():
        for shape, should_flip in zip(color_shapes, flipped_shapes[color]):
            if should_flip:
                shape = flip_shape_vertically(shape)
            
            for r, c in shape:
                new_grid[r][c] = color
    
    return ColoredGrid(values=new_grid)
