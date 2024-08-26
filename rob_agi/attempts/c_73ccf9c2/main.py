from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_73ccf9c2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Identify the most figure-like shape in the input grid, simplify it, and return a scaled-down version.
    
    The function performs the following steps:
    1. Find all non-black shapes in the input grid
    2. Select the most figure-like shape based on symmetry and complexity
    3. Simplify the selected shape to its key points
    4. Scale down and center the simplified shape in a smaller output grid
    """
    shapes = find_shapes(input_grid)
    if not shapes:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if no shapes found
    
    most_figure_like = select_most_figure_like(shapes)
    simplified = simplify_shape(most_figure_like)
    output_size = determine_output_size(input_grid.get_dimensions())
    return scale_and_center(simplified, output_size)

def find_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                shape = []
                color = grid.get_cell(r, c)
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                        visited.add((curr_r, curr_c))
                        shape.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                stack.append((nr, nc))
                shapes.append(shape)
    return shapes

def select_most_figure_like(shapes: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    # For now, just select the largest shape
    return max(shapes, key=len)

def simplify_shape(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # For now, just return the original shape
    return shape

def determine_output_size(input_size: Tuple[int, int]) -> Tuple[int, int]:
    # For now, use a fixed size of 4x4
    return (4, 4)

def scale_and_center(shape: List[Tuple[int, int]], output_size: Tuple[int, int]) -> ColoredGrid:
    rows, cols = output_size
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Find the bounding box of the shape
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    
    # Calculate scaling factors
    scale_r = (rows - 1) / (max_r - min_r) if max_r > min_r else 1
    scale_c = (cols - 1) / (max_c - min_c) if max_c > min_c else 1
    scale = min(scale_r, scale_c)
    
    # Scale and center the shape
    for r, c in shape:
        new_r = int((r - min_r) * scale + (rows - (max_r - min_r) * scale) / 2)
        new_c = int((c - min_c) * scale + (cols - (max_c - min_c) * scale) / 2)
        if 0 <= new_r < rows and 0 <= new_c < cols:
            output[new_r][new_c] = 1  # Assuming all shapes are blue (1)
    
    return ColoredGrid(values=output)
