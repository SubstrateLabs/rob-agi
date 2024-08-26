from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_73ccf9c2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Identify the most figure-like shape in the input grid, simplify it, and return a scaled-down version.
    
    The function performs the following steps:
    1. Find all non-black shapes in the input grid
    2. Select the most figure-like shape based on complexity and distinctive features
    3. Extract key features of the selected shape
    4. Transform and simplify the shape
    5. Scale down and center the simplified shape in a smaller output grid
    6. Apply the original color to the output shape
    """
    shapes = find_shapes(input_grid)
    if not shapes:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if no shapes found
    
    most_figure_like = select_most_figure_like(shapes)
    key_points = extract_key_points(most_figure_like)
    simplified = simplify_shape(key_points)
    output_size = determine_output_size(input_grid.get_dimensions())
    color = input_grid.get_cell(most_figure_like[0][0], most_figure_like[0][1])
    return scale_and_center(simplified, output_size, color)

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
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                stack.append((nr, nc))
                shapes.append(shape)
    return shapes

def select_most_figure_like(shapes: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    def score_shape(shape):
        # Calculate complexity score
        complexity = len(shape)
        
        # Calculate distinctive features score
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        bounding_box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        distinctive_features = complexity / bounding_box_area
        
        return complexity * distinctive_features
    
    return max(shapes, key=score_shape)

def extract_key_points(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # For simplicity, we'll use corner points as key points
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    
    corners = [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
    center = ((min_r + max_r) // 2, (min_c + max_c) // 2)
    
    return corners + [center] + [p for p in shape if p not in corners and p != center]

def simplify_shape(key_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # For now, we'll just use the key points as the simplified shape
    return key_points

def determine_output_size(input_size: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = input_size
    if rows <= 10 and cols <= 10:
        return (4, 4)
    elif rows <= 20 and cols <= 20:
        return (5, 4)
    else:
        return (7, 5)

def scale_and_center(shape: List[Tuple[int, int]], output_size: Tuple[int, int], color: int) -> ColoredGrid:
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
    scaled_shape = []
    for r, c in shape:
        new_r = int((r - min_r) * scale + (rows - (max_r - min_r) * scale) / 2)
        new_c = int((c - min_c) * scale + (cols - (max_c - min_c) * scale) / 2)
        if 0 <= new_r < rows and 0 <= new_c < cols:
            scaled_shape.append((new_r, new_c))
    
    # Draw lines between adjacent points
    for i in range(len(scaled_shape)):
        r1, c1 = scaled_shape[i]
        r2, c2 = scaled_shape[(i + 1) % len(scaled_shape)]
        for r, c in bresenham_line(r1, c1, r2, c2):
            if 0 <= r < rows and 0 <= c < cols:
                output[r][c] = color
    
    return ColoredGrid(values=output)

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm for drawing lines"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
