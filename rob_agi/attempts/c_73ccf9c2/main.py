from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_73ccf9c2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Identify the largest shape in the input grid, scale it, and center it in a 7x5 output grid.
    
    The function performs the following steps:
    1. Find all non-black shapes in the input grid
    2. Select the largest shape based on pixel count
    3. Determine the bounding box of the selected shape
    4. Scale the shape to fit within a 7x5 grid while maintaining aspect ratio
    5. Center the scaled shape in the 7x5 output grid
    6. Apply the original color to the output shape
    """
    shapes = find_shapes(input_grid)
    if not shapes:
        return ColoredGrid(values=[[0] * 5 for _ in range(7)])  # Return a 7x5 black grid if no shapes found
    
    largest_shape = max(shapes, key=len)
    color = input_grid.get_cell(largest_shape[0][0], largest_shape[0][1])
    return scale_and_center(largest_shape, (7, 5), color)

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
        
        # Calculate symmetry score
        center_r = (min_r + max_r) / 2
        center_c = (min_c + max_c) / 2
        symmetry_score = sum(1 for r, c in shape if (2*center_r-r, 2*center_c-c) in shape)
        symmetry_score /= len(shape)
        
        return complexity * distinctive_features * (1 + symmetry_score)
    
    return max(shapes, key=score_shape)

def extract_key_points(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    
    corners = [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
    center = ((min_r + max_r) // 2, (min_c + max_c) // 2)
    
    # Find points furthest from the center in each quadrant
    quadrants = [[], [], [], []]
    for r, c in shape:
        if r <= center[0] and c <= center[1]:
            quadrants[0].append((r, c))
        elif r <= center[0] and c > center[1]:
            quadrants[1].append((r, c))
        elif r > center[0] and c <= center[1]:
            quadrants[2].append((r, c))
        else:
            quadrants[3].append((r, c))
    
    extremities = []
    for quadrant in quadrants:
        if quadrant:
            extremities.append(max(quadrant, key=lambda p: (p[0]-center[0])**2 + (p[1]-center[1])**2))
    
    return list(set(corners + [center] + extremities))

def simplify_shape(key_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Sort points by their angle from the center
    center = key_points[key_points.index(min(key_points, key=lambda p: p[0]**2 + p[1]**2))]
    sorted_points = sorted(key_points, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0]))
    
    # Remove points that are too close to each other
    simplified = [sorted_points[0]]
    for point in sorted_points[1:]:
        if math.hypot(point[0]-simplified[-1][0], point[1]-simplified[-1][1]) > 1:
            simplified.append(point)
    
    return simplified

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
    
    # Scale the shape
    scaled_shape = []
    for r, c in shape:
        new_r = (r - min_r) * scale
        new_c = (c - min_c) * scale
        scaled_shape.append((new_r, new_c))
    
    # Find the bounding box of the scaled shape
    min_scaled_r = min(r for r, _ in scaled_shape)
    max_scaled_r = max(r for r, _ in scaled_shape)
    min_scaled_c = min(c for _, c in scaled_shape)
    max_scaled_c = max(c for _, c in scaled_shape)
    
    # Calculate centering offsets
    offset_r = (rows - (max_scaled_r - min_scaled_r)) / 2 - min_scaled_r
    offset_c = (cols - (max_scaled_c - min_scaled_c)) / 2 - min_scaled_c
    
    # Center and draw the scaled shape
    for r, c in scaled_shape:
        new_r = int(r + offset_r)
        new_c = int(c + offset_c)
        if 0 <= new_r < rows and 0 <= new_c < cols:
            output[new_r][new_c] = color
    
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
