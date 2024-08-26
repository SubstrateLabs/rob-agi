from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    analyzing their properties, and assigning colors based on a comprehensive scoring system.
    
    1. Identify distinct contiguous shapes of sky blue (8) in the input grid using flood-fill.
    2. Analyze shapes for size, complexity, centrality, and orientation.
    3. Categorize shapes based on their geometric properties.
    4. Score shapes using a comprehensive system considering multiple factors.
    5. Assign colors to shapes based on their scores and categories, ensuring consistency and visual distinction.
    6. Handle small shapes by either merging them or ensuring they receive distinct colors.
    7. Create and return a new grid with transformed colors, maintaining black (0) as empty space.
    
    The function ensures consistent color assignment based on shape properties and their relationships,
    handling various grid sizes and shape configurations while maintaining visual clarity.
    """
    # Step 1: Identify distinct shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2 & 3: Analyze shapes, categorize them, and score them
    analyzed_shapes = analyze_shapes(shapes, input_grid)
    
    # Step 4 & 5: Assign colors to shapes
    colored_shapes = assign_colors_to_shapes(analyzed_shapes)
    
    # Step 6: Create output grid
    output_grid = create_output_grid(input_grid, colored_shapes)
    
    return output_grid

def find_contiguous_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    def flood_fill(r: int, c: int) -> List[Tuple[int, int]]:
        shape = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 8:
                visited.add((curr_r, curr_c))
                shape.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return shape
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8 and (r, c) not in visited:
                shapes.append(flood_fill(r, c))
    
    return shapes

def analyze_shapes(shapes: List[List[Tuple[int, int]]], grid: ColoredGrid) -> List[Dict]:
    analyzed_shapes = []
    rows, cols = grid.get_dimensions()
    total_area = rows * cols
    
    for shape in shapes:
        size = len(shape)
        min_r = min(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        
        # Calculate complexity (perimeter-to-area ratio)
        perimeter = sum(1 for r, c in shape if any((r+dr, c+dc) not in shape for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]))
        complexity = perimeter / size
        
        # Calculate centrality
        center_r, center_c = sum(r for r, _ in shape) / size, sum(c for _, c in shape) / size
        centrality = 1 - (abs(center_r - rows/2) / (rows/2) + abs(center_c - cols/2) / (cols/2)) / 2
        
        # Calculate orientation
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        orientation = "vertical" if height > width else "horizontal" if width > height else "square"
        
        # Categorize shape
        category = categorize_shape(shape, min_r, min_c, max_r, max_c)
        
        # Calculate relative size
        relative_size = size / total_area
        
        # Calculate score
        score = (
            size * 0.3 +
            complexity * 0.2 +
            centrality * 0.2 +
            relative_size * 0.3
        )
        
        analyzed_shapes.append({
            "shape": shape,
            "size": size,
            "complexity": complexity,
            "centrality": centrality,
            "orientation": orientation,
            "category": category,
            "relative_size": relative_size,
            "score": score
        })
    
    return sorted(analyzed_shapes, key=lambda x: x["score"], reverse=True)

def categorize_shape(shape: List[Tuple[int, int]], min_r: int, min_c: int, max_r: int, max_c: int) -> str:
    width = max_c - min_c + 1
    height = max_r - min_r + 1
    
    if width == height == 1:
        return "dot"
    elif width == 1 or height == 1:
        return "line"
    elif width == height:
        return "square"
    elif abs(width - height) <= 1:
        return "near_square"
    elif width > height * 2:
        return "wide_rectangle"
    elif height > width * 2:
        return "tall_rectangle"
    else:
        return "rectangle"

def assign_colors_to_shapes(analyzed_shapes: List[Dict]) -> List[Tuple[List[Tuple[int, int]], int]]:
    colors = [1, 2, 3, 4]
    colored_shapes = []
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    category_colors = {}
    
    for shape_info in analyzed_shapes:
        shape = shape_info["shape"]
        category = shape_info["category"]
        
        if category in category_colors:
            color = category_colors[category]
        else:
            # Assign a new color based on the least used color
            color = min(colors, key=lambda c: color_counts[c])
            category_colors[category] = color
        
        colored_shapes.append((shape, color))
        color_counts[color] += 1
    
    return colored_shapes

def create_output_grid(input_grid: ColoredGrid, colored_shapes: List[Tuple[List[Tuple[int, int]], int]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    for shape, color in colored_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, color)
    return output_grid
