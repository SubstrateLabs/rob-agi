from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    analyzing their properties, and assigning colors based on shape characteristics and relationships.
    
    1. Identify distinct contiguous shapes of sky blue (8) in the input grid using flood-fill.
    2. Analyze shapes for size, form, complexity, and relative position.
    3. Create a hierarchy of shapes, identifying nested relationships.
    4. Rank shapes based on their characteristics and global patterns.
    5. Assign colors to shapes based on their rank, ensuring consistency and visual distinction.
    6. Handle nested shapes by assigning contrasting colors.
    7. Balance color distribution across the grid.
    8. Create and return a new grid with transformed colors, maintaining black (0) as empty space.
    
    The function ensures consistent color assignment based on shape properties, their relationships,
    and global patterns, prioritizing larger and more complex shapes while maintaining visual clarity
    and pattern consistency across different grid layouts.
    """
    # Step 1: Identify distinct shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2 & 3: Analyze shapes and create hierarchy
    analyzed_shapes = analyze_shapes(shapes, input_grid)
    
    # Step 4: Rank shapes
    ranked_shapes = rank_shapes(analyzed_shapes)
    
    # Step 5, 6 & 7: Assign colors to shapes
    colored_shapes = assign_colors_to_shapes(ranked_shapes)
    
    # Step 8: Create output grid
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
    
    for shape in shapes:
        size = len(shape)
        min_r = min(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        form = categorize_shape(shape, min_r, min_c, max_r, max_c)
        
        has_hole = any(grid.get_cell(r, c) == 0 for r in range(min_r, max_r+1) for c in range(min_c, max_c+1) if (r, c) not in shape)
        
        complexity = calculate_complexity(shape)
        
        position = calculate_position(min_r, max_r, min_c, max_c, rows, cols)
        
        analyzed_shapes.append({
            "shape": shape,
            "size": size,
            "form": form,
            "has_hole": has_hole,
            "complexity": complexity,
            "position": position,
            "bounding_box": (min_r, min_c, max_r, max_c)
        })
    
    return analyzed_shapes

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

def calculate_complexity(shape: List[Tuple[int, int]]) -> int:
    # Count the number of corners as a measure of complexity
    corners = 0
    for r, c in shape:
        neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if (r+dr, c+dc) in shape)
        if neighbors <= 2:
            corners += 1
    return corners

def calculate_position(min_r: int, max_r: int, min_c: int, max_c: int, rows: int, cols: int) -> str:
    center_r = (min_r + max_r) / 2
    center_c = (min_c + max_c) / 2
    
    vertical = "top" if center_r < rows / 3 else "bottom" if center_r > 2 * rows / 3 else "middle"
    horizontal = "left" if center_c < cols / 3 else "right" if center_c > 2 * cols / 3 else "center"
    
    return f"{vertical}-{horizontal}"

def rank_shapes(analyzed_shapes: List[Dict]) -> List[Dict]:
    def rank_key(shape):
        return (
            shape["size"],
            shape["complexity"],
            -abs(int(shape["position"].split("-")[0] == "middle")),  # Prefer shapes in the middle
            -abs(int(shape["position"].split("-")[1] == "center")),  # Prefer shapes in the center
            shape["has_hole"],
        )
    
    return sorted(analyzed_shapes, key=rank_key, reverse=True)

def assign_colors_to_shapes(ranked_shapes: List[Dict]) -> List[Tuple[List[Tuple[int, int]], int]]:
    colors = [1, 2, 3]
    colored_shapes = []
    used_colors = set()
    
    for i, shape_info in enumerate(ranked_shapes):
        shape = shape_info["shape"]
        
        if i < 3:
            color = colors[i]
        else:
            # For shapes beyond the top 3, assign colors based on similarity to top shapes
            similarities = [shape_similarity(shape_info, ranked_shapes[j]) for j in range(3)]
            color = colors[similarities.index(max(similarities))]
        
        colored_shapes.append((shape, color))
        used_colors.add(color)
    
    # Ensure all three colors are used if there are enough shapes
    if len(colored_shapes) >= 3 and len(used_colors) < 3:
        for unused_color in set(colors) - used_colors:
            for i, (shape, color) in enumerate(colored_shapes):
                if color != unused_color:
                    colored_shapes[i] = (shape, unused_color)
                    break
    
    return colored_shapes

def shape_similarity(shape1: Dict, shape2: Dict) -> float:
    # Calculate a similarity score between two shapes based on their properties
    form_similarity = int(shape1["form"] == shape2["form"])
    size_similarity = 1 - abs(shape1["size"] - shape2["size"]) / max(shape1["size"], shape2["size"])
    complexity_similarity = 1 - abs(shape1["complexity"] - shape2["complexity"]) / max(shape1["complexity"], shape2["complexity"])
    position_similarity = int(shape1["position"] == shape2["position"])
    hole_similarity = int(shape1["has_hole"] == shape2["has_hole"])
    
    return (form_similarity + size_similarity + complexity_similarity + position_similarity + hole_similarity) / 5

def create_output_grid(input_grid: ColoredGrid, colored_shapes: List[Tuple[List[Tuple[int, int]], int]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    for shape, color in colored_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, color)
    return output_grid
