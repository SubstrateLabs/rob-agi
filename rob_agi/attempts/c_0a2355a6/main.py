from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    analyzing their properties, and assigning colors based on shape characteristics and size.
    
    1. Identify distinct contiguous shapes of sky blue (8) in the input grid using flood-fill.
    2. Analyze shapes for size, complexity, and position.
    3. Rank shapes based primarily on their size, with complexity as a secondary factor.
    4. Assign colors to shapes based on their rank:
       - The largest shape always gets the highest available color (4 if 4+ shapes, else 3).
       - Remaining shapes are assigned colors in descending order of size and color number.
    5. Create and return a new grid with transformed colors, maintaining black (0) as empty space.
    
    This approach ensures consistent color assignment across different grid layouts,
    prioritizing the largest shapes and maintaining visual distinction between shapes.
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
            shape["size"] * 1000 + shape["complexity"],  # Prioritize size heavily
            -abs(int(shape["position"].split("-")[0] == "top")),  # Slight preference for top shapes
            -abs(int(shape["position"].split("-")[1] == "left")),  # Slight preference for left shapes
        )
    
    return sorted(analyzed_shapes, key=rank_key, reverse=True)

def assign_colors_to_shapes(ranked_shapes: List[Dict]) -> List[Tuple[List[Tuple[int, int]], int]]:
    num_shapes = len(ranked_shapes)
    colors = [1, 2, 3, 4] if num_shapes >= 4 else list(range(1, num_shapes + 1))
    colored_shapes = []
    
    for i, shape_info in enumerate(ranked_shapes):
        shape = shape_info["shape"]
        if i == 0:  # Largest shape
            color = colors[-1]  # Assign the highest available color
        else:
            color = colors[i - 1] if i < len(colors) else colors[-1]
        
        colored_shapes.append((shape, color))
    
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
