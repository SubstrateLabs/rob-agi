from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    analyzing their properties, and assigning colors based on shape characteristics and relationships.
    
    1. Identify distinct contiguous shapes of sky blue (8) in the input grid using flood-fill.
    2. Analyze shapes for size, form, and relative position.
    3. Categorize shapes based on their geometric properties and relationships.
    4. Assign colors to shapes based on their categories and relative importance, ensuring consistency across the grid.
    5. Handle nested shapes and maintain proper color relationships.
    6. Create and return a new grid with transformed colors, maintaining black (0) as empty space.
    
    The function ensures consistent color assignment based on shape properties and their relationships,
    prioritizing larger and more complex shapes while maintaining visual clarity and pattern consistency.
    """
    # Step 1: Identify distinct shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2 & 3: Analyze and categorize shapes
    analyzed_shapes = analyze_and_categorize_shapes(shapes, input_grid)
    
    # Step 4: Assign colors to shapes
    colored_shapes = assign_colors_to_shapes(analyzed_shapes)
    
    # Step 5 & 6: Create output grid
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

def analyze_and_categorize_shapes(shapes: List[List[Tuple[int, int]]], grid: ColoredGrid) -> List[Dict]:
    analyzed_shapes = []
    rows, cols = grid.get_dimensions()
    
    for shape in shapes:
        size = len(shape)
        min_r = min(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        
        # Calculate form
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        form = categorize_shape(shape, min_r, min_c, max_r, max_c)
        
        # Check if the shape contains a hole
        has_hole = any(grid.get_cell(r, c) == 0 for r in range(min_r, max_r+1) for c in range(min_c, max_c+1) if (r, c) not in shape)
        
        # Calculate position (top, middle, bottom)
        position = "top" if max_r < rows / 3 else "bottom" if min_r > 2 * rows / 3 else "middle"
        
        analyzed_shapes.append({
            "shape": shape,
            "size": size,
            "form": form,
            "has_hole": has_hole,
            "position": position,
            "bounding_box": (min_r, min_c, max_r, max_c)
        })
    
    return sorted(analyzed_shapes, key=lambda x: (-x["size"], x["form"], x["position"]))

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
    colors = [1, 2, 3]
    colored_shapes = []
    used_colors = set()
    form_colors = {}
    
    for shape_info in analyzed_shapes:
        shape = shape_info["shape"]
        form = shape_info["form"]
        
        if form in form_colors:
            color = form_colors[form]
        else:
            # Assign a new color based on the shape's characteristics
            if shape_info["has_hole"] or shape_info["form"] in ["square", "near_square"]:
                color = 1  # Blue for shapes with holes or square-like shapes
            elif shape_info["position"] == "bottom" or shape_info["form"] in ["wide_rectangle", "tall_rectangle"]:
                color = 3  # Green for shapes at the bottom or elongated rectangles
            else:
                color = 2  # Red for other shapes
            
            form_colors[form] = color
        
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

def create_output_grid(input_grid: ColoredGrid, colored_shapes: List[Tuple[List[Tuple[int, int]], int]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    for shape, color in colored_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, color)
    return output_grid
