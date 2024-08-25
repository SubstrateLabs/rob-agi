from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    analyzing their properties, and assigning colors based on a scoring system.
    
    1. Identify distinct contiguous shapes of sky blue (8) in the input grid using flood-fill.
    2. Analyze shapes for size, complexity, and position.
    3. Score shapes based on their properties and sort them.
    4. Assign colors to shapes using a dynamic scoring system that considers color balance and shape distinctions.
    5. Optimize color assignment to minimize the number of colors used while maintaining visual distinction.
    6. Create and return a new grid with transformed colors, maintaining black (0) as empty space.
    
    The function ensures consistent color assignment based on shape properties and their relationships,
    handling various grid sizes and shape configurations.
    """
    # Step 1: Identify distinct shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2 & 3: Analyze shapes and score them
    scored_shapes = analyze_and_score_shapes(shapes, input_grid)
    
    # Step 4 & 5: Assign colors to shapes
    colored_shapes = assign_colors_to_shapes(scored_shapes)
    
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

def analyze_and_score_shapes(shapes: List[List[Tuple[int, int]]], grid: ColoredGrid) -> List[Tuple[List[Tuple[int, int]], float]]:
    scored_shapes = []
    rows, cols = grid.get_dimensions()
    
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
        
        # Calculate score
        score = size * 0.5 + complexity * 0.3 + centrality * 0.2
        
        scored_shapes.append((shape, score))
    
    return sorted(scored_shapes, key=lambda x: x[1], reverse=True)

def assign_colors_to_shapes(scored_shapes: List[Tuple[List[Tuple[int, int]], float]]) -> List[Tuple[List[Tuple[int, int]], int]]:
    colors = [1, 2, 3, 4]
    colored_shapes = []
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for shape, _ in scored_shapes:
        best_color = min(colors, key=lambda c: color_counts[c])
        colored_shapes.append((shape, best_color))
        color_counts[best_color] += 1
    
    return colored_shapes

def create_output_grid(input_grid: ColoredGrid, colored_shapes: List[Tuple[List[Tuple[int, int]], int]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    for shape, color in colored_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, color)
    return output_grid
