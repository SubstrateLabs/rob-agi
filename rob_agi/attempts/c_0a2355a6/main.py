from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    assigning unique numbers based on their position, and coloring them accordingly.
    
    1. Identify distinct contiguous shapes of sky blue (8) in the input grid using flood-fill.
    2. Assign unique numbers to shapes based on their top-left position (highest number to top-left-most shape).
    3. Determine the number of colors to use based on the number of distinct shapes.
    4. Create a color mapping based on the number of colors needed.
    5. Create and return a new grid with transformed colors, maintaining black (0) as empty space.
    
    The function ensures consistent color assignment based on the relative positions of shapes
    and handles various grid sizes and shape configurations.
    """
    # Step 1: Identify distinct shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2: Assign unique numbers to shapes
    numbered_shapes = assign_shape_numbers(shapes)
    
    # Step 3 & 4: Determine color palette and create color mapping
    num_colors = min(len(shapes), 4)
    color_mapping = create_color_mapping(num_colors)
    
    # Step 5: Create output grid
    output_grid = create_output_grid(input_grid, numbered_shapes, color_mapping)
    
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

def assign_shape_numbers(shapes: List[List[Tuple[int, int]]]) -> List[Tuple[List[Tuple[int, int]], int]]:
    sorted_shapes = sorted(shapes, key=lambda shape: (min(r for r, _ in shape), min(c for _, c in shape)))
    return [(shape, len(sorted_shapes) - i) for i, shape in enumerate(sorted_shapes)]

def create_color_mapping(num_colors: int) -> Dict[int, int]:
    colors = [1, 2, 3, 4][:num_colors]
    return {i + 1: color for i, color in enumerate(reversed(colors))}

def create_output_grid(input_grid: ColoredGrid, numbered_shapes: List[Tuple[List[Tuple[int, int]], int]], color_mapping: Dict[int, int]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    for shape, number in numbered_shapes:
        color = color_mapping[number]
        for r, c in shape:
            output_grid.set_cell(r, c, color)
    return output_grid
