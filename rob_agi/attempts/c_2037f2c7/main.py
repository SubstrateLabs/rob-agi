from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified, abstract representation.
    
    1. Detects and analyzes shapes in the input grid.
    2. Plans the output grid layout based on the number and characteristics of shapes.
    3. Generates an abstract representation of each shape.
    4. Balances the overall composition and ensures key features are represented.
    5. Returns a ColoredGrid object with the simplified representation.
    """
    shapes = detect_shapes(input_grid)
    output_size = plan_output_grid(shapes)
    output_grid = create_base_grid(output_size)
    output_grid = generate_abstract_representation(output_grid, shapes)
    output_grid = balance_composition(output_grid)
    
    return ColoredGrid(values=output_grid)

def detect_shapes(grid: ColoredGrid) -> List[dict]:
    rows, cols = grid.get_dimensions()
    visited = set()
    shapes = []

    def flood_fill(r: int, c: int) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        queue = [(r, c)]
        shape = []
        color = grid.get_cell(r, c)
        min_r, min_c, max_r, max_c = r, c, r, c

        while queue:
            curr_r, curr_c = queue.pop(0)
            if (curr_r, curr_c) in visited or grid.get_cell(curr_r, curr_c) != color:
                continue

            visited.add((curr_r, curr_c))
            shape.append((curr_r, curr_c))
            min_r, min_c = min(min_r, curr_r), min(min_c, curr_c)
            max_r, max_c = max(max_r, curr_r), max(max_c, curr_c)

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

        return shape, (min_r, min_c, max_r, max_c)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                shape, bbox = flood_fill(r, c)
                if shape:
                    height = bbox[2] - bbox[0] + 1
                    width = bbox[3] - bbox[1] + 1
                    shapes.append({
                        'cells': shape,
                        'bbox': bbox,
                        'aspect_ratio': height / width,
                        'density': len(shape) / (height * width),
                        'centroid': (sum(x[0] for x in shape) / len(shape), sum(x[1] for x in shape) / len(shape))
                    })

    return shapes

def plan_output_grid(shapes: List[dict]) -> Tuple[int, int]:
    num_shapes = len(shapes)
    width = num_shapes + 1
    height = 4 if any(shape['aspect_ratio'] > 1.5 for shape in shapes) else 3
    return height, width

def create_base_grid(size: Tuple[int, int]) -> List[List[int]]:
    return [[0 for _ in range(size[1])] for _ in range(size[0])]

def generate_abstract_representation(grid: List[List[int]], shapes: List[dict]) -> List[List[int]]:
    for i, shape in enumerate(shapes):
        col = i + 1
        if shape['aspect_ratio'] > 1.5:  # Vertical shape
            grid[0][col] = grid[1][col] = grid[2][col] = 8
            if len(grid) > 3:
                grid[3][col] = 8
        else:  # Horizontal or square shape
            grid[1][col] = grid[2][col] = 8
            if shape['density'] > 0.5:
                grid[0][col] = 8

    return grid

def balance_composition(grid: List[List[int]]) -> List[List[int]]:
    # Ensure rightmost column has at least one 8
    if all(row[-1] == 0 for row in grid):
        grid[len(grid) // 2][-1] = 8

    # Balance middle rows
    middle_rows = grid[1:-1] if len(grid) > 3 else [grid[1]]
    for row in middle_rows:
        if sum(row) < 2:
            for i in range(1, len(row) - 1):
                if row[i-1] == 0 and row[i+1] == 0:
                    row[i] = 8
                    break

    return grid
