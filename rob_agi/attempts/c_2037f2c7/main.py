from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified, abstract representation.
    
    1. Detects and analyzes shapes in the input grid.
    2. Determines significant shapes based on size and position.
    3. Creates a compact output grid representing key features of significant shapes.
    4. Balances the composition and ensures proper representation of shape characteristics.
    5. Returns a ColoredGrid object with the simplified, abstract representation.
    """
    shapes = detect_shapes(input_grid)
    significant_shapes = determine_significant_shapes(shapes)
    output_size = calculate_output_size(significant_shapes)
    output_grid = create_abstract_representation(significant_shapes, output_size)
    output_grid = balance_composition(output_grid)
    
    return ColoredGrid(values=output_grid)

def detect_shapes(grid: ColoredGrid) -> List[Dict]:
    rows, cols = grid.get_dimensions()
    visited = set()
    shapes = []

    def flood_fill(r: int, c: int) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        queue = [(r, c)]
        shape = []
        color_count = Counter()
        min_r, min_c, max_r, max_c = r, c, r, c

        while queue:
            curr_r, curr_c = queue.pop(0)
            if (curr_r, curr_c) in visited or grid.get_cell(curr_r, curr_c) == 0:
                continue

            visited.add((curr_r, curr_c))
            shape.append((curr_r, curr_c))
            color_count[grid.get_cell(curr_r, curr_c)] += 1
            min_r, min_c = min(min_r, curr_r), min(min_c, curr_c)
            max_r, max_c = max(max_r, curr_r), max(max_c, curr_c)

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

        return shape, color_count, (min_r, min_c, max_r, max_c)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                shape, color_count, bbox = flood_fill(r, c)
                if shape:
                    height = bbox[2] - bbox[0] + 1
                    width = bbox[3] - bbox[1] + 1
                    shapes.append({
                        'cells': shape,
                        'bbox': bbox,
                        'aspect_ratio': height / width,
                        'size': len(shape),
                        'color_count': color_count,
                        'centroid': (sum(x[0] for x in shape) / len(shape), sum(x[1] for x in shape) / len(shape))
                    })

    return shapes

def determine_significant_shapes(shapes: List[Dict]) -> List[Dict]:
    shapes.sort(key=lambda x: x['size'], reverse=True)
    avg_size = sum(shape['size'] for shape in shapes) / len(shapes)
    significant_shapes = [shape for shape in shapes if shape['size'] > 0.5 * avg_size]
    return significant_shapes[:min(len(significant_shapes), 6)]

def calculate_output_size(shapes: List[Dict]) -> Tuple[int, int]:
    width = max(2, min(len(shapes), 6))
    avg_aspect_ratio = sum(shape['aspect_ratio'] for shape in shapes) / len(shapes)
    height = 2 if avg_aspect_ratio < 1.2 else 3 if avg_aspect_ratio < 1.8 else 4
    return height, width

def create_abstract_representation(shapes: List[Dict], size: Tuple[int, int]) -> List[List[int]]:
    height, width = size
    grid = [[0 for _ in range(width)] for _ in range(height)]
    max_size = max(shape['size'] for shape in shapes)

    for i, shape in enumerate(shapes):
        relative_size = shape['size'] / max_size
        vertical_emphasis = shape['aspect_ratio']
        cells_to_fill = 1 if relative_size < 0.33 else 2 if relative_size < 0.67 else 3

        if vertical_emphasis < 1:
            start = height - cells_to_fill
        elif vertical_emphasis < 1.5:
            start = (height - cells_to_fill) // 2
        else:
            start = 0

        for j in range(start, start + cells_to_fill):
            if j < height:
                grid[j][i] = 8

    # Ensure bottom row has at least one filled cell
    if all(cell == 0 for cell in grid[-1]):
        grid[-1][-1] = 8

    # Fill rightmost column if empty
    if all(row[-1] == 0 for row in grid):
        grid[height // 2][-1] = 8

    return grid

def balance_composition(grid: List[List[int]]) -> List[List[int]]:
    height, width = len(grid), len(grid[0])
    filled_cells = sum(sum(1 for cell in row if cell == 8) for row in grid)
    total_cells = height * width

    # Adjust if too few or too many filled cells
    while filled_cells < total_cells * 0.25 or filled_cells > total_cells * 0.75:
        if filled_cells < total_cells * 0.25:
            # Add a cell
            for r in range(height):
                for c in range(width):
                    if grid[r][c] == 0:
                        grid[r][c] = 8
                        filled_cells += 1
                        break
                if filled_cells >= total_cells * 0.25:
                    break
        elif filled_cells > total_cells * 0.75:
            # Remove a cell
            for r in range(height):
                for c in range(width):
                    if grid[r][c] == 8:
                        grid[r][c] = 0
                        filled_cells -= 1
                        break
                if filled_cells <= total_cells * 0.75:
                    break

    return grid
