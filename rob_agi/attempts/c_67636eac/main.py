from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_67636eac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts shapes from the input grid and arranges them in a new grid.
    
    1. Scans the input grid to identify non-black shapes.
    2. Extracts each shape in its minimal bounding box, preserving its structure including internal black cells.
    3. Arranges shapes vertically in the order they appear in the input grid (top-to-bottom, left-to-right).
    4. Creates a new grid with the extracted shapes stacked vertically, centered horizontally.
    5. Preserves the exact internal structure of each shape, including all black cells within the shape's bounding box.
    
    Args:
    input_grid (ColoredGrid): The input grid containing shapes.
    
    Returns:
    ColoredGrid: A new grid with extracted shapes arranged vertically.
    """
    shapes = []
    rows, cols = input_grid.get_dimensions()
    visited = set()

    # Identify and extract shapes
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and input_grid.get_cell(r, c) != 0:
                # Find the bounding box of the shape
                min_r, max_r, min_c, max_c = r, r, c, c
                stack = [(r, c)]
                shape_cells = set()
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited:
                        visited.add((curr_r, curr_c))
                        shape_cells.add((curr_r, curr_c))
                        min_r, max_r = min(min_r, curr_r), max(max_r, curr_r)
                        min_c, max_c = min(min_c, curr_c), max(max_c, curr_c)
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and input_grid.get_cell(nr, nc) == input_grid.get_cell(r, c):
                                stack.append((nr, nc))
                
                # Extract shape (including all cells within the bounding box)
                shape = input_grid.extract_subgrid(min_r, min_c, max_r - min_r + 1, max_c - min_c + 1)
                shapes.append((min_r, min_c, shape))

    # Sort shapes based on their original position (top-to-bottom, left-to-right)
    shapes.sort(key=lambda x: (x[0], x[1]))

    # Calculate output grid dimensions
    output_width = max(shape.get_dimensions()[1] for _, _, shape in shapes)
    output_height = sum(shape.get_dimensions()[0] for _, _, shape in shapes)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(output_width)] for _ in range(output_height)])

    # Populate output grid
    current_pos = 0
    for _, _, shape in shapes:
        shape_height, shape_width = shape.get_dimensions()
        start_col = (output_width - shape_width) // 2
        for r in range(shape_height):
            for c in range(shape_width):
                output_grid.set_cell(current_pos + r, start_col + c, shape.get_cell(r, c))
        current_pos += shape_height

    return output_grid
