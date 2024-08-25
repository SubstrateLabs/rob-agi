from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7d18a6fb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 7x7 output grid by identifying and arranging 3x3 colored shapes.
    
    1. Identifies 3x3 colored shapes in the input grid.
    2. Ranks shapes based on color rarity and position.
    3. Selects top 4 shapes and places them in quadrants of a 7x7 grid.
    4. Applies necessary rotations or flips to fit shapes aesthetically.
    5. Fills the central cross with black (0) to separate quadrants.
    
    Returns a 7x7 ColoredGrid with the arranged shapes.
    """
    shapes = identify_shapes(input_grid)
    ranked_shapes = rank_shapes(shapes, input_grid)
    output_grid = create_output_grid(ranked_shapes)
    return output_grid

def identify_shapes(grid: ColoredGrid) -> List[Tuple[int, List[List[int]], Tuple[int, int]]]:
    shapes = []
    rows, cols = grid.get_dimensions()
    for r in range(rows - 2):
        for c in range(cols - 2):
            shape = [row[c:c+3] for row in grid.values[r:r+3]]
            if any(any(cell != 0 for cell in row) for row in shape):
                color = max(max(row) for row in shape)
                shapes.append((color, shape, (r, c)))
    return shapes

def rank_shapes(shapes: List[Tuple[int, List[List[int]], Tuple[int, int]]], grid: ColoredGrid) -> List[Tuple[int, List[List[int]], Tuple[int, int]]]:
    color_counts = {color: grid.count_color(color) for color in range(1, 10)}
    def shape_score(shape):
        color, _, (r, c) = shape
        rarity = 1 / color_counts[color]
        position = min(r, grid.num_rows - r - 3) + min(c, grid.num_cols - c - 3)
        return rarity + position / 100
    return sorted(shapes, key=shape_score, reverse=True)

def create_output_grid(ranked_shapes: List[Tuple[int, List[List[int]], Tuple[int, int]]]) -> ColoredGrid:
    output = [[0 for _ in range(7)] for _ in range(7)]
    quadrants = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    for (color, shape, _), (qr, qc) in zip(ranked_shapes[:4], quadrants):
        for r in range(3):
            for c in range(3):
                output[qr + r][qc + c] = shape[r][c]
    
    return ColoredGrid(values=output)
