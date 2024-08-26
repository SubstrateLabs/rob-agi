from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_358ba94e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 358ba94e challenge by analyzing 'S' shapes in the input grid and creating a standardized output.
    
    The function identifies all 5x5 'S' shapes of the same color, analyzes their hole configurations,
    and creates a new 5x5 grid with the most common 'S' shape configuration.
    
    Steps:
    1. Identify the target color and locate all 'S' shapes.
    2. Analyze hole configurations in valid 'S' shapes.
    3. Create a new 5x5 grid with the most common 'S' shape configuration.
    4. Ensure the output is a valid 'S' shape with appropriate holes.
    
    Returns a 5x5 ColoredGrid representing the standardized 'S' shape.
    """
    
    def find_target_color(grid: ColoredGrid) -> int:
        for row in grid.values:
            for cell in row:
                if cell != 0:
                    return cell
        return 0  # Default to black if no non-zero color found

    def is_valid_s_shape(subgrid: ColoredGrid) -> bool:
        if subgrid.get_dimensions() != (5, 5):
            return False
        color = find_target_color(subgrid)
        border = [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (2,4), (3,4), (4,4),
                  (4,3), (4,2), (4,1), (4,0), (3,0), (2,0), (1,0)]
        for r, c in border:
            if subgrid.values[r][c] != color:
                return False
        return True

    def find_holes(subgrid: ColoredGrid) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(5) for c in range(5) if subgrid.values[r][c] == 0]

    def count_hole_frequencies(s_shapes: List[ColoredGrid]) -> dict:
        frequencies = {}
        for shape in s_shapes:
            for hole in find_holes(shape):
                frequencies[hole] = frequencies.get(hole, 0) + 1
        return frequencies

    target_color = find_target_color(input_grid)
    regions = input_grid.find_connected_regions(target_color)
    s_shapes = [input_grid.extract_subgrid(r, c, 5, 5) for region in regions for r, c in region if r+4 < input_grid.num_rows and c+4 < input_grid.num_cols]
    s_shapes = [shape for shape in s_shapes if is_valid_s_shape(shape)]

    if not s_shapes:
        return ColoredGrid(values=[[target_color]*5 for _ in range(5)])

    hole_frequencies = count_hole_frequencies(s_shapes)
    sorted_holes = sorted(hole_frequencies.items(), key=lambda x: x[1], reverse=True)

    output = ColoredGrid(values=[[target_color]*5 for _ in range(5)])
    
    # Add first hole
    first_hole = (2, 1) if (2, 1) in hole_frequencies else (2, 3)
    output.values[first_hole[0]][first_hole[1]] = 0

    # Add second hole if majority of shapes have two holes
    if len(sorted_holes) > 1 and sum(freq for _, freq in sorted_holes[:2]) > len(s_shapes) // 2:
        second_hole = next((hole for hole, _ in sorted_holes if hole != first_hole), None)
        if second_hole:
            output.values[second_hole[0]][second_hole[1]] = 0

    return output
