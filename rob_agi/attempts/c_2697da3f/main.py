from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2697da3f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger, symmetrical pattern.
    
    The transformation involves:
    1. Analyzing the input grid for colored cells, edge touches, and corner touches.
    2. Creating a larger output grid (2n-1 where n is the max input dimension).
    3. Building the top-left quadrant of the output grid based on the input.
    4. Applying rotational symmetry to create the full output grid.
    5. Handling special cases for corners and edges.
    6. Cleaning up any isolated black cells for pattern consistency.
    """
    def analyze_input(grid: ColoredGrid) -> dict:
        rows, cols = grid.get_dimensions()
        colored_cells = []
        edge_touches = {'top': set(), 'left': set(), 'bottom': set(), 'right': set()}
        corner_touches = set()
        
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 0:
                    colored_cells.append((r, c, grid.values[r][c]))
                    if r == 0:
                        edge_touches['top'].add(c)
                    if r == rows - 1:
                        edge_touches['bottom'].add(c)
                    if c == 0:
                        edge_touches['left'].add(r)
                    if c == cols - 1:
                        edge_touches['right'].add(r)
                    if (r, c) in [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]:
                        corner_touches.add((r, c))
        
        return {
            'colored_cells': colored_cells,
            'edge_touches': edge_touches,
            'corner_touches': corner_touches
        }

    def create_output_grid(input_size: Tuple[int, int]) -> List[List[int]]:
        output_size = max(input_size) * 2 - 1
        return [[0 for _ in range(output_size)] for _ in range(output_size)]

    def create_top_left_quadrant(input_grid: ColoredGrid, quadrant_size: int, analysis: dict) -> List[List[int]]:
        quadrant = [[0 for _ in range(quadrant_size)] for _ in range(quadrant_size)]
        input_rows, input_cols = input_grid.get_dimensions()
        input_center = (input_rows // 2, input_cols // 2)
        quadrant_center = (quadrant_size // 2, quadrant_size // 2)
        
        for r, c, color in analysis['colored_cells']:
            rel_r, rel_c = r - input_center[0], c - input_center[1]
            quadrant[quadrant_center[0] + rel_r][quadrant_center[1] + rel_c] = color
            
            # Extend to edges if on the edge of input
            if r in analysis['edge_touches']['left'] or c in analysis['edge_touches']['top']:
                if rel_r == 0:
                    for i in range(quadrant_center[1] + rel_c, -1, -1):
                        quadrant[quadrant_center[0]][i] = color
                if rel_c == 0:
                    for i in range(quadrant_center[0] + rel_r, -1, -1):
                        quadrant[i][quadrant_center[1]] = color
        
        return quadrant

    def apply_rotational_symmetry(quadrant: List[List[int]]) -> List[List[int]]:
        size = len(quadrant) * 2 - 1
        output = [[0 for _ in range(size)] for _ in range(size)]
        
        for r in range(len(quadrant)):
            for c in range(len(quadrant)):
                if quadrant[r][c] != 0:
                    # Top-left quadrant
                    output[r][c] = quadrant[r][c]
                    # Top-right quadrant
                    output[r][size-1-c] = quadrant[r][c]
                    # Bottom-left quadrant
                    output[size-1-r][c] = quadrant[r][c]
                    # Bottom-right quadrant
                    output[size-1-r][size-1-c] = quadrant[r][c]
        
        return output

    def handle_corners_and_edges(output_grid: List[List[int]], analysis: dict) -> None:
        size = len(output_grid)
        
        # Handle corners
        if analysis['corner_touches']:
            corner_color = output_grid[0][0] or output_grid[0][size-1] or output_grid[size-1][0] or output_grid[size-1][size-1]
            if corner_color:
                output_grid[0][0] = corner_color
                output_grid[0][size-1] = corner_color
                output_grid[size-1][0] = corner_color
                output_grid[size-1][size-1] = corner_color
        
        # Handle edges
        for edge, touches in analysis['edge_touches'].items():
            if touches:
                if edge == 'top':
                    for c in range(size):
                        if output_grid[0][c] != 0:
                            output_grid[0] = [output_grid[0][c]] * size
                            break
                elif edge == 'bottom':
                    for c in range(size):
                        if output_grid[size-1][c] != 0:
                            output_grid[size-1] = [output_grid[size-1][c]] * size
                            break
                elif edge == 'left':
                    for r in range(size):
                        if output_grid[r][0] != 0:
                            for i in range(size):
                                output_grid[i][0] = output_grid[r][0]
                            break
                elif edge == 'right':
                    for r in range(size):
                        if output_grid[r][size-1] != 0:
                            for i in range(size):
                                output_grid[i][size-1] = output_grid[r][size-1]
                            break

    def cleanup_isolated_cells(output_grid: List[List[int]]) -> None:
        size = len(output_grid)
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                if output_grid[r][c] == 0:
                    neighbors = [
                        output_grid[r-1][c], output_grid[r+1][c],
                        output_grid[r][c-1], output_grid[r][c+1]
                    ]
                    if all(neighbor != 0 for neighbor in neighbors):
                        output_grid[r][c] = max(set(neighbors), key=neighbors.count)

    # Main function logic
    analysis = analyze_input(input_grid)
    input_size = input_grid.get_dimensions()
    output_grid = create_output_grid(input_size)
    quadrant_size = max(input_size)
    top_left_quadrant = create_top_left_quadrant(input_grid, quadrant_size, analysis)
    output_grid = apply_rotational_symmetry(top_left_quadrant)
    handle_corners_and_edges(output_grid, analysis)
    cleanup_isolated_cells(output_grid)

    return ColoredGrid(values=output_grid)
