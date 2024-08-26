from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2697da3f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger, symmetrical pattern.
    
    The transformation involves:
    1. Centering the input pattern in a square grid of size max(width, height).
    2. Creating a larger output grid of size (2n-1) x (2n-1) where n is the max dimension.
    3. Mapping the centered input to the top-left quadrant of the output grid.
    4. Applying rotational symmetry to create the full output grid.
    5. Extending the pattern to touch all edges if the original input touched any edge.
    6. Cleaning up any isolated black cells for pattern consistency.
    7. Ensuring perfect rotational symmetry and edge-touching in the final output.
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

    def create_centered_square_grid(input_grid: ColoredGrid) -> List[List[int]]:
        input_rows, input_cols = input_grid.get_dimensions()
        max_dim = max(input_rows, input_cols)
        centered_grid = [[0 for _ in range(max_dim)] for _ in range(max_dim)]
        
        start_row = (max_dim - input_rows) // 2
        start_col = (max_dim - input_cols) // 2
        
        for r in range(input_rows):
            for c in range(input_cols):
                centered_grid[start_row + r][start_col + c] = input_grid.values[r][c]
        
        return centered_grid

    def create_output_grid(max_dim: int) -> List[List[int]]:
        output_size = max_dim * 2 - 1
        return [[0 for _ in range(output_size)] for _ in range(output_size)]

    def map_to_top_left_quadrant(centered_grid: List[List[int]], output_grid: List[List[int]]) -> None:
        centered_size = len(centered_grid)
        output_size = len(output_grid)
        start = (output_size - centered_size) // 2
        
        for r in range(centered_size):
            for c in range(centered_size):
                output_grid[start + r][start + c] = centered_grid[r][c]

    def apply_rotational_symmetry(output_grid: List[List[int]]) -> None:
        size = len(output_grid)
        half = size // 2
        
        for r in range(half + 1):
            for c in range(half + 1):
                if output_grid[r][c] != 0:
                    # Top-right quadrant
                    output_grid[r][size-1-c] = output_grid[r][c]
                    # Bottom-left quadrant
                    output_grid[size-1-r][c] = output_grid[r][c]
                    # Bottom-right quadrant
                    output_grid[size-1-r][size-1-c] = output_grid[r][c]

    def extend_to_edges(output_grid: List[List[int]], input_grid: ColoredGrid) -> None:
        size = len(output_grid)
        half = size // 2
        input_rows, input_cols = input_grid.get_dimensions()
        
        # Check if input touches edges and extend if necessary
        if any(input_grid.values[0][c] != 0 for c in range(input_cols)):
            color = next(input_grid.values[0][c] for c in range(input_cols) if input_grid.values[0][c] != 0)
            for c in range(size):
                output_grid[0][c] = color
        if any(input_grid.values[input_rows-1][c] != 0 for c in range(input_cols)):
            color = next(input_grid.values[input_rows-1][c] for c in range(input_cols) if input_grid.values[input_rows-1][c] != 0)
            for c in range(size):
                output_grid[size-1][c] = color
        if any(input_grid.values[r][0] != 0 for r in range(input_rows)):
            color = next(input_grid.values[r][0] for r in range(input_rows) if input_grid.values[r][0] != 0)
            for r in range(size):
                output_grid[r][0] = color
        if any(input_grid.values[r][input_cols-1] != 0 for r in range(input_rows)):
            color = next(input_grid.values[r][input_cols-1] for r in range(input_rows) if input_grid.values[r][input_cols-1] != 0)
            for r in range(size):
                output_grid[r][size-1] = color

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
    input_size = input_grid.get_dimensions()
    max_dim = max(input_size)
    centered_grid = create_centered_square_grid(input_grid)
    output_grid = create_output_grid(max_dim)
    map_to_top_left_quadrant(centered_grid, output_grid)
    apply_rotational_symmetry(output_grid)
    extend_to_edges(output_grid, input_grid)
    cleanup_isolated_cells(output_grid)

    return ColoredGrid(values=output_grid)
