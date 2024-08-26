from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2697da3f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger, symmetrical pattern.
    
    The transformation involves:
    1. Analyzing the input pattern to extract its core.
    2. Creating a larger output grid of size (2n-1) x (2n-1) where n is the max dimension of the input.
    3. Scaling and centering the core pattern in the output grid.
    4. Applying rotational and reflectional symmetry to create a complex, symmetrical design.
    5. Refining the pattern by filling gaps and ensuring consistency.
    6. Creating a central void if the input has low density.
    7. Extending the pattern to touch all edges if the original input touched any edge.
    8. Making final adjustments for perfect symmetry and pattern flow.
    """
    # Step 1: Analyze the input grid
    input_analysis = analyze_input(input_grid)
    core_pattern = extract_core_pattern(input_grid)

    # Step 2: Determine the output grid size
    max_dim = max(input_grid.get_dimensions())
    output_size = max_dim * 2 - 1

    # Step 3: Scale and center the core pattern
    scaled_core = scale_core_pattern(core_pattern, output_size // 2)
    output_grid = center_pattern(scaled_core, output_size)

    # Step 4: Apply symmetry
    output_grid = apply_symmetry(output_grid)

    # Step 5: Refine the pattern
    refine_pattern(output_grid)

    # Step 6: Create central void if necessary
    if input_analysis['density'] < 0.3:
        create_central_void(output_grid)

    # Step 7: Extend to edges if necessary
    if any(input_analysis['edge_touches'].values()):
        extend_to_edges(output_grid)

    # Step 8: Final adjustments
    final_adjustments(output_grid)

    # Return the final ColoredGrid
    return ColoredGrid(values=output_grid)

def analyze_input(grid: ColoredGrid) -> dict:
    rows, cols = grid.get_dimensions()
    colored_cells = []
    edge_touches = {'top': set(), 'left': set(), 'bottom': set(), 'right': set()}
    total_cells = rows * cols
    colored_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                colored_cells.append((r, c, grid.values[r][c]))
                colored_count += 1
                if r == 0:
                    edge_touches['top'].add(c)
                if r == rows - 1:
                    edge_touches['bottom'].add(c)
                if c == 0:
                    edge_touches['left'].add(r)
                if c == cols - 1:
                    edge_touches['right'].add(r)

    return {
        'colored_cells': colored_cells,
        'edge_touches': edge_touches,
        'density': colored_count / total_cells
    }

def extract_core_pattern(input_grid: ColoredGrid) -> List[List[int]]:
    rows, cols = input_grid.get_dimensions()
    min_row, min_col = rows, cols
    max_row, max_col = 0, 0

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                min_row = min(min_row, r)
                min_col = min(min_col, c)
                max_row = max(max_row, r)
                max_col = max(max_col, c)

    core_pattern = [
        [input_grid.values[r][c] for c in range(min_col, max_col + 1)]
        for r in range(min_row, max_row + 1)
    ]
    return core_pattern

def scale_core_pattern(core_pattern: List[List[int]], target_size: int) -> List[List[int]]:
    core_rows, core_cols = len(core_pattern), len(core_pattern[0])
    scale_factor = max(1, target_size // max(core_rows, core_cols))
    
    return [[cell for cell in row for _ in range(scale_factor)] for row in core_pattern for _ in range(scale_factor)]

def center_pattern(pattern: List[List[int]], output_size: int) -> List[List[int]]:
    centered = [[0 for _ in range(output_size)] for _ in range(output_size)]
    pattern_size = len(pattern)
    start = (output_size - pattern_size) // 2
    
    for r in range(pattern_size):
        for c in range(pattern_size):
            centered[start + r][start + c] = pattern[r][c]
    
    return centered

def apply_symmetry(grid: List[List[int]]) -> List[List[int]]:
    size = len(grid)
    symmetrical = [[0 for _ in range(size)] for _ in range(size)]
    
    for r in range(size):
        for c in range(size):
            if grid[r][c] != 0:
                symmetrical[r][c] = grid[r][c]
                symmetrical[r][size-1-c] = grid[r][c]
                symmetrical[size-1-r][c] = grid[r][c]
                symmetrical[size-1-r][size-1-c] = grid[r][c]
    
    return symmetrical

def refine_pattern(grid: List[List[int]]) -> None:
    size = len(grid)
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            neighbors = [
                grid[r-1][c], grid[r+1][c],
                grid[r][c-1], grid[r][c+1]
            ]
            if grid[r][c] == 0 and all(n != 0 for n in neighbors):
                grid[r][c] = max(set(neighbors), key=neighbors.count)

def create_central_void(grid: List[List[int]]) -> None:
    size = len(grid)
    void_size = max(3, size // 5)
    start = (size - void_size) // 2
    for r in range(start, start + void_size):
        for c in range(start, start + void_size):
            grid[r][c] = 0

def extend_to_edges(grid: List[List[int]]) -> None:
    size = len(grid)
    for r in range(size):
        if grid[r][0] == 0:
            grid[r][0] = grid[r][size//2]
        if grid[r][size-1] == 0:
            grid[r][size-1] = grid[r][size//2]
    for c in range(size):
        if grid[0][c] == 0:
            grid[0][c] = grid[size//2][c]
        if grid[size-1][c] == 0:
            grid[size-1][c] = grid[size//2][c]

def final_adjustments(grid: List[List[int]]) -> None:
    size = len(grid)
    # Ensure corner cells are black
    grid[0][0] = grid[0][size-1] = grid[size-1][0] = grid[size-1][size-1] = 0
    
    # Ensure perfect symmetry
    for r in range(size):
        for c in range(size):
            if grid[r][c] != 0:
                grid[r][c] = grid[c][r] = grid[size-1-r][c] = grid[r][size-1-c] = grid[size-1-r][size-1-c]
def scale_core_pattern(core_pattern: List[List[int]], target_size: int) -> List[List[int]]:
    scale_factor = max(1, target_size // max(len(core_pattern), len(core_pattern[0])))
    return [[cell for cell in row for _ in range(scale_factor)] for row in core_pattern for _ in range(scale_factor)]

def create_quadrupled_pattern(core: List[List[int]], output_size: int) -> List[List[int]]:
    output = [[0 for _ in range(output_size)] for _ in range(output_size)]
    core_size = len(core)
    for r in range(core_size):
        for c in range(core_size):
            # Top-left quadrant
            output[r][c] = core[r][c]
            # Top-right quadrant
            output[r][output_size-1-c] = core[r][c]
            # Bottom-left quadrant
            output[output_size-1-r][c] = core[r][c]
            # Bottom-right quadrant
            output[output_size-1-r][output_size-1-c] = core[r][c]
    return output

def enhance_symmetry(grid: List[List[int]]) -> None:
    size = len(grid)
    for r in range(size):
        for c in range(size):
            if grid[r][c] != 0:
                grid[size-1-r][c] = grid[r][size-1-c] = grid[size-1-r][size-1-c] = grid[r][c]

def create_central_void(grid: List[List[int]]) -> None:
    size = len(grid)
    void_size = 3 if size < 15 else 5
    start = (size - void_size) // 2
    for r in range(start, start + void_size):
        for c in range(start, start + void_size):
            grid[r][c] = 0

def extend_to_edges(grid: List[List[int]]) -> None:
    size = len(grid)
    for i in range(size):
        if grid[i][0] == 0:
            grid[i][0] = grid[i][size//2]
        if grid[i][size-1] == 0:
            grid[i][size-1] = grid[i][size//2]
        if grid[0][i] == 0:
            grid[0][i] = grid[size//2][i]
        if grid[size-1][i] == 0:
            grid[size-1][i] = grid[size//2][i]

def set_corner_cells_black(grid: List[List[int]]) -> None:
    size = len(grid)
    grid[0][0] = grid[0][size-1] = grid[size-1][0] = grid[size-1][size-1] = 0

def final_adjustments(grid: List[List[int]]) -> None:
    size = len(grid)
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            neighbors = [
                grid[r-1][c], grid[r+1][c],
                grid[r][c-1], grid[r][c+1]
            ]
            if grid[r][c] == 0 and all(n != 0 for n in neighbors):
                grid[r][c] = max(set(neighbors), key=neighbors.count)
