from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_09c534e7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying complex color progression to shapes while preserving structure.
    
    The transformation follows these steps:
    1. Analyze the grid to identify distinct shapes using flood-fill.
    2. Categorize shapes based on size: small (< 9 cells), medium (9-25 cells), large (> 25 cells).
    3. Process each shape based on its category and special colored cells:
       - Small shapes: Increase colors uniformly by 1-2 steps.
       - Medium shapes: Create a simple gradient from edge to center.
       - Large shapes: Create a complex gradient with multiple local maxima.
    4. Handle shape borders and inter-shape interactions for smooth transitions.
    5. Expand the influence of special colored cells based on their value and shape size.
    6. Apply global color adjustments to ensure consistency and no color decreases.
    7. Handle complex structures like connected but distinct shapes.
    8. Perform a final consistency check to preserve overall structure and color progression patterns.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 1. Grid Analysis
    shapes = identify_and_categorize_shapes(output_grid)
    
    # 2-3. Shape Processing
    for shape, category in shapes:
        process_shape(output_grid, shape, category, color_sequence)
    
    # 4. Border Handling and Inter-shape Interaction
    handle_borders_and_interactions(output_grid, shapes)
    
    # 5. Special Cell Expansion
    expand_special_cells(output_grid, shapes)
    
    # 6. Global Color Adjustment
    global_color_adjustment(output_grid, input_grid)
    
    # 7. Complex Structure Handling
    handle_complex_structures(output_grid, shapes)
    
    # 8. Final Consistency Check
    final_consistency_check(output_grid, input_grid)
    
    return output_grid

def identify_and_categorize_shapes(grid: ColoredGrid) -> List[Tuple[Set[Tuple[int, int]], str]]:
    shapes = []
    visited = set()
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                shape = flood_fill(grid, r, c, visited)
                category = categorize_shape(shape)
                shapes.append((shape, category))
    
    return shapes

def flood_fill(grid: ColoredGrid, r: int, c: int, visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    color = grid.values[r][c]
    shape = set()
    stack = [(r, c)]
    
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == color:
            visited.add((r, c))
            shape.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((r + dr, c + dc))
    
    return shape

def categorize_shape(shape: Set[Tuple[int, int]]) -> str:
    size = len(shape)
    if size < 9:
        return "small"
    elif 9 <= size <= 25:
        return "medium"
    else:
        return "large"

def process_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], category: str, color_sequence: List[int]):
    if category == "small":
        process_small_shape(grid, shape, color_sequence)
    elif category == "medium":
        process_medium_shape(grid, shape, color_sequence)
    else:
        process_large_shape(grid, shape, color_sequence)

def process_small_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], color_sequence: List[int]):
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    target_color = next_color_in_sequence(current_color, color_sequence, 1 + len(shape) // 3)
    for r, c in shape:
        grid.values[r][c] = target_color

def process_medium_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], color_sequence: List[int]):
    border = find_border(shape)
    center = find_center(shape)
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    max_distance = max(max(abs(r - center[0]), abs(c - center[1])) for r, c in shape)
    
    for r, c in shape:
        if (r, c) in border:
            grid.values[r][c] = current_color
        else:
            distance = max(abs(r - center[0]), abs(c - center[1]))
            steps = int((max_distance - distance) / max_distance * 3) + 1
            grid.values[r][c] = next_color_in_sequence(current_color, color_sequence, steps)

def process_large_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], color_sequence: List[int]):
    border = find_border(shape)
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    local_maxima = find_local_maxima(shape, 3)
    
    for r, c in shape:
        if (r, c) in border:
            grid.values[r][c] = current_color
        else:
            distances = [((r-mr)**2 + (c-mc)**2)**0.5 for mr, mc in local_maxima]
            min_distance = min(distances)
            steps = int((1 - min_distance / max(distances)) * 5) + 1
            grid.values[r][c] = next_color_in_sequence(current_color, color_sequence, steps)

def handle_borders_and_interactions(grid: ColoredGrid, shapes: List[Tuple[Set[Tuple[int, int]], str]]):
    for shape, _ in shapes:
        border = find_border(shape)
        for r, c in border:
            neighbors = [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]]
            valid_neighbors = [(nr, nc) for nr, nc in neighbors if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols]
            neighbor_colors = [grid.values[nr][nc] for nr, nc in valid_neighbors if (nr, nc) not in shape]
            if neighbor_colors:
                grid.values[r][c] = min(grid.values[r][c], max(neighbor_colors) + 1)

def expand_special_cells(grid: ColoredGrid, shapes: List[Tuple[Set[Tuple[int, int]], str]]):
    for shape, category in shapes:
        special_cells = [(r, c) for r, c in shape if grid.values[r][c] > 1]
        for sr, sc in special_cells:
            color = grid.values[sr][sc]
            influence_radius = len(shape) // 4 if category == "large" else len(shape) // 2
            for r, c in shape:
                distance = ((r-sr)**2 + (c-sc)**2)**0.5
                if distance <= influence_radius:
                    influence = int((1 - distance/influence_radius) * color)
                    grid.values[r][c] = max(grid.values[r][c], influence)

def global_color_adjustment(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            output_grid.values[r][c] = max(output_grid.values[r][c], input_grid.values[r][c])
    
    for r in range(1, output_grid.num_rows - 1):
        for c in range(1, output_grid.num_cols - 1):
            neighbors = [output_grid.values[r+dr][c+dc] for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]]
            if max(neighbors) - min(neighbors) > 2:
                output_grid.values[r][c] = (max(neighbors) + min(neighbors)) // 2

def handle_complex_structures(grid: ColoredGrid, shapes: List[Tuple[Set[Tuple[int, int]], str]]):
    for i, (shape1, _) in enumerate(shapes):
        for shape2, _ in shapes[i+1:]:
            if are_shapes_connected(shape1, shape2):
                smooth_transition(grid, shape1, shape2)

def final_consistency_check(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            if input_grid.values[r][c] == 0:
                assert output_grid.values[r][c] == 0, f"Structure not preserved at ({r}, {c})"
            else:
                assert output_grid.values[r][c] >= input_grid.values[r][c], f"Color decreased at ({r}, {c})"

def find_border(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    border = set()
    for r, c in shape:
        if any((r+dr, c+dc) not in shape for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]):
            border.add((r, c))
    return border

def find_center(shape: Set[Tuple[int, int]]) -> Tuple[int, int]:
    rs, cs = zip(*shape)
    return (sum(rs) // len(rs), sum(cs) // len(cs))

def next_color_in_sequence(color: int, color_sequence: List[int], steps: int = 1) -> int:
    if color not in color_sequence:
        return color
    index = color_sequence.index(color)
    return color_sequence[(index + steps) % len(color_sequence)]

def find_local_maxima(shape: Set[Tuple[int, int]], num_maxima: int) -> List[Tuple[int, int]]:
    rs, cs = zip(*shape)
    maxima = []
    for _ in range(num_maxima):
        r = random.randint(min(rs), max(rs))
        c = random.randint(min(cs), max(cs))
        maxima.append((r, c))
    return maxima

def are_shapes_connected(shape1: Set[Tuple[int, int]], shape2: Set[Tuple[int, int]]) -> bool:
    return any((r+dr, c+dc) in shape2 for r, c in shape1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)])

def smooth_transition(grid: ColoredGrid, shape1: Set[Tuple[int, int]], shape2: Set[Tuple[int, int]]):
    border1 = find_border(shape1)
    border2 = find_border(shape2)
    transition_cells = border1.intersection(border2)
    for r, c in transition_cells:
        neighbors = [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]]
        valid_neighbors = [(nr, nc) for nr, nc in neighbors if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols]
        neighbor_colors = [grid.values[nr][nc] for nr, nc in valid_neighbors]
        grid.values[r][c] = sum(neighbor_colors) // len(neighbor_colors)
