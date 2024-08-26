from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_09c534e7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying color progression to shapes while preserving structure.
    
    The transformation follows these rules:
    1. Identify distinct shapes in the grid using flood-fill.
    2. Categorize shapes based on size and structure (e.g., large, medium, small, special).
    3. Apply color progression rules based on shape category:
       - Large shapes: Create internal gradients with higher values towards the center.
       - Medium shapes: Increase color values while maintaining overall structure.
       - Small shapes: Maintain structure or increase color value uniformly.
       - Special structures (e.g., L-shapes): Apply specific rules.
    4. Preserve shape borders, typically keeping them at the original color value.
    5. Handle high-value color expansion within shapes.
    6. Balance color distribution across the entire grid.
    7. Refine borders for smooth color transitions.
    8. Ensure no decrease in color values from input to output.
    9. Handle edge cases like maintaining zero values and small grid/shape sizes.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Identify and categorize shapes
    shapes = identify_and_categorize_shapes(output_grid)
    
    # Process each shape
    for shape, category in shapes:
        process_shape(output_grid, shape, category, color_sequence)
    
    # Handle high-value color expansion
    expand_high_value_colors(output_grid)
    
    # Balance color distribution
    balance_colors(output_grid)
    
    # Refine borders
    refine_borders(output_grid)
    
    # Ensure no decrease in values
    ensure_no_decrease(input_grid, output_grid)
    
    return output_grid

def identify_and_categorize_shapes(grid: ColoredGrid) -> List[Tuple[Set[Tuple[int, int]], str]]:
    shapes = []
    visited = set()
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                shape = flood_fill(grid, r, c, visited)
                category = categorize_shape(shape, grid)
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

def categorize_shape(shape: Set[Tuple[int, int]], grid: ColoredGrid) -> str:
    size = len(shape)
    if size > 9:
        return "large"
    elif 4 <= size <= 9:
        return "medium"
    elif size < 4:
        return "small"
    # Add more categories here if needed, e.g., "L-shape", "cross", etc.

def process_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], category: str, color_sequence: List[int]):
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    highest_color = max(grid.values[r][c] for r, c in shape)
    
    if category == "large":
        process_large_shape(grid, shape, color_sequence)
    elif category == "medium":
        process_medium_shape(grid, shape, color_sequence)
    elif category == "small":
        process_small_shape(grid, shape, color_sequence)
    # Add more shape processing functions as needed

def process_large_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], color_sequence: List[int]):
    border = find_border(grid, shape)
    center = find_center(shape)
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    target_color = next_color_in_sequence(next_color_in_sequence(current_color, color_sequence), color_sequence)
    
    for r, c in shape:
        if (r, c) in border:
            grid.values[r][c] = current_color
        elif is_adjacent_to_border(r, c, border):
            grid.values[r][c] = next_color_in_sequence(current_color, color_sequence)
        elif (r, c) == center:
            grid.values[r][c] = target_color
        else:
            distance_to_center = max(abs(r - center[0]), abs(c - center[1]))
            color_index = min(distance_to_center, len(color_sequence) - 1)
            grid.values[r][c] = color_sequence[color_index]

def process_medium_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], color_sequence: List[int]):
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    target_color = next_color_in_sequence(current_color, color_sequence)
    border = find_border(grid, shape)
    
    for r, c in shape:
        if (r, c) in border:
            grid.values[r][c] = current_color
        else:
            grid.values[r][c] = target_color

def process_small_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]], color_sequence: List[int]):
    current_color = grid.values[list(shape)[0][0]][list(shape)[0][1]]
    target_color = next_color_in_sequence(current_color, color_sequence)
    
    for r, c in shape:
        grid.values[r][c] = target_color

def next_color_in_sequence(color: int, color_sequence: List[int]) -> int:
    if color not in color_sequence:
        return color
    return color_sequence[(color_sequence.index(color) + 1) % len(color_sequence)]

def find_border(grid: ColoredGrid, shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    border = set()
    for r, c in shape:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in shape or nr < 0 or nr >= grid.num_rows or nc < 0 or nc >= grid.num_cols:
                border.add((r, c))
                break
    return border

def find_center(shape: Set[Tuple[int, int]]) -> Tuple[int, int]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    return ((min_r + max_r) // 2, (min_c + max_c) // 2)

def is_adjacent_to_border(r: int, c: int, border: Set[Tuple[int, int]]) -> bool:
    return any((r + dr, c + dc) in border for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)])

def find_corner_structures(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    # Implementation to find corner structures (L-shapes)
    # This is a placeholder and needs to be implemented
    return []

def process_corner_structure(grid: ColoredGrid, structure: Set[Tuple[int, int]], color_sequence: List[int]):
    # Implementation to process corner structures
    # This is a placeholder and needs to be implemented
    pass

def apply_general_progression(grid: ColoredGrid, color_sequence: List[int]):
    # Implementation to apply general color progression
    # This is a placeholder and needs to be implemented
    pass

def balance_colors(grid: ColoredGrid):
    color_counts = {i: 0 for i in range(1, 10)}
    total_colored_cells = 0
    
    for row in grid.values:
        for cell in row:
            if cell != 0:
                color_counts[cell] += 1
                total_colored_cells += 1
    
    target_count = total_colored_cells // len(color_counts)
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != 0:
                current_color = grid.values[r][c]
                if color_counts[current_color] > target_count:
                    new_color = min(color_counts, key=color_counts.get)
                    if new_color > current_color:
                        grid.values[r][c] = new_color
                        color_counts[current_color] -= 1
                        color_counts[new_color] += 1

def preserve_structure(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] != 0 and output_grid.values[r][c] == 0:
                output_grid.values[r][c] = input_grid.values[r][c]

def refine_borders(grid: ColoredGrid):
    # Implementation to refine borders of shapes
    # This is a placeholder and needs to be implemented
    pass

def find_region(grid: ColoredGrid, r: int, c: int) -> Set[Tuple[int, int]]:
    color = grid.values[r][c]
    region = set()
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        if (r, c) not in region and 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == color:
            region.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((r + dr, c + dc))
    return region

def process_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color_sequence: List[int]):
    color = grid.values[list(region)[0][0]][list(region)[0][1]]
    next_color = color_sequence[(color_sequence.index(color) + 1) % len(color_sequence)]
    border = find_border(grid, region)
    for r, c in region - border:
        grid.values[r][c] = next_color
    for r, c in border:
        grid.values[r][c] = color

def find_border(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    border = set()
    for r, c in region:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region or nr < 0 or nr >= grid.num_rows or nc < 0 or nc >= grid.num_cols:
                border.add((r, c))
                break
    return border

def expand_higher_value_cells(grid: ColoredGrid):
    for r in range(grid.num_rows - 1):
        for c in range(grid.num_cols - 1):
            max_value = max(grid.values[r][c], grid.values[r][c+1], grid.values[r+1][c], grid.values[r+1][c+1])
            if max_value > 1:
                grid.values[r][c] = max_value
                grid.values[r][c+1] = max_value
                grid.values[r+1][c] = max_value
                grid.values[r+1][c+1] = max_value

def ensure_no_decrease(input_grid: ColoredGrid, output_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            output_grid.values[r][c] = max(output_grid.values[r][c], input_grid.values[r][c])

def reconnect_borders(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] > 1:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] == 0:
                        grid.values[nr][nc] = grid.values[r][c] - 1
def expand_high_value_colors(grid: ColoredGrid):
    high_value_threshold = 4
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] >= high_value_threshold:
                expand_color(grid, r, c)

def expand_color(grid: ColoredGrid, r: int, c: int):
    color = grid.values[r][c]
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            if grid.values[nr][nc] != 0 and grid.values[nr][nc] < color:
                grid.values[nr][nc] = color
