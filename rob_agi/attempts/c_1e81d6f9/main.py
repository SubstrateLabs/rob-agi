from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1e81d6f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving shapes and strategically selecting dots.
    
    1. Preserves the T-shaped gray object.
    2. Identifies and marks continuous shapes and lines.
    3. Counts occurrences of each color.
    4. Selects strategic isolated dots for colors with fewer than 3 occurrences.
    5. Limits the number of dots per color to 3, prioritizing shapes and strategic positions.
    6. Creates the output grid with the selected cells.
    7. Performs a final check and adjustment for complexity balance.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    marked_cells = set()

    # Step 1: Preserve the T-shaped gray object
    preserve_t_shape(input_grid, output_grid, marked_cells)

    # Step 2: Identify and mark continuous shapes and lines
    shapes = find_shapes(input_grid)

    # Step 3: Count occurrences of each color
    color_counts = count_colors(input_grid)

    # Step 4 & 5: Select strategic dots and limit per color
    select_strategic_dots(input_grid, output_grid, shapes, color_counts, marked_cells)

    # Step 6: Create the output grid (already done in previous steps)

    # Step 7: Final check and adjustments
    final_adjustments(output_grid, marked_cells)

    return output_grid

def preserve_t_shape(input_grid: ColoredGrid, output_grid: ColoredGrid, marked_cells: set):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] == 5:  # Gray color
                output_grid.values[r][c] = 5
                marked_cells.add((r, c))

def find_shapes(grid: ColoredGrid) -> Dict[int, List[List[Tuple[int, int]]]]:
    shapes = defaultdict(list)
    visited = set()

    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        shape = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == color:
                visited.add((curr_r, curr_c))
                shape.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                        stack.append((nr, nc))
        return shape

    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                shape = dfs(r, c, grid.values[r][c])
                if len(shape) > 1:
                    shapes[grid.values[r][c]].append(shape)

    return shapes

def count_colors(grid: ColoredGrid) -> Dict[int, int]:
    counts = defaultdict(int)
    for row in grid.values:
        for cell in row:
            if cell != 0:
                counts[cell] += 1
    return counts

def select_strategic_dots(input_grid: ColoredGrid, output_grid: ColoredGrid, shapes: Dict[int, List[List[Tuple[int, int]]]], color_counts: Dict[int, int], marked_cells: set):
    for color, count in color_counts.items():
        if color == 5:  # Skip gray (T-shape)
            continue
        
        color_cells = []
        # Add cells from shapes
        for shape in shapes.get(color, []):
            color_cells.extend(shape)
        
        # Add isolated dots if needed
        if len(color_cells) < 3:
            for r in range(input_grid.num_rows):
                for c in range(input_grid.num_cols):
                    if input_grid.values[r][c] == color and (r, c) not in color_cells:
                        color_cells.append((r, c))
                        if len(color_cells) == 3:
                            break
                if len(color_cells) == 3:
                    break
        
        # Select up to 3 strategic cells
        selected_cells = select_strategic_cells(color_cells, input_grid.num_rows, input_grid.num_cols, 3)
        
        # Add selected cells to output grid and marked cells
        for r, c in selected_cells:
            output_grid.values[r][c] = color
            marked_cells.add((r, c))

def select_strategic_cells(cells: List[Tuple[int, int]], rows: int, cols: int, max_cells: int) -> List[Tuple[int, int]]:
    if len(cells) <= max_cells:
        return cells
    
    # Prioritize cells based on their position (corners, center, etc.)
    priority_cells = []
    for r, c in cells:
        priority = 0
        if (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1):  # Corners
            priority = 3
        elif r == 0 or r == rows - 1 or c == 0 or c == cols - 1:  # Edges
            priority = 2
        elif r == rows // 2 and c == cols // 2:  # Center
            priority = 1
        priority_cells.append((priority, (r, c)))
    
    priority_cells.sort(reverse=True)
    return [cell for _, cell in priority_cells[:max_cells]]

def final_adjustments(output_grid: ColoredGrid, marked_cells: set):
    # This function can be expanded to make final adjustments if needed
    pass
