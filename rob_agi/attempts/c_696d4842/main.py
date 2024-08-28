from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_696d4842(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a series of operations:
    1. Identify colored elements and categorize them.
    2. Apply edge transformations based on a color cycle.
    3. Extend vertical and horizontal lines.
    4. Transform and extend shapes.
    5. Handle isolated dots.
    6. Resolve intersections based on line/shape length.
    7. Propagate colors to fill enclosed areas.
    8. Apply secondary color transformations.
    9. Fill remaining empty spaces adjacent to colored cells.
    10. Perform final cleanup and validation.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    color_cycle = {4: 2, 2: 6, 6: 1, 1: 3, 3: 8, 8: 4}  # Yellow -> Red -> Magenta -> Blue -> Green -> Sky -> Yellow

    def identify_elements():
        vertical = []
        horizontal = []
        shapes = []
        isolated = []
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] != 0:
                    if r > 0 and output_grid.values[r-1][c] == output_grid.values[r][c]:
                        vertical.append((r, c))
                    elif c > 0 and output_grid.values[r][c-1] == output_grid.values[r][c]:
                        horizontal.append((r, c))
                    elif any(output_grid.values[nr][nc] == output_grid.values[r][c] 
                             for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                             if 0 <= nr < rows and 0 <= nc < cols):
                        shapes.append((r, c))
                    else:
                        isolated.append((r, c))
        return vertical, horizontal, shapes, isolated

    def apply_edge_transformations():
        for i in range(rows):
            if output_grid.values[i][0] != 0:
                output_grid.values[i][0] = color_cycle.get(output_grid.values[i][0], output_grid.values[i][0])
        for j in range(1, cols):
            if output_grid.values[0][j] != 0:
                output_grid.values[0][j] = color_cycle.get(output_grid.values[0][j], output_grid.values[0][j])

    def extend_line(r, c, direction):
        color = output_grid.values[r][c]
        if direction in ['up', 'down']:
            step = -1 if direction == 'up' else 1
            for i in range(r + step, -1 if direction == 'up' else rows, step):
                if output_grid.values[i][c] == 0:
                    output_grid.values[i][c] = color
                else:
                    break
        else:  # left or right
            step = -1 if direction == 'left' else 1
            for j in range(c + step, -1 if direction == 'left' else cols, step):
                if output_grid.values[r][j] == 0:
                    output_grid.values[r][j] = color
                else:
                    break

    def extend_shape(r, c):
        color = output_grid.values[r][c]
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if output_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = color
                        stack.append((nr, nc))
                    elif output_grid.values[nr][nc] == color:
                        stack.append((nr, nc))

    def handle_isolated(r, c):
        color = output_grid.values[r][c]
        if r <= 1 or r >= rows - 2 or c <= 1 or c >= cols - 2:
            extend_line(r, c, 'up')
            extend_line(r, c, 'down')
            extend_line(r, c, 'left')
            extend_line(r, c, 'right')
        else:
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            colored_neighbors = [n for n in neighbors if output_grid.values[n[0]][n[1]] != 0]
            if colored_neighbors:
                output_grid.values[r][c] = output_grid.values[colored_neighbors[0][0]][colored_neighbors[0][1]]
            else:
                extend_line(r, c, 'up')
                extend_line(r, c, 'down')
                extend_line(r, c, 'left')
                extend_line(r, c, 'right')

    def resolve_intersections():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] != 0:
                    vertical_length = sum(1 for i in range(rows) if output_grid.values[i][c] == output_grid.values[r][c])
                    horizontal_length = sum(1 for j in range(cols) if output_grid.values[r][j] == output_grid.values[r][c])
                    if vertical_length > horizontal_length:
                        for j in range(cols):
                            if output_grid.values[r][j] != 0:
                                output_grid.values[r][j] = output_grid.values[r][c]
                    elif horizontal_length > vertical_length:
                        for i in range(rows):
                            if output_grid.values[i][c] != 0:
                                output_grid.values[i][c] = output_grid.values[r][c]

    def propagate_colors():
        changed = True
        while changed:
            changed = False
            for r in range(rows):
                for c in range(cols):
                    if output_grid.values[r][c] != 0:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] == 0:
                                output_grid.values[nr][nc] = output_grid.values[r][c]
                                changed = True

    def apply_secondary_transformations():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == 3:  # Green
                    if any(output_grid.values[nr][nc] == 3 for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] if 0 <= nr < rows and 0 <= nc < cols):
                        output_grid.values[r][c] = 8  # Sky Blue
                elif output_grid.values[r][c] == 4 and r > 0 and output_grid.values[r-1][c] == 4:  # Yellow at top of vertical line
                    output_grid.values[r][c] = 2  # Red

    def fill_empty_spaces():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == 0:
                    neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                    valid_neighbors = [(nr, nc) for nr, nc in neighbors if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] != 0]
                    if valid_neighbors:
                        most_common_color = max(set(output_grid.values[nr][nc] for nr, nc in valid_neighbors), key=lambda x: valid_neighbors.count((output_grid.values[nr][nc] for nr, nc in valid_neighbors).index(x)))
                        output_grid.values[r][c] = most_common_color

    # Main transformation process
    apply_edge_transformations()
    vertical, horizontal, shapes, isolated = identify_elements()
    
    for r, c in vertical:
        extend_line(r, c, 'up')
        extend_line(r, c, 'down')
    
    for r, c in horizontal:
        extend_line(r, c, 'left')
        extend_line(r, c, 'right')
    
    for r, c in shapes:
        extend_shape(r, c)
    
    for r, c in isolated:
        handle_isolated(r, c)
    
    resolve_intersections()
    propagate_colors()
    apply_secondary_transformations()
    fill_empty_spaces()

    # Final cleanup
    apply_edge_transformations()
    propagate_colors()

    return output_grid
