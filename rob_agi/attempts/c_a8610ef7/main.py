from rob_agi.colored_grid import ColoredGrid

def create_checkerboard(rows, cols):
    return [[2 if (r + c) % 2 == 0 else 5 for c in range(cols)] for r in range(rows)]

def count_adjacent_colors(grid, row, col):
    counts = {2: 0, 5: 0}
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] in [2, 5]:
                counts[grid[r][c]] += 1
    return counts

def get_checkerboard_color(row, col):
    return 2 if (row + col) % 2 == 0 else 5

def solve_a8610ef7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing sky blue (8) regions with red (2) and gray (5) colors.
    The algorithm preserves the original structure of non-8 colors and replaces 8s with a pattern
    that maintains connectivity of regions while following these rules:
    1. Edges of 8-regions are always gray (5)
    2. Interior cells alternate between red (2) and gray (5) in a checkerboard pattern
    3. Isolated single 8-cells become red (2)
    """
    output = []
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    
    def is_edge(r, c):
        return any(
            0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 8
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        )
    
    def is_isolated(r, c):
        return all(
            nr < 0 or nr >= rows or nc < 0 or nc >= cols or input_grid.values[nr][nc] != 8
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        )
    
    for r, row in enumerate(input_grid.values):
        new_row = []
        for c, cell in enumerate(row):
            if cell == 0:
                new_row.append(0)
            elif cell == 8:
                if is_isolated(r, c):
                    new_row.append(2)
                elif is_edge(r, c):
                    new_row.append(5)
                else:
                    new_row.append(5 if (r + c) % 2 == 0 else 2)
            else:
                new_row.append(cell)
        output.append(new_row)
    return ColoredGrid(values=output)
