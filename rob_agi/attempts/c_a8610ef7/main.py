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
    that maintains connectivity of regions while alternating colors.
    """
    output = []
    for r, row in enumerate(input_grid.values):
        new_row = []
        for c, cell in enumerate(row):
            if cell == 0:
                new_row.append(0)
            elif cell == 8:
                # Check if we're at the edge of an 8-region
                is_edge = any(
                    0 <= nr < len(input_grid.values) and 0 <= nc < len(row) and input_grid.values[nr][nc] != 8
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                )
                if is_edge:
                    # Use 5 for even sum of coordinates, 2 for odd
                    new_row.append(5 if (r + c) % 2 == 0 else 2)
                else:
                    # For interior cells, use the opposite pattern
                    new_row.append(2 if (r + c) % 2 == 0 else 5)
            else:
                new_row.append(cell)
        output.append(new_row)
    return ColoredGrid(values=output)
