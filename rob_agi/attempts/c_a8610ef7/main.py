from rob_agi.colored_grid import ColoredGrid

def solve_a8610ef7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing sky blue (8) regions with red (2) and gray (5) colors.
    The algorithm follows these rules:
    1. Non-8 colors are preserved.
    2. Isolated 8-cells become red (2).
    3. Edges of 8-regions are always gray (5).
    4. Interior cells of 8-regions follow a global checkerboard pattern of red (2) and gray (5).
    5. The global checkerboard pattern ensures the top-left cell of the grid is gray (5).
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
    
    def get_checkerboard_color(r, c):
        return 5 if (r + c) % 2 == 0 else 2
    
    # First pass: Mark edges and isolated cells
    for r in range(rows):
        new_row = []
        for c in range(cols):
            cell = input_grid.values[r][c]
            if cell != 8:
                new_row.append(cell)
            elif is_isolated(r, c):
                new_row.append(2)
            elif is_edge(r, c):
                new_row.append(5)
            else:
                new_row.append(8)  # Temporarily keep interior cells as 8
        output.append(new_row)
    
    # Second pass: Apply checkerboard pattern to interior cells
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 8:
                output[r][c] = get_checkerboard_color(r, c)
    
    return ColoredGrid(values=output)
