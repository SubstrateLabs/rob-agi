from rob_agi.colored_grid import ColoredGrid

def solve_f25fbde4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting a core pattern, expanding it,
    and applying horizontal mirroring to create a 6x8 output grid.
    
    1. Extract the core pattern containing yellow (4) cells.
    2. Expand each cell of the core pattern to a 2x2 block.
    3. Apply horizontal mirroring to complete the pattern.
    4. Adjust the size to ensure a 6x8 output grid, preserving the pattern.
    5. If the pattern is smaller than 6x8, center it within the grid.
    6. If the pattern is larger than 6x8, crop it to fit.
    """
    def extract_core(grid):
        rows, cols = len(grid), len(grid[0])
        top, left, bottom, right = rows, cols, 0, 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 4:
                    top, left = min(top, r), min(left, c)
                    bottom, right = max(bottom, r), max(right, c)
        return [row[left:right+1] for row in grid[top:bottom+1]]

    def expand_core(core):
        return [[cell for cell in row for _ in range(2)] for row in core for _ in range(2)]

    def mirror_horizontally(grid):
        width = len(grid[0])
        return [row[:width//2] + row[:width//2][::-1] for row in grid]

    def adjust_size(grid):
        height, width = len(grid), len(grid[0])
        new_grid = [[0] * 8 for _ in range(6)]
        
        start_row = (6 - min(height, 6)) // 2
        start_col = (8 - min(width, 8)) // 2
        
        for r in range(min(height, 6)):
            for c in range(min(width, 8)):
                new_grid[start_row + r][start_col + c] = grid[r][c]
        
        return new_grid

    core = extract_core(input_grid.values)
    expanded = expand_core(core)
    mirrored = mirror_horizontally(expanded)
    final_grid = adjust_size(mirrored)

    return ColoredGrid(values=final_grid)
