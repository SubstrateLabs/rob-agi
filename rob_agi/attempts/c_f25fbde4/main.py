from rob_agi.colored_grid import ColoredGrid

def solve_f25fbde4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting a core pattern, expanding it,
    and applying horizontal mirroring to create a 6x8 output grid.
    
    1. Extract the core pattern containing yellow (4) cells.
    2. Expand each cell of the core pattern to a 2x2 block.
    3. Apply horizontal mirroring to complete the pattern.
    4. Adjust the size to ensure a 6x8 output grid.
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
        if width <= 4:
            return [row + row[::-1] for row in grid]
        return grid

    def adjust_size(grid):
        height, width = len(grid), len(grid[0])
        if width > 8:
            return [row[:8] for row in grid[:6]]
        elif width < 8:
            return [row + [0] * (8 - width) for row in grid] + [[0] * 8] * (6 - height)
        elif height < 6:
            return grid + [[0] * 8] * (6 - height)
        return grid[:6]

    core = extract_core(input_grid.values)
    expanded = expand_core(core)
    mirrored = mirror_horizontally(expanded)
    final_grid = adjust_size(mirrored)

    return ColoredGrid(values=final_grid)
