from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_a3f84088(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a nested outline pattern.
    
    The function does the following:
    1. Analyzes the input grid to determine its size and outer boundary.
    2. Selects an appropriate template based on the grid size.
    3. Applies the template, creating nested outlines of different colors.
    4. Adjusts the pattern if necessary to fit the grid size.
    5. Returns the transformed grid.

    The pattern typically consists of an outer gray outline, followed by
    alternating red and gray outlines moving inward, with a specific
    center pattern that varies based on the grid size.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # Find the boundaries of the outer gray outline
    top, left, bottom, right = find_outer_boundary(new_grid)

    # Select and apply the appropriate template
    template = select_template(rows, cols)
    apply_template(new_grid, top, left, bottom, right, template)

    return new_grid

def find_outer_boundary(grid: ColoredGrid) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    top = next(r for r in range(rows) if 5 in grid.values[r])
    bottom = next(r for r in range(rows-1, -1, -1) if 5 in grid.values[r])
    left = min(grid.values[r].index(5) for r in range(top, bottom+1))
    right = max(cols - 1 - grid.values[r][::-1].index(5) for r in range(top, bottom+1))
    return top, left, bottom, right

def select_template(rows: int, cols: int) -> Dict:
    # Simplified template selection based on grid size
    if rows <= 6 and cols <= 6:
        return {
            "outlines": [
                {"color": 5, "thickness": 1},
                {"color": 2, "thickness": 1},
            ],
            "center": {"type": "solid", "color": 5, "size": (2, 2)}
        }
    else:
        return {
            "outlines": [
                {"color": 5, "thickness": 1},
                {"color": 2, "thickness": 1},
                {"color": 5, "thickness": 1},
            ],
            "center": {"type": "solid", "color": 0, "size": (3, 3)}
        }

def apply_template(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, template: Dict):
    for outline in template["outlines"]:
        draw_outline(grid, top, left, bottom, right, outline["color"])
        top += outline["thickness"]
        left += outline["thickness"]
        bottom -= outline["thickness"]
        right -= outline["thickness"]
    
    # Apply center pattern
    center = template["center"]
    center_height = bottom - top + 1
    center_width = right - left + 1
    if center["type"] == "solid" and center_height > 0 and center_width > 0:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                grid.values[r][c] = center["color"]

def draw_outline(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int):
    for c in range(left, right + 1):
        grid.values[top][c] = color
        grid.values[bottom][c] = color
    for r in range(top + 1, bottom):
        grid.values[r][left] = color
        grid.values[r][right] = color
