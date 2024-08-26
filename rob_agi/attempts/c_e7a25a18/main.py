from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7a25a18(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by identifying a red frame and four colored squares within it,
    then creates a new grid where the frame is preserved and the colored squares are expanded
    to fill equal quadrants within the frame.

    1. Identifies the red (2) frame in the input grid.
    2. Locates the four non-black, non-red colored squares within the frame.
    3. Creates a new grid with dimensions matching the red frame.
    4. Reconstructs the red frame in the new grid.
    5. Expands each colored square to fill a quadrant within the frame.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with expanded color quadrants within the red frame.
    """
    # Find the red frame boundaries
    frame = find_frame(input_grid)
    if not frame:
        return input_grid  # Return original if no frame found

    # Extract frame dimensions
    top, left, bottom, right = frame
    frame_height = bottom - top + 1
    frame_width = right - left + 1

    # Find the four colors
    colors = find_colors(input_grid, frame)

    # Create new grid
    new_grid = [[2 for _ in range(frame_width)] for _ in range(frame_height)]

    # Calculate inner quadrant dimensions
    inner_height = (frame_height - 2) // 2
    inner_width = (frame_width - 2) // 2

    # Fill inner quadrants
    fill_quadrant(new_grid, 1, 1, inner_height, inner_width, colors[0])
    fill_quadrant(new_grid, 1, inner_width + 1, inner_height, frame_width - 2, colors[1])
    fill_quadrant(new_grid, inner_height + 1, 1, frame_height - 2, inner_width, colors[2])
    fill_quadrant(new_grid, inner_height + 1, inner_width + 1, frame_height - 2, frame_width - 2, colors[3])

    return ColoredGrid(values=new_grid)

def find_frame(grid: ColoredGrid) -> Tuple[int, int, int, int]:
    """Finds the boundaries of the red frame."""
    rows, cols = grid.get_dimensions()
    top = next((r for r in range(rows) if 2 in grid.values[r]), None)
    bottom = next((r for r in range(rows - 1, -1, -1) if 2 in grid.values[r]), None)
    left = next((c for c in range(cols) if any(grid.values[r][c] == 2 for r in range(rows))), None)
    right = next((c for c in range(cols - 1, -1, -1) if any(grid.values[r][c] == 2 for r in range(rows))), None)
    
    if top is None or bottom is None or left is None or right is None:
        return None
    return (top, left, bottom, right)

def find_colors(grid: ColoredGrid, frame: Tuple[int, int, int, int]) -> List[int]:
    """Finds the four colors within the frame."""
    top, left, bottom, right = frame
    colors = []
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            color = grid.values[r][c]
            if color not in [0, 2] and color not in colors:
                colors.append(color)
                if len(colors) == 4:
                    return colors
    return colors

def fill_quadrant(grid: List[List[int]], top: int, left: int, bottom: int, right: int, color: int):
    """Fills a quadrant of the grid with a specific color."""
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            grid[r][c] = color
