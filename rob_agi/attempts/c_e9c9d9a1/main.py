from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_e9c9d9a1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling sections formed by hierarchical green (3) lines.
    
    The solution:
    1. Identifies all frames formed by green (3) lines, creating a hierarchical structure.
    2. Fills sections based on their position relative to the frames:
       - Outermost corners: red (2), yellow (4), blue (1), sky blue (8)
       - Inside frames: orange (7)
       - Outside frames: remains black (0)
    3. Preserves all green (3) lines and any non-black (0) cells from the input.
    4. Handles complex cases with multiple nested frames and disconnected sections.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    def find_frames(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
        rows, cols = len(grid.values), len(grid.values[0])
        frames = []
        visited = set()

        def dfs(r: int, c: int, frame: List[int]):
            if (r, c) in visited or grid.values[r][c] != 3:
                return
            visited.add((r, c))
            frame[0] = min(frame[0], r)
            frame[1] = min(frame[1], c)
            frame[2] = max(frame[2], r)
            frame[3] = max(frame[3], c)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    dfs(nr, nc, frame)

        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 3 and (r, c) not in visited:
                    frame = [r, c, r, c]
                    dfs(r, c, frame)
                    frames.append(tuple(frame))

        return frames

    def is_inside_frame(r: int, c: int, frame: Tuple[int, int, int, int]) -> bool:
        return frame[0] < r < frame[2] and frame[1] < c < frame[3]

    def fill_section(grid: ColoredGrid, frame: Tuple[int, int, int, int], color: int):
        for r in range(frame[0] + 1, frame[2]):
            for c in range(frame[1] + 1, frame[3]):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color

    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    rows, cols = len(output_grid.values), len(output_grid.values[0])

    # Find all frames
    frames = find_frames(input_grid)
    if not frames:
        frames = [(0, 0, rows - 1, cols - 1)]  # Treat entire grid as one frame if no green lines

    # Sort frames from outermost to innermost
    frames.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)

    # Fill outermost corners
    outermost_frame = frames[0]
    corners = [
        (outermost_frame[0], outermost_frame[1], 2),  # Top-left: red
        (outermost_frame[0], outermost_frame[3], 4),  # Top-right: yellow
        (outermost_frame[2], outermost_frame[1], 1),  # Bottom-left: blue
        (outermost_frame[2], outermost_frame[3], 8)   # Bottom-right: sky blue
    ]
    for r, c, color in corners:
        if output_grid.values[r][c] == 0:
            output_grid.values[r][c] = color

    # Process frames
    for frame in frames:
        fill_section(output_grid, frame, 7)

    # Preserve original elements
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                output_grid.values[r][c] = input_grid.values[r][c]

    return output_grid
