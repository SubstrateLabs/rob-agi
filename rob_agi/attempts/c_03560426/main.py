from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_03560426(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging colored shapes into a snake-like pattern.
    
    1. Extracts and sorts shapes by area (largest to smallest).
    2. Places shapes in a new grid, starting from the top-left corner.
    3. Connects shapes edge-to-edge, forming a continuous line.
    4. Moves to the next row when reaching the right edge.
    5. Fills remaining space with black (0).
    
    Returns the transformed grid.
    """
    shapes = extract_shapes(input_grid)
    sorted_shapes = sort_shapes_by_area(shapes)
    output_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    current_position = (0, 0)
    
    for shape in sorted_shapes:
        orientations = get_shape_orientations(shape)
        for orientation in orientations:
            if can_place_shape(output_grid, orientation, current_position):
                place_shape(output_grid, orientation, current_position)
                current_position = get_next_position(output_grid, current_position)
                break
    
    return output_grid

def extract_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    for color in range(1, 10):  # Exclude black (0)
        regions = grid.find_connected_regions(color)
        shapes.extend(regions)
    return shapes

def sort_shapes_by_area(shapes: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    return sorted(shapes, key=lambda shape: len(shape), reverse=True)

def get_shape_orientations(shape: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    # Original orientation
    orientations = [shape]
    
    # Rotated 90 degrees
    rotated = [(y, -x) for x, y in shape]
    orientations.append(rotated)
    
    # Flipped horizontally
    flipped = [(-x, y) for x, y in shape]
    orientations.append(flipped)
    
    return orientations

def can_place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int]) -> bool:
    rows, cols = grid.get_dimensions()
    for x, y in shape:
        new_x, new_y = position[0] + x, position[1] + y
        if new_x < 0 or new_x >= rows or new_y < 0 or new_y >= cols or grid.values[new_x][new_y] != 0:
            return False
    return True

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int]) -> None:
    color = grid.values[shape[0][0]][shape[0][1]]
    for x, y in shape:
        new_x, new_y = position[0] + x, position[1] + y
        grid.values[new_x][new_y] = color

def get_next_position(grid: ColoredGrid, current_position: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    x, y = current_position
    
    # Try moving right
    if y + 1 < cols and grid.values[x][y + 1] == 0:
        return (x, y + 1)
    
    # Move to the next row
    for new_x in range(x + 1, rows):
        for new_y in range(cols):
            if grid.values[new_x][new_y] == 0:
                # Check if it's adjacent to a non-zero cell
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_x, adj_y = new_x + dx, new_y + dy
                    if 0 <= adj_x < rows and 0 <= adj_y < cols and grid.values[adj_x][adj_y] != 0:
                        return (new_x, new_y)
    
    # If no suitable position found, return the current position
    return current_position
