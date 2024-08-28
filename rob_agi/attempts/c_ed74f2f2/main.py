from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Analyzes the input grid to identify key features and patterns.
    2. Determines the output color based on the total number of gray cells.
    3. Creates an initial shape based on the density of gray cells in 3x3 sections.
    4. Refines the shape to reflect input characteristics and maintain balance.
    5. Applies symmetry based on the total number of gray cells.
    6. Makes final adjustments to ensure a valid and interesting output.
    7. Preserves disconnected patterns and ensures balance between colored and black cells.
    8. Returns the final 3x3 ColoredGrid output.
    """
    features = analyze_input(input_grid)
    color = determine_color(features)
    initial_shape = create_initial_shape(features, color)
    refined_shape = refine_shape(initial_shape, features)
    transformed_shape = apply_transformation(refined_shape, features)
    final_shape = final_adjustments(transformed_shape, features)
    return ColoredGrid(values=final_shape)

def analyze_input(input_grid: ColoredGrid) -> dict:
    features = {
        'density': [[0 for _ in range(3)] for _ in range(3)],
        'corners': [input_grid.get_cell(i, j) == 5 for i, j in [(1, 1), (1, 7), (3, 1), (3, 7)]],
        'edges': [input_grid.get_cell(i, j) == 5 for i, j in [(1, 4), (2, 1), (2, 7), (3, 4)]],
        'center': input_grid.get_cell(2, 4) == 5,
        'total_gray': sum(cell == 5 for row in input_grid.values for cell in row),
        'disconnected': is_disconnected(input_grid)
    }
    
    for i in range(3):
        for j in range(3):
            section = input_grid.extract_subgrid(i*2, j*3, 3, 3)
            features['density'][i][j] = sum(cell == 5 for row in section.values for cell in row)
    
    return features

def is_disconnected(grid: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    visited = set()

    def dfs(r, c):
        if (r, c) in visited or grid.get_cell(r, c) != 5:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc)

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 5:
                dfs(r, c)
                return sum(grid.get_cell(r, c) == 5 for r in range(rows) for c in range(cols)) != len(visited)
    return False

def determine_color(features: dict) -> int:
    if features['total_gray'] <= 11:
        return 2  # Red
    elif features['total_gray'] <= 15:
        return 1  # Blue
    else:
        return 3  # Green

def create_initial_shape(features: dict, color: int) -> List[List[int]]:
    shape = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if features['density'][i][j] >= 2:
                shape[i][j] = color
    return shape

def refine_shape(shape: List[List[int]], features: dict) -> List[List[int]]:
    color = max(max(row) for row in shape)
    
    # Adjust corners
    for i, corner in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        if features['corners'][i]:
            shape[corner[0]][corner[1]] = color
    
    # Adjust edges
    for i, edge in enumerate([(0, 1), (1, 0), (1, 2), (2, 1)]):
        if features['edges'][i]:
            shape[edge[0]][edge[1]] = color
    
    # Adjust center
    if features['center']:
        shape[1][1] = color
    
    return shape

def apply_transformation(shape: List[List[int]], features: dict) -> List[List[int]]:
    color = max(max(row) for row in shape)
    transformed = [row[:] for row in shape]
    
    # Apply rotational symmetry for even total_gray
    if features['total_gray'] % 2 == 0:
        for i in range(3):
            for j in range(3):
                if shape[i][j] == color:
                    transformed[2-i][2-j] = color
    
    # Apply vertical reflection symmetry for odd total_gray
    else:
        for i in range(3):
            for j in range(3):
                transformed[i][2-j] = shape[i][j]
    
    return transformed

def final_adjustments(shape: List[List[int]], features: dict) -> List[List[int]]:
    color = max(max(row) for row in shape)
    colored_cells = sum(cell == color for row in shape for cell in row)
    
    if colored_cells == 0:
        shape[1][1] = color
    elif colored_cells == 9:
        shape[1][1] = 0
    elif colored_cells < 3:
        shape[0][0] = color
        shape[1][0] = color
        shape[0][1] = color
    elif colored_cells > 7:
        if shape[1][1] == color:
            shape[1][1] = 0
        else:
            for i, j in [(2, 1), (1, 2)]:  # Prefer bottom or right edge
                if shape[i][j] == color:
                    shape[i][j] = 0
                    break
    
    # Preserve disconnected patterns
    if features['disconnected'] and colored_cells > 5:
        disconnected = False
        for i in range(3):
            for j in range(3):
                if shape[i][j] == color:
                    neighbors = sum(shape[ni][nj] == color 
                                    for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                                    if 0 <= ni < 3 and 0 <= nj < 3)
                    if neighbors == 0:
                        disconnected = True
                        break
            if disconnected:
                break
        if not disconnected:
            shape[1][1] = 0  # Disconnect by removing center if connected
    
    # Ensure at least one black cell and one colored cell
    if colored_cells == 9:
        shape[2][2] = 0
    elif colored_cells == 0:
        shape[0][0] = color
    
    return shape
