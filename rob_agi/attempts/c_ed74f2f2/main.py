from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Analyzes the input grid to identify key features and patterns.
    2. Determines the output color based on the complexity and distribution of gray cells.
    3. Creates an initial shape based on the identified patterns.
    4. Refines the shape to reflect input characteristics and maintain balance.
    5. Applies pattern transformation to capture the essence of the input.
    6. Makes final adjustments to ensure a valid and interesting output.
    7. Returns the final 3x3 ColoredGrid output.
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
        'total_gray': sum(cell == 5 for row in input_grid.values for cell in row)
    }
    
    for i in range(3):
        for j in range(3):
            section = input_grid.extract_subgrid(i*2, j*3, 3, 3)
            features['density'][i][j] = sum(cell == 5 for row in section.values for cell in row)
    
    return features

def determine_color(features: dict) -> int:
    total_density = sum(sum(row) for row in features['density'])
    if features['total_gray'] <= 10:
        return 2  # Red
    elif features['total_gray'] <= 15:
        return 1  # Blue
    else:
        return 3  # Green

def create_initial_shape(features: dict, color: int) -> List[List[int]]:
    shape = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if features['density'][i][j] > 1:
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
    
    # Apply rotational symmetry
    if features['total_gray'] % 2 == 0:
        for i in range(3):
            for j in range(3):
                if shape[i][j] == color:
                    transformed[2-i][2-j] = color
    
    # Apply reflective symmetry
    else:
        for i in range(3):
            transformed[i][2] = shape[i][0]
    
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
            for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                if shape[i][j] == color:
                    shape[i][j] = 0
                    break
    
    return shape
