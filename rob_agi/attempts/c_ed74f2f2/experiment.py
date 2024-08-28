from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ed74f2f2.main import solve_ed74f2f2
from typing import List, Tuple

def analyze_input_output_relationship(input_grid: ColoredGrid, output_grid: ColoredGrid) -> dict:
    input_features = {
        'total_gray': sum(cell == 5 for row in input_grid.values for cell in row),
        'gray_distribution': [sum(cell == 5 for cell in row) for row in input_grid.values],
        'corners': [input_grid.get_cell(i, j) == 5 for i, j in [(1, 1), (1, 7), (3, 1), (3, 7)]],
        'edges': [input_grid.get_cell(i, j) == 5 for i, j in [(1, 4), (2, 1), (2, 7), (3, 4)]],
        'center': input_grid.get_cell(2, 4) == 5,
    }
    
    output_features = {
        'color': max(max(row) for row in output_grid.values),
        'total_colored': sum(cell == output_features['color'] for row in output_grid.values for cell in row),
        'colored_distribution': [sum(cell == output_features['color'] for cell in row) for row in output_grid.values],
        'corners': [output_grid.get_cell(i, j) == output_features['color'] for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)]],
        'edges': [output_grid.get_cell(i, j) == output_features['color'] for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]],
        'center': output_grid.get_cell(1, 1) == output_features['color'],
    }
    
    return {
        'input': input_features,
        'output': output_features
    }

def run_experiment():
    test_cases = [
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 0, 0, 5, 5, 5, 0],
         [0, 0, 5, 0, 0, 5, 0, 5, 0],
         [0, 0, 5, 5, 0, 5, 0, 5, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 5, 5, 0, 5, 0, 5, 0],
         [0, 0, 5, 0, 0, 5, 0, 5, 0],
         [0, 5, 5, 0, 0, 5, 5, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 5, 0, 5, 0, 5, 0],
         [0, 0, 5, 0, 0, 0, 5, 5, 0],
         [0, 0, 5, 0, 0, 5, 0, 5, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 5, 0, 5, 0, 5, 0],
         [0, 0, 5, 0, 0, 5, 5, 5, 0],
         [0, 0, 5, 0, 0, 5, 5, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 0, 0, 5, 5, 0, 0],
         [0, 0, 5, 0, 0, 0, 5, 5, 0],
         [0, 0, 5, 5, 0, 0, 5, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 0, 0, 5, 0, 0, 0],
         [0, 0, 5, 0, 0, 0, 5, 5, 0],
         [0, 0, 5, 5, 0, 5, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        input_grid = ColoredGrid(values=test_case)
        output_grid = solve_ed74f2f2(input_grid)
        analysis = analyze_input_output_relationship(input_grid, output_grid)
        results.append(analysis)
        
        print(f"Test Case {i + 1}:")
        print(f"Input: {analysis['input']}")
        print(f"Output: {analysis['output']}")
        print("---")
    
    # Additional analysis
    print("\nObservations:")
    for i, result in enumerate(results):
        input_features = result['input']
        output_features = result['output']
        
        print(f"\nTest Case {i + 1}:")
        print(f"Gray cells: {input_features['total_gray']} -> Colored cells: {output_features['total_colored']}")
        print(f"Input center filled: {input_features['center']} -> Output center filled: {output_features['center']}")
        print(f"Input corners filled: {sum(input_features['corners'])} -> Output corners filled: {sum(output_features['corners'])}")
        print(f"Input edges filled: {sum(input_features['edges'])} -> Output edges filled: {sum(output_features['edges'])}")
        print(f"Input distribution: {input_features['gray_distribution']} -> Output distribution: {output_features['colored_distribution']}")

if __name__ == "__main__":
    run_experiment()
