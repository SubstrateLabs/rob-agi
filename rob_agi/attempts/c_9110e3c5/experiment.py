from rob_agi.colored_grid import ColoredGrid
import yaml

def analyze_grid(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    second_to_last_col = [row[-2] for row in grid.values]
    edges = [
        grid.values[0],  # top edge
        [row[-1] for row in grid.values],  # right edge
        grid.values[-1],  # bottom edge
        [row[0] for row in grid.values]  # left edge
    ]
    
    edge_strengths = [sum(1 for cell in edge if cell != 0) / len(edge) for edge in edges]
    second_to_last_col_strength = sum(1 for cell in second_to_last_col if cell != 0) / len(second_to_last_col)
    
    return {
        "edge_strengths": edge_strengths,
        "second_to_last_col_strength": second_to_last_col_strength,
        "non_black_ratio": sum(sum(1 for cell in row if cell != 0) for row in grid.values) / (rows * cols)
    }

# Load visual descriptions
with open('rob_agi/attempts/c_9110e3c5/visual_descriptions.yaml', 'r') as file:
    descriptions = yaml.safe_load(file)

# Analyze each example
for example, data in descriptions.items():
    if 'input' in data:
        print(f"\nAnalyzing {example}:")
        input_desc = data['input'].strip().split('\n')[0]  # Get the first line of the input description
        print(f"Input: {input_desc}")
        
        # Create a mock grid based on the description
        mock_grid = ColoredGrid(values=[[1 if 'prominent' in input_desc or 'vertical line' in input_desc else 0 for _ in range(7)] for _ in range(7)])
        
        analysis = analyze_grid(mock_grid)
        print(f"Edge strengths: {analysis['edge_strengths']}")
        print(f"Second-to-last column strength: {analysis['second_to_last_col_strength']}")
        print(f"Non-black cell ratio: {analysis['non_black_ratio']}")
        
        if 'output' in data:
            output_desc = data['output'].strip().split('\n')[0]  # Get the first line of the output description
            print(f"Output: {output_desc}")

print("\nNote: This analysis uses mock grids based on the descriptions. Actual results may vary with real input data.")
