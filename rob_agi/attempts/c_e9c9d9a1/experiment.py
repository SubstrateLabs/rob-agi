from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e9c9d9a1.main import solve_e9c9d9a1, find_frames

def visualize_frames_and_corners(input_grid):
    frames = find_frames(input_grid)
    output_grid = input_grid.deep_copy()
    
    # Mark frames
    for frame in frames:
        for r in range(frame[0], frame[2] + 1):
            for c in range(frame[1], frame[3] + 1):
                if r in [frame[0], frame[2]] or c in [frame[1], frame[3]]:
                    output_grid.values[r][c] = 5  # Use gray (5) to mark frame boundaries
    
    # Mark corners of the outermost frame
    if frames:
        outermost = frames[0]
        output_grid.values[outermost[0]][outermost[1]] = 2  # Top-left: red
        output_grid.values[outermost[0]][outermost[3]] = 4  # Top-right: yellow
        output_grid.values[outermost[2]][outermost[1]] = 1  # Bottom-left: blue
        output_grid.values[outermost[2]][outermost[3]] = 8  # Bottom-right: sky blue
    
    return output_grid

# Test with example_0
input_grid = ColoredGrid(values=[
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0]
])

visualized_grid = visualize_frames_and_corners(input_grid)
print("Visualized Grid:")
for row in visualized_grid.values:
    print(" ".join(str(cell) for cell in row))

print("\nDetected Frames:")
frames = find_frames(input_grid)
for i, frame in enumerate(frames):
    print(f"Frame {i + 1}: {frame}")
