from rob_agi.colored_grid import ColoredGrid

def experiment_path_generation(start, end):
    path = []
    current = start
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    while current != end:
        path.append(current)
        if abs(dx) > 0 and dy > 0:
            # Move diagonally
            current = (current[0] + (1 if dx > 0 else -1), current[1] + 1)
            dx += -1 if dx > 0 else 1
            dy -= 1
        elif dy > 0:
            # Move vertically
            current = (current[0], current[1] + 1)
            dy -= 1
        else:
            # Move horizontally
            current = (current[0] + (1 if dx > 0 else -1), current[1])
            dx += -1 if dx > 0 else 1
    
    path.append(end)
    return path

# Test case from the failing example
start = (2, 2)  # Adjacent to red square at (1, 1)
end = (8, 13)   # Adjacent to blue square at (8, 13)

path = experiment_path_generation(start, end)

print("Generated path:")
for point in path:
    print(point)

# Visualize the path
grid = [[0 for _ in range(12)] for _ in range(15)]
grid[1][1] = 2  # Red square
grid[13][8] = 1  # Blue square

for x, y in path:
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        grid[y][x] = 3  # Green path

print("\nVisualized grid:")
for row in grid:
    print(' '.join(str(cell) for cell in row))
