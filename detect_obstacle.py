import numpy as np
from shapely.geometry import Point, LineString, Polygon


# Simulate the behavior of range sensor in simulation
# the input is the robot state (position and orientation) 
# and polygon object of the obstacle using shapely
def detect_obstacle(px, py, theta, obstaclePoly, sensing_range, collision_range):
    # assert that the point should be outside polygon
    assert not obstaclePoly.contains(Point(px, py)), \
        'detect_obstacle ERROR. (px,py) should be outside obstaclePoly'
    # assert that the robot not touching the polygon
    assert obstaclePoly.exterior.distance(Point(px, py)) >= collision_range, \
        'detect_obstacle ERROR. distance between obstaclePoly and (px,py) is less than collision_range: ' + str(
            collision_range)

    N = 8  # number of sensors divided in equal in-between angle
    # Create beam lines on 8 directions
    sensors_theta = [theta + i * 2 * np.pi / N for i in range(N)]
    endpoint_x = [px + sensing_range * np.cos(th) for th in sensors_theta]
    endpoint_y = [py + sensing_range * np.sin(th) for th in sensors_theta]
    lines = [LineString([(px, py), (endpoint_x[i], endpoint_y[i])]) for i in range(N)]

    distance = [sensing_range] * N
    for i in range(N):
        sensor_line = lines[i].difference(obstaclePoly)
        # Filter if difference return multiLines (e.g., when obstacle is too small)
        if sensor_line.type == 'MultiLineString': sensor_line = sensor_line[0]
        distance[i] = sensor_line.length

    return distance


# FUNCTIONS TO BE USED DURING THE EXERCISES
# -----------------------------------------------------------------------------


def detect_obstacle_e5t2(px, py, theta):
    sensing_range = 1.  # in meter
    collision_range = 0.21  # in meter
    obst_vertices = np.array([[-1., -1.5], [1., -1.5], [1., 1.5], [-1., 1.5], [-1., 1.], [0.5, 1.], [0.5, -1.], [-1., -1.],
                              [-1., -1.5]])  # ordered position of vertices in [x, y]

    # Create Polygon from list of vertices
    obstaclePoly = Polygon([(obst_vertices[i, 0], obst_vertices[i, 1]) for i in range(obst_vertices.shape[0])])
    return detect_obstacle(px, py, theta, obstaclePoly, sensing_range, collision_range)
