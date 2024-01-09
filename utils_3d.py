import open3d as o3d
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor

def is_present(point):
    if point[2] == -1:
        return False
    else:
        return True

def create_floor_wall(vis):
    half_length = 3
    grid_depth = 3
    floor_grid = create_grid([-half_length, -1, 0], [half_length, -1, 0], [half_length, -1, grid_depth],
                             [-half_length, -1, grid_depth], 2 * half_length, grid_depth)  # Floor
    wall_grid = create_grid([-half_length, 1, 0], [half_length, 1, 0], [half_length, -1, 0], [-half_length, -1, 0],
                            2 * half_length, 2)  # Wall
    vis.add_geometry(floor_grid)
    vis.add_geometry(wall_grid)

def determine_color(color_position):
    if color_position < 9:
        line_color = (1, 0, 0)
        return line_color
    if color_position < 14:
        line_color = (1, 1, 0)
        return line_color
    if color_position < 19:
        line_color = (0, 1, 0)
        return line_color

def create_dynamic_grid(vis, a, b, c):
    half_length = 3
    grid_depth = 3

    floor_grid = create_grid([-half_length, -1, -half_length * a - b + c],
                             [half_length, -1, half_length * a - b + c],
                             [half_length, -1, half_length * a - b + c + grid_depth],
                             [-half_length, -1, -half_length * a - b + c + grid_depth],
                             2 * half_length, grid_depth)  # Floor
    wall_grid = create_grid([-half_length, 1, -half_length * a + b + c],
                            [half_length, 1, half_length * a - b + c],
                            [half_length, -1, half_length * a - b + c],
                            [-half_length, -1, -half_length * a - b + c],
                            2 * half_length, 2)  # Wall

    vis.add_geometry(floor_grid)
    vis.add_geometry(wall_grid)

def coordinates_dynamic_grid(kpts_xyd):
    try:
        # object_coordinates = [[x / 100, -y / 100,  float(d) /pow(10, len(str(int(d))) - 1)] for x, y, d in kpts_xyvd]
        object_coordinates = [[x / 100, -y / 100, d / 10000] for x, y, d in kpts_xyd]
        return object_coordinates

    except IndexError as e:
        print(f"{e}")

def find_plane_points(kpts_xyvd):
    try:
        print(kpts_xyvd)
        rated_points1 = [kpts_xyvd[15][0] / 100, -kpts_xyvd[15][1] / 100, kpts_xyvd[15][2] / 10000]
        rated_points2 = [kpts_xyvd[16][0] / 100, -kpts_xyvd[16][1] / 100, kpts_xyvd[16][2] / 10000]

        foot_points = np.array([rated_points1, rated_points2])

        additional_points = np.random.uniform(-0.00003, 0.00003, size=(10, 3))

        plane_points = np.concatenate([foot_points, additional_points])

        return plane_points
    except IndexError as e:
        print(f"{e}")

def find_plane(plane_points):
    X = plane_points[:, :2]  # x和y坐标
    y = plane_points[:, 2]  # z坐标

    ransac = RANSACRegressor()

    ransac.fit(X, y)

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    print(f"Estimated plane: z = {a}x + {b}y + {c}")

    coefficient = [a, b, c]

    return coefficient

def coordinates_moving_operation(kpts_xyd):
    try:
        print(kpts_xyd)
        # object_coordinates = [[x / 100, -y / 100,  float(d) /pow(10, len(str(int(d))) - 1)] for x, y, d in kpts_xyvd]
        object_coordinates = [[x / 100, -y / 100, d / 10000] for x, y, d in kpts_xyd]
        reference_point = [0, -1, 1]
        difference = np.array(object_coordinates[16]) - np.array(reference_point)

        for i in range(len(object_coordinates)):
            object_coordinates[i] = np.array(object_coordinates[i]) - difference

        object_coordinates[16] = reference_point

        return object_coordinates
    except IndexError as e:
        print(f"{e}")
    except ValueError as e:
        print(f"ValueError: {e}")

def create_kpts_xyd(kpts_xyvd):
    kpts_xyd = []

    for item in kpts_xyvd:
        kpts_xyd.append([item[0], item[1], item[3]])

    return kpts_xyd

def create_cylinder(height=1, radius=None, resolution=20):
    """
    Create an cylinder in Open3D
    """
    radius = height / 20 if radius is None else radius
    mesh_frame = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=resolution)
    return (mesh_frame)

def calculate_zy_rotation_for_arrow(v):
    """
    Calculates the rotations required to go from the vector v to the
    z axis vector. The first rotation that is
    calculated is over the z axis. This will leave the vector v on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector v into the same orientation as axis z

    Args:
        - v ():
    """
    # Rotation over z axis
    gamma = np.arctan(v[1] / v[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # Rotate v to calculate next rotation
    v = Rz.T @ v.reshape(-1, 1)
    v = v.reshape(-1)
    # Rotation over y axis
    beta = np.arctan(v[0] / v[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    return Rz @ Ry

def create_segment(a, b, radius=0.05, color=(1, 1, 0), resolution=20):
    """
    Creates an line(cylinder) from an pointa to point b,
    or create an line from a vector v starting from origin.
    Args:
        - a, b: End points [x,y,z]
        - radius: radius cylinder
    """
    a = np.array(a)
    b = np.array(b)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = a
    v = b - a

    height = np.linalg.norm(v)
    if height == 0: return None
    R = calculate_zy_rotation_for_arrow(v)
    mesh = create_cylinder(height, radius)
    mesh.rotate(R, center=np.array([0, 0, 0]))
    mesh.translate((a + b) / 2)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def create_grid(p0, p1, p2, p3, ni1, ni2, color=(0, 0, 0)):
    '''
    p0, p1, p2, p3 : points defining a quadrilateral
    ni1: nb of equidistant intervals on segments p0p1 and p3p2
    ni2: nb of equidistant intervals on segments p1p2 and p0p3
    '''
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    vertices = [p0, p1, p2, p3]
    lines = [[0, 1], [0, 3], [1, 2], [2, 3]]
    for i in range(1, ni1):
        l = len(vertices)
        vertices.append((p0 * (ni1 - i) + p1 * i) / ni1)
        vertices.append((p3 * (ni1 - i) + p2 * i) / ni1)
        lines.append([l, l + 1])
    for i in range(1, ni2):
        l = len(vertices)
        vertices.append((p1 * (ni2 - i) + p2 * i) / ni2)
        vertices.append((p0 * (ni2 - i) + p3 * i) / ni2)
        lines.append([l, l + 1])
    vertices = o3d.utility.Vector3dVector(vertices)
    lines = o3d.utility.Vector2iVector(lines)
    mesh = o3d.geometry.LineSet(vertices, lines)
    mesh.paint_uniform_color(color)

    return mesh
