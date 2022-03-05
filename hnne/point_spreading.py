import networkx as nx
from scipy.spatial import Voronoi
from fa2 import ForceAtlas2
import numpy as np

from hnne.cool_functions import cool_normalize


def norm_angle(data, theta, partition):
    rot = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    data, norm1_params = cool_normalize(data, partition)

    rotated_data = np.dot(data, rot)
    rotated_data, norm2_params = cool_normalize(rotated_data, partition)

    return np.dot(rotated_data, np.linalg.inv(rot)), [rot, norm1_params, norm2_params]


def norm_angles(data, angles, partition_mapping):
    inflation_params = []
    for angle in angles:
        data, params = norm_angle(data, angle, partition_mapping)
        inflation_params.append(params)
    return data, inflation_params


def norm_angle_3d(data, alpha, beta, gamma, partition):
    r_x = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    r_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    r_z = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

    rot = np.dot(r_x, np.dot(r_y, r_z))

    data, norm1_params = cool_normalize(data, partition)

    rotated_data = np.dot(data, rot)
    rotated_data, norm2_params = cool_normalize(rotated_data, partition)

    return np.dot(rotated_data, np.linalg.inv(rot)), [rot, norm1_params, norm2_params]


def norm_angles_3d(data, alphas, betas, gammas, partition_mapping):
    inflation_params = []
    for alpha, beta, gamma in zip(alphas, betas, gammas):
        data, params = norm_angle_3d(data, alpha, beta, gamma, partition_mapping)
        inflation_params.append(params)
    return data, inflation_params


# Force atlas decompression
def force_directed_graph_decompression(points, weights, verbose=False, iterations=50):
    voronoi = Voronoi(points)

    graph = nx.Graph()
    for i, point in enumerate(points):
        graph.add_node(i, pos=point)
    for i, j in voronoi.ridge_points:
        graph.add_edge(i, j, weight=1 / (weights[i] * weights[j]))

    initial_positions = nx.get_node_attributes(graph, 'pos')
    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=verbose)
    positions = forceatlas2.forceatlas2_networkx_layout(graph, pos=initial_positions, iterations=iterations)

    max_radius = np.linalg.norm(np.array(list(positions.values())), axis=1).max()
    normalized_points = [(i, tuple(np.array(point) / max_radius)) for i, point in positions.items()]
    moved_points = np.array(list(map(lambda x: x[1], sorted(normalized_points, key=lambda x: x[0]))))

    return moved_points


def atlas_decompression(points):
    weights = np.ones(len(points))
    return force_directed_graph_decompression(points, weights=weights)
