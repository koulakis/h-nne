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
