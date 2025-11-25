import cupy as cp
from hnne.cool_functions import cool_normalize


def norm_angle(data, theta, partition):
    rot = cp.array([[cp.cos(theta), cp.sin(theta)], [-cp.sin(theta), cp.cos(theta)]])

    data, norm1_params = cool_normalize(data, partition)

    rotated_data = cp.dot(data, rot)
    rotated_data, norm2_params = cool_normalize(rotated_data, partition)

    return cp.dot(rotated_data, rot.T), [rot, norm1_params, norm2_params]


def norm_angles(data, angles, partition_mapping):
    inflation_params = []
    for angle in angles:
        data, params = norm_angle(data, angle, partition_mapping)
        inflation_params.append(params)
    return data, inflation_params


def norm_angle_3d(data, alpha, beta, gamma, partition):
    r_x = cp.array(
        [
            [cp.cos(alpha), -cp.sin(alpha), 0],
            [cp.sin(alpha), cp.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    r_y = cp.array(
        [[cp.cos(beta), 0, cp.sin(beta)], [0, 1, 0], [-cp.sin(beta), 0, cp.cos(beta)]]
    )
    r_z = cp.array(
        [
            [1, 0, 0],
            [0, cp.cos(gamma), -cp.sin(gamma)],
            [0, cp.sin(gamma), cp.cos(gamma)],
        ]
    )

    rot = cp.dot(r_x, cp.dot(r_y, r_z))

    data, norm1_params = cool_normalize(data, partition)

    rotated_data = cp.dot(data, rot)
    rotated_data, norm2_params = cool_normalize(rotated_data, partition)

    return cp.dot(rotated_data, cp.linalg.inv(rot)), [rot, norm1_params, norm2_params]


def norm_angles_3d(data, alphas, betas, gammas, partition_mapping):
    inflation_params = []
    for alpha, beta, gamma in zip(alphas, betas, gammas):
        data, params = norm_angle_3d(data, alpha, beta, gamma, partition_mapping)
        inflation_params.append(params)
    return data, inflation_params
