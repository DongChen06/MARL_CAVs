import copy
import numpy as np
from highway_env import utils


"""Control parameters"""
TAU_A = 0.6  # [s]
TAU_DS = 0.2  # [s]
PURSUIT_TAU = 0.5 * TAU_DS  # [s]
KP_A = 1 / TAU_A
KP_HEADING = 1 / TAU_DS
KP_LATERAL = 1 / 3 * KP_HEADING  # [1/s]
MAX_STEERING_ANGLE = np.pi / 3  # [rad]
DELTA_SPEED = 5  # [m/s]
LENGTH = 5.0  # Vehicle length [m]
MAX_SPEED = 40  # Maximum reachable speed [m/s]


def mdp_controller(vehicle, env_copy, action):
    """
            Perform a high-level action to change the desired lane or speed.

            - If a high-level action is provided, update the target speed and lane;
            - then, perform longitudinal and lateral control.

            :param action: a high-level action
            """
    dt = 1 / env_copy.config["simulation_frequency"]

    follow_road(vehicle, env_copy)
    # "FASTER"
    if action == 3:
        vehicle.target_speed += DELTA_SPEED
    # "SLOWER"
    elif action == 4:
        vehicle.target_speed -= DELTA_SPEED
    # "LANE_RIGHT"
    elif action == 2:
        _from, _to, _id = vehicle.target_lane_index  # ('a', 'b', 0)
        target_lane_index = _from, _to, np.clip(_id + 1, 0, len(env_copy.road.network.graph[_from][_to]) - 1)
        if env_copy.road.network.get_lane(target_lane_index).is_reachable_from(vehicle.position):
            vehicle.target_lane_index = target_lane_index
    # "LANE_LEFT"
    elif action == 0:
        _from, _to, _id = vehicle.target_lane_index
        target_lane_index = _from, _to, np.clip(_id - 1, 0, len(env_copy.road.network.graph[_from][_to]) - 1)
        if env_copy.road.network.get_lane(target_lane_index).is_reachable_from(vehicle.position):
            vehicle.target_lane_index = target_lane_index

    action = {"steering": steering_control(vehicle.target_lane_index, vehicle, env_copy),
              "acceleration": speed_control(vehicle, vehicle.target_speed)}
    action['steering'] = np.clip(action['steering'], -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    vehicle.action = action
    # step the vehicle
    clip_actions(action, vehicle.speed, vehicle.crashed)
    delta_f = action['steering']
    beta = np.arctan(1 / 2 * np.tan(delta_f))
    v = vehicle.speed * np.array([np.cos(vehicle.heading + beta),
                                  np.sin(vehicle.heading + beta)])
    vehicle.position += v * dt
    vehicle.heading += vehicle.speed * np.sin(beta) / (LENGTH / 2) * dt
    vehicle.speed += action['acceleration'] * dt
    vehicle.trajectories.append([copy.deepcopy(vehicle.position), vehicle.heading, vehicle.speed])


def follow_road(vehicle, env_copy):
    """At the end of a lane, automatically switch to a next one."""
    if env_copy.road.network.get_lane(vehicle.target_lane_index).after_end(vehicle.position):
        vehicle.target_lane_index = env_copy.road.network.next_lane(vehicle.target_lane_index,
                                                                    route=vehicle.route,
                                                                    position=vehicle.position,
                                                                    np_random=env_copy.road.np_random)


def steering_control(target_lane_index, vehicle, env_copy):
    """
    Steer the vehicle to follow the center of an given lane.

    1. Lateral position is controlled by a proportional controller yielding a lateral speed command
    2. Lateral speed command is converted to a heading reference
    3. Heading is controlled by a proportional controller yielding a heading rate command
    4. Heading rate command is converted to a steering angle

    :param target_lane_index: index of the lane to follow
    :return: a steering wheel angle command [rad]
    """
    target_lane = env_copy.road.network.get_lane(target_lane_index)
    lane_coords = target_lane.local_coordinates(vehicle.position)
    lane_next_coords = lane_coords[0] + vehicle.speed * PURSUIT_TAU
    lane_future_heading = target_lane.heading_at(lane_next_coords)

    # Lateral position control
    lateral_speed_command = - KP_LATERAL * lane_coords[1]
    # Lateral speed to heading
    heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(vehicle.speed), -1, 1))
    heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
    # Heading control
    heading_rate_command = KP_HEADING * utils.wrap_to_pi(heading_ref - vehicle.heading)
    # Heading rate to steering angle
    steering_angle = np.arcsin(np.clip(LENGTH / 2 / utils.not_zero(vehicle.speed) * heading_rate_command,
                                       -1, 1))
    steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    return float(steering_angle)


def speed_control(vehicle, target_speed):
    """
    Control the speed of the vehicle.
    Using a simple proportional controller.

    :param target_speed: the desired speed
    :return: an acceleration command [m/s2]
    """
    return KP_A * (target_speed - vehicle.speed)


def clip_actions(action, speed, crashed):
    if crashed:
        action['steering'] = 0
        action['acceleration'] = -1.0 * speed
    action['steering'] = float(action['steering'])
    action['acceleration'] = float(action['acceleration'])
    if speed > MAX_SPEED:
        action['acceleration'] = min(action['acceleration'], 1.0 * (MAX_SPEED - speed))
    elif speed < - MAX_SPEED:
        action['acceleration'] = max(action['acceleration'], 1.0 * (MAX_SPEED - speed))