import copy
import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.objects import Landmark
from highway_env.road.objects import RoadObject

"""polite behavior"""
# """Longitudinal policy parameters"""
# Maximum acceleration.
ACC_MAX = 6.0  # [m/s2]
# Desired maximum acceleration.
COMFORT_ACC_MAX = 3.0  # [m/s2]
# Desired maximum deceleration.
COMFORT_ACC_MIN = -5.0  # [m/s2]
# Desired jam distance to the front vehicle.
DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
# Desired time gap to the front vehicle.
TIME_WANTED = 1.5  # [s]
# Exponent of the velocity term.
DELTA = 4.0  # []

"""Lateral policy parameters"""
POLITENESS = 0.  # in [0, 1]
LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
LANE_CHANGE_MAX_BRAKING_IMPOSED = 9.0  # [m/s2]
LANE_CHANGE_DELAY = 1.0  # [s]
MAX_STEERING_ANGLE = np.pi / 3

"""Control parameters"""
TAU_A = 0.6  # [s]
TAU_DS = 0.2  # [s]
PURSUIT_TAU = 0.5 * TAU_DS  # [s]
MAX_SPEED = 40  # Maximum reachable speed [m/s]
LENGTH = 5.0  # Vehicle length [m]
KP_A = 1 / TAU_A
KP_HEADING = 1 / TAU_DS
KP_LATERAL = 1 / 3 * KP_HEADING  # [1/s]


def idm_controller(vehicle, env_copy, action):
    if vehicle.crashed:
        vehicle.trajectories.append([vehicle.position, vehicle.heading, vehicle.speed])
        return
    dt = 1 / env_copy.config["simulation_frequency"]

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


def generate_actions(vehicle, env_copy):
    action = {}
    front_vehicle, rear_vehicle = neighbour_vehicles(vehicle, env_copy)

    # Lateral: MOBIL
    follow_road(vehicle, env_copy)
    target_lane_index = change_lane_policy(vehicle, env_copy)
    action['steering'] = steering_control(target_lane_index, vehicle, env_copy)
    action['steering'] = np.clip(action['steering'] * (np.random.rand() * 0.1 + 0.95), -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    # Longitudinal: IDM
    action['acceleration'] = acceleration(ego_vehicle=vehicle,
                                          front_vehicle=front_vehicle,
                                          rear_vehicle=rear_vehicle)
    # action['acceleration'] = self.recover_from_stop(action['acceleration'])
    action['acceleration'] = np.clip(action['acceleration'] * (np.random.rand() * 0.1 + 0.95), - ACC_MAX, ACC_MAX)
    vehicle.action = action
    return action

def follow_road(vehicle, env_copy):
    """At the end of a lane, automatically switch to a next one."""
    if env_copy.road.network.get_lane(vehicle.target_lane_index).after_end(vehicle.position):
        vehicle.target_lane_index = env_copy.road.network.next_lane(vehicle.target_lane_index,
                                                                    route=vehicle.route,
                                                                    position=vehicle.position,
                                                                    np_random=env_copy.road.np_random)


def change_lane_policy(vehicle, env_copy):
    """
    Decide when to change lane.

    Based on:
    - frequency;
    - closeness of the target lane;
    - MOBIL model.
    """
    # If a lane change already ongoing
    target_lane_index = vehicle.target_lane_index

    if vehicle.lane_index != target_lane_index:
        # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
        if vehicle.lane_index[:2] == target_lane_index[:2]:
            for v in env_copy.road.vehicles:
                if v is v.lane_index != target_lane_index \
                        and isinstance(v, ControlledVehicle) \
                        and v.target_lane_index == target_lane_index:
                    d = lane_distance_to(v, vehicle, env_copy)
                    d_star = desired_gap(v, vehicle, env_copy)
                    if 0 < d < d_star:
                        target_lane_index = vehicle.lane_index
                        break
        return vehicle.lane_index

    # decide to make a lane change
    for lane_index in env_copy.road.network.side_lanes(vehicle.lane_index):
        # Is the candidate lane close enough?
        if not env_copy.road.network.get_lane(lane_index).is_reachable_from(vehicle.position):
            continue
        # Does the MOBIL model recommend a lane change?
        if mobil(vehicle, env_copy):
            target_lane_index = lane_index

    return target_lane_index


def mobil(vehicle, env_copy):
    """
    MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

        The vehicle should change lane only if:
        - after changing it (and/or following vehicles) can accelerate more;
        - it doesn't impose an unsafe braking on its new following vehicle.

    :param lane_index: the candidate lane for the change
    :return: whether the lane change should be performed
    """
    # Is the maneuver unsafe for the new following vehicle?
    new_preceding, new_following = neighbour_vehicles(vehicle, env_copy)
    new_following_a = acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
    new_following_pred_a = acceleration(ego_vehicle=new_following, front_vehicle=vehicle)
    if new_following_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED:
        return False

    # Do I have a planned route for a specific lane which is safe for me to access?
    old_preceding, old_following = neighbour_vehicles(vehicle, env_copy)
    self_pred_a = acceleration(ego_vehicle=vehicle, front_vehicle=new_preceding)
    if vehicle.route and vehicle.route[0][2]:
        # Wrong direction
        if np.sign(vehicle.lane_index[2] - vehicle.target_lane_index[2]) != np.sign(
                vehicle.route[0][2] - vehicle.target_lane_index[2]):
            return False
        # Unsafe braking required
        elif self_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

    # Is there an acceleration advantage for me and/or my followers to change lane?
    else:
        self_a = acceleration(ego_vehicle=vehicle, front_vehicle=old_preceding)
        old_following_a = acceleration(ego_vehicle=old_following, front_vehicle=vehicle)
        old_following_pred_a = acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
        jerk = self_pred_a - self_a + POLITENESS * (new_following_pred_a - new_following_a
                                                    + old_following_pred_a - old_following_a)
        if jerk < LANE_CHANGE_MIN_ACC_GAIN:
            return False

    # All clear, let's go!
    return True


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


def acceleration(ego_vehicle,
                 front_vehicle,
                 rear_vehicle=None) -> float:
    """
    Compute an acceleration command with the Intelligent Driver Model.

    The acceleration is chosen so as to:
    - reach a target speed;
    - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

    :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                        IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                        reason about other vehicles behaviors even though they may not IDMs.
    :param front_vehicle: the vehicle preceding the ego-vehicle
    :param rear_vehicle: the vehicle following the ego-vehicle
    :return: the acceleration command for the ego-vehicle [m/s2]
    """
    if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
        return 0
    ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", 0))
    acceleration = COMFORT_ACC_MAX * (
            1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, DELTA))

    if front_vehicle:
        d = ego_vehicle.lane_distance_to(front_vehicle)
        acceleration -= COMFORT_ACC_MAX * \
                        np.power(desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
    return acceleration


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


def neighbour_vehicles(vehicle, env_copy):
    """
    Find the preceding and following vehicles of a given vehicle.

    :param vehicle: the vehicle whose neighbours must be found
    :param lane_index: the lane on which to look for preceding and following vehicles.
                 It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                 vehicle is projected on it considering its local coordinates in the lane.
    :return: its preceding vehicle, its following vehicle
    """
    lane_index = vehicle.lane_index or vehicle.lane_index
    if not lane_index:
        return None, None
    lane = env_copy.road.network.get_lane(lane_index)
    s = env_copy.road.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
    s_front = s_rear = None
    v_front = v_rear = None
    for v in env_copy.road.vehicles + env_copy.road.objects:
        if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
            # lane_index, same_lane=True):
            s_v, lat_v = lane.local_coordinates(v.position)
            if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                continue
            if s <= s_v and (s_front is None or s_v <= s_front):
                s_front = s_v
                v_front = v
            if s_v < s and (s_rear is None or s_v > s_rear):
                s_rear = s_v
                v_rear = v
    return v_front, v_rear


def desired_gap(ego_vehicle, front_vehicle, projected=True):
    """
    Compute the desired distance between a vehicle and its leading vehicle.

    :param ego_vehicle: the vehicle being controlled
    :param front_vehicle: its leading vehicle
    :param projected: project 2D velocities in 1D space
    :return: the desired distance between the two [m]
    """
    d0 = DISTANCE_WANTED
    tau = TIME_WANTED
    ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
    dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
        else ego_vehicle.speed - front_vehicle.speed
    d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
    return d_star


def lane_distance_to(v, vehicle, env_copy, lane=None):
    """
    Compute the signed distance to another vehicle along a lane.

    :param v: the other vehicle
    :param vehicle: the autonomous vehicle
    :param lane: a lane
    :return: the distance to the other vehicle [m]
    """
    if not v:
        return np.nan
    if not lane:
        lane = env_copy.road.network.get_lane(vehicle.lane_index)
    return lane.local_coordinates(v.position)[0] - lane.local_coordinates(vehicle.position)[0]