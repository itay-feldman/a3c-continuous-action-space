from .utils import *
import lib.utils
import sys
import os
import glob
import random
import numpy as np
import cv2
import math

try:
    carla_egg = get_carla_egg()
    sys.path.append(carla_egg)
except (IndexError, TypeError) as e:
    raise e("Please specify path to Carla .egg file in config/config.ini")

import carla

IMG_SIZE_X = 1280  # width
IMG_SIZE_Y = 720   # height
IMG_FOV = 110



class CarlaEnv:
    """
        Environment mangement class for the CARLA simulator
    """
    # TODO add simulator settings change, might be better to put it in utils.py
    # TODO add car NPCs and pedestrians (add later, after seeing progress in training)
    def __init__(self, host=("localhost", 2000), threads=0, client_timeout=20, vehicle="vehicle.bmw.isetta", max_steps=0, img_type='RGB', img_x=1280, img_y=720, img_fov=110, calc_rotation=True, traffic_light=True, calc_lane_invasion=True, calc_speed_limit=True, calc_distace=True, calc_lane_type=True, calc_velocity=True, **kwargs):
        super().__init__()
        # general setup:
        self.client = carla.Client(host[0], host[1], threads)
        self.client.set_timeout(client_timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.transform = carla.Transform(carla.Location(x=1.5, z=0.7))

        # simulation setting
        self.settings = self.world.get_settings()
        self.dict_settings = kwargs

        # Env params:
        self.img_x = img_x
        self.img_y = img_y
        self.img_fov = img_fov
        self.traffic_light = traffic_light
        self.calc_lane_invasion = calc_lane_invasion
        self.calc_speed_limit = calc_speed_limit
        self.calc_distance = calc_distace
        self.calc_velocity = calc_velocity
        self.calc_lane_type = calc_lane_type
        self.calc_rotation = calc_rotation
        assert isinstance(img_type, str)
        self.img_type = img_type.upper()

        self.max_steps = max_steps
        self.vehicle_name = vehicle

        # setup values
        self.rgb, self.semantic = None, None
        self.collided, self.done = False, False
        self.actor_list, self.lanes_invaded = [], []

        # env
        self.current_state = self.reset()
        self.observation_space = np.zeros(self.current_state.shape)
        self.blocked = [self.last_location, 0]
        
    def reset(self):
        """
            Reset the environment
        """
        for i in range(5):
            try:
                self._clean_actors()
                # reset values
                self.rgb, self.semantic = None, None
                self.collided, self.done = False, False
                self.time_step = 0

                self.rewards = []
                self.actor_list = []
                
                # vehicle
                self.vehicle = self._get_vehicle(self.vehicle_name)
                self.actor_list.append(self.vehicle)
                
                # rgb camera
                self.rgb_cam = self._get_camera('rgb')
                self.actor_list.append(self.rgb_cam)
                self.rgb_cam.listen(lambda data: self._process_img_rgb(data))
                
                # semantic segmentation camera
                self.semantic_cam = self._get_camera('semantic_segmentation')
                self.actor_list.append(self.semantic_cam)
                self.cc = carla.ColorConverter.CityScapesPalette
                self.semantic_cam.listen(lambda data: self._process_img_semantic(data))
                
                # collision sensor
                col_sensor_bp = self.blueprint_lib.find('sensor.other.collision')
                self.col_sensor = self.world.spawn_actor(col_sensor_bp, self.transform, attach_to=self.vehicle)
                self.actor_list.append(self.col_sensor)
                self.col_sensor.listen(lambda event: self._on_collision(event))

                # Lane invasion sensor
                lane_inv_bp = self.blueprint_lib.find('sensor.other.lane_invasion')
                self.lane_inv_sensor = self.world.spawn_actor(lane_inv_bp, self.transform, attach_to=self.vehicle)
                self.actor_list.append(self.lane_inv_sensor)
                self.lane_inv_sensor.listen(lambda event: self._on_lane_invasion(event))
                break
            except RuntimeError as e:
                # Trying to reset for 5 times, on 5th try, raising error
                if i == 4:
                    raise RuntimeError(e)
                print(f'Encountered RuntimeError, assuming temporary connection issues, trying again...({i+2}/5 attempts)')

        
        while self.rgb is None or self.semantic is None:
            pass
        
        ret = None

        return np.rollaxis(np.array(self.rgb), -1, 0)  # first observation

    def step(self, action):
        """
            Takes in an action and does it in the environment
            Returns next state, reward, and if the env is done
        """
        if action[1] > 0:
            throttle = action[1]
            brake = 0.0
            reverse = False
        elif action[1] < 0:
            throttle = -action[1]
            brake = 0.0
            reverse = True
        else:
            throttle = 0.0
            reverse = False
            brake = 1.0

        for i in range(5):
            try:
                self.vehicle.apply_control(carla.VehicleControl(steer=action[0], throttle=throttle, brake=brake, reverse=reverse))
                reward = self._calculate_reward()
                break
            except RuntimeError as e:
                # Trying to reset for 5 times, on 5th try, raising error
                if i == 4:
                    # assuming major error, raising error and stopping process
                    raise RuntimeError(e)
                # print(f'Encountered RuntimeError, assuming temporary connection issues, trying again...({i+2}/5 attempts)')

        if self.collided:
            self.done = True
            # TODO should the collision reward be subtracted from the total reward, or rather should it replace the reward entirely?
            reward = self._reward_on_collision(self.collided)  

        if self.time_step != 0 and self.time_step == self.max_steps:
            self.done = True
        
        # trying to prevent staying in one place
        current_location = self.vehicle.get_location()
        if self.blocked[0] == current_location:
            self.blocked[1] += 1
        else:
            self.blocked[0] = current_location
            self.blocked[1] = 0
        
        if self.blocked[1] >= 15:  # x/10 sec
            reward = -1
            self.done = True

        self.rewards.append(reward)
        self.time_step += 1

        return np.rollaxis(np.array(self.rgb), -1, 0), reward, self.done, None  # we return None to be compatible with a gym environment


    def _calculate_reward(self):
        """
            Calculate reward for step (assuming no collision)
        """

        reward = 0.0  # base reward

        if self.calc_lane_invasion:
            waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
            change_allowed = waypoint.lane_change
            left_lane = waypoint.left_lane_marking
            right_lane = waypoint.right_lane_marking

            if change_allowed is not carla.LaneChange.Both:
                if (change_allowed is not carla.LaneChange.Right and right_lane in self.lanes_invaded) or (change_allowed is not carla.LaneChange.Left and left_lane in self.lanes_invaded):
                    reward += -1
            
            if self._lane_type_invasion(self.lanes_invaded):
                # crossing a lane marker of a lane that shouldn't be crossed
                reward += -1
        
        if self.calc_speed_limit:
            velocity = self.vehicle.get_velocity()

            if self.vehicle.get_speed_limit() < math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2):
                # over speeding
                reward += -1

        if self.traffic_light:
            if self.vehicle.is_at_traffic_light() and (self.vehicle.get_control().brake < 0.2):
                # negative reward for not decelarating in a red light
                # perhaps a part should be added to check if the agent crossed the light
                # while it was red and give a more substantial negative reward
                reward += -1

        # TODO maybe add a positive reward for speed, some coefficient time speed (velocity below x)

        if self.calc_distance:
            # reward for distance traveled
            current_location = self.vehicle.get_location()
            distance = self.last_location.distance(current_location)
            reward += distance  # adding a coefficient to the distance to increase importance of distance (make the car prefer moving)
            # if distance <= 0.3 and not self.vehicle.is_at_traffic_light():
                ## small negative reward for not moving, it's large enough to make the car move,
                ## but small enough to have it prioratize not breaking the law
                # reward += -1
            self.last_location = current_location

        if self.calc_lane_type:
            # reward for lane type
            current_w = self.map.get_waypoint(self.vehicle.get_location())
            if current_w.lane_type != carla.LaneType.Driving and current_w != carla.LaneType.Sidewalk:
                reward += -1
            elif current_w.lane_type == carla.LaneType.Sidewalk and not current_w.is_junction:
                reward += -1
        
        if self.calc_velocity:
            # reward for velocity
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # speed in km/h
            if speed >= 50:
                reward += 1
                # reward += math.log(speed, 4) if speed <= 65 else 1
        
        if self.calc_rotation:
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            lane_yaw = self.map.get_waypoint(self.vehicle.get_location()).transform.rotation.yaw
            yaw_diff = np.abs(vehicle_yaw - lane_yaw)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff  # 180 is the max
            reward += -round(yaw_diff / 100, 3)
            

        """# expirament: trying a very simple reward scheme based on speed:
        velocity_vector = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)  # speed in km/h
        if speed >= 55:
            reward += 1.5
        elif speed >= 25:
            reward += 1.25
        elif speed >= 10:
            reward += 1"""
        
        return reward

    def _reward_on_collision(self, event):
        """
            Return reward for collision
        """
        reward = -3

        return reward

    def _get_vehicle(self, vehicle):
        """
            Create a vehicle and return it
        """
        vehicle_bp = self.blueprint_lib.find(vehicle)
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.last_location = spawn_point.location
        vehicle = None
        for _ in range(10):
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)  # this will return None if spawning failed
            if vehicle:
                break
        if not vehicle:
            raise("Spawning failed, perhaps there are too many actors already in world")
        
        return vehicle
    
    def _get_camera(self, mode):
        """
            Create camera sensor attached to vehicle and return it
        """
        cam_bp = self.blueprint_lib.find(f"sensor.camera.{mode}")
        cam_bp.set_attribute("image_size_x", f"{self.img_x}")
        cam_bp.set_attribute("image_size_y", f"{self.img_y}")
        cam_bp.set_attribute("fov", f"{self.img_fov}")
        cam = self.world.spawn_actor(cam_bp, self.transform, attach_to=self.vehicle)  # spawing isn't expected to fail
        
        return cam

    def _process_img_rgb(self, sensor_data):
        """
            This function is called when there is new output
            from the rgb camera sensor
        """
        img = np.array(sensor_data.raw_data).reshape((self.img_y, self.img_x, 4))
        img = img[:, :, :3]  # sensor is actualy rgba, we dont need alpha values
        self.rgb = img # need to scale rgb values to be {0,1}

    def _process_img_semantic(self, sensor_data):
        """
            This function is called when there is new output
            from the semantice segmentation camera sensor
        """
        sensor_data.convert(self.cc)
        img = np.array(sensor_data.raw_data).reshape((self.img_y, self.img_x, 4))
        img = img[:, :, :3]  # sensor is actualy rgba, we dont need alpha values
        self.semantic = img # need to scale rgb values to be {0,1}

    def _on_collision(self, event):
        """
            This function is called when there is a collision
        """
        self.collided = event

    def _clean_actors(self):
        """
            Clean all actors this environment (client) is handling
        """
        for actor in self.actor_list:
            actor.destroy()

    def _on_lane_invasion(self, event):
        """
            This function is called when the lane invasion sensor
            has new output
        """
        self.lanes_invaded = event.crossed_lane_markings

    def _lane_type_invasion(self, lanes):
        """
            Check which type of lane was invaded
        """
        for lane in lanes:
            if lane.type is carla.LaneMarkingType.Solid or lane.type is carla.LaneMarkingType.SolidSolid:
                return True
        return False

    def close(self):
        self._clean_actors()

    def get_rewards(self):
        """
            Returns numpy array of rewards collected in the current episode (episode may not be done)
        """
        return np.array(self.rewards)

    def _apply_settings(self):
        """
            Applies settings to the simulation, settings that aren't supported are simply ignored
            To add a supported setting, simply add a condition in this function
        """
        if 'fixed_delta_seconds' in self.dict_settings:
            self.settings.fixed_delta_seconds = self.dict_settings['fixed_delta_seconds']

        self.world.apply_settings(self.settings)

    def _get_returned_state(self):
        """
            Chooses which state to return based on user
            configuration when env was created
        """
        if self.img_type == 'RGB':
            return np.rollaxis(np.array(self.rgb), -1, 0)
        elif self.img_type == 'SEMANTIC':
            return np.rollaxis(np.array(self.semantic), -1, 0)
        else:
            raise Exception(f'There is no {self.img_type} observation type')
        

