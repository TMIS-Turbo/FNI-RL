from __future__ import absolute_import
from __future__ import print_function

import gym
import numpy as np
import os
import sys
import math
import xml.dom.minidom


# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

gui = True
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')
config_path = os.path.dirname(__file__)+"/../../../Environment/environment/env5/Navigation.sumocfg"  # Navigation task


class Traffic_Env(gym.Env):

    def __init__(self):
        self.AutoCarID = 'Auto'
        self.maxDistance = 200.0
        self.maxSpeed = 30.0
        self.max_angle = 360.0
        self.x_goal = 2100.0
        self.y_goal = 150.0
        self.max_dis_navigation = 2500.0
        self.max_acc = 7.6
        self.reset_times = 0

    def raw_obs(self, vehicle_params):  # dimension: 24+5
        obs = []

        if self.AutoCarID in vehicle_params:
            zone = [[[],[],[]] for _ in range(6)]

            ego_veh_x, ego_veh_y = traci.vehicle.getPosition(self.AutoCarID)
            dis_goal_ego = np.linalg.norm(np.array([self.x_goal-ego_veh_x, self.y_goal-ego_veh_y]))

            obs_light = traci.vehicle.getNextTLS(self.AutoCarID)
            if obs_light is ():
                dis2light = self.maxDistance
                red_light = 0.0
            else:
                dis2light = min(obs_light[0][2], self.maxDistance)
                if obs_light[0][3] is 'r':
                    red_light = 1.0
                else:
                    red_light = 0.0

            for VehID in vehicle_params:
                veh_x, veh_y = traci.vehicle.getPosition(VehID)  # position, X & Y
                dis = np.linalg.norm(np.array([veh_x-ego_veh_x, veh_y-ego_veh_y]))

                if VehID != self.AutoCarID and dis < self.maxDistance:
                    angle = math.degrees(math.atan2(veh_y-ego_veh_y, veh_x-ego_veh_x))

                    if 0 <= angle < math.degrees(math.atan2(3**0.5, 1)): # 0~60
                        zone[0][0].append(VehID)
                        zone[0][1].append(dis)
                        zone[0][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, 1)) <= angle < math.degrees(math.atan2(3**0.5, -1)): # 60~120
                        zone[1][0].append(VehID)
                        zone[1][1].append(dis)
                        zone[1][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, -1)) <= angle < 180: # 120~180
                        zone[2][0].append(VehID)
                        zone[2][1].append(dis)
                        zone[2][2].append(angle)
                    elif -180 <= angle < math.degrees(math.atan2(-3**0.5, -1)): # -180~-120
                        zone[3][0].append(VehID)
                        zone[3][1].append(dis)
                        zone[3][2].append(angle)
                    elif math.degrees(math.atan2(-3**0.5, -1)) <= angle < math.degrees(math.atan2(-3**0.5, 1)): # -120~-60
                        zone[4][0].append(VehID)
                        zone[4][1].append(dis)
                        zone[4][2].append(angle)
                    else: # -60~0
                        zone[5][0].append(VehID)
                        zone[5][1].append(dis)
                        zone[5][2].append(angle)

            for z in zone:
                if len(z[0]) == 0:
                    obs.append(self.maxDistance)
                    obs.append(0.0)
                    obs.append(0.0)
                    obs.append(0.0)
                else:
                    mindis_index = z[1].index(min(z[1]))
                    obs.append(min(z[1]))
                    obs.append(z[2][mindis_index])
                    obs.append(traci.vehicle.getSpeed(z[0][mindis_index]))
                    obs.append(traci.vehicle.getAngle(z[0][mindis_index]))

            obs.append(traci.vehicle.getSpeed(self.AutoCarID))
            obs.append(traci.vehicle.getAngle(self.AutoCarID))
            obs.append(dis2light)
            obs.append(red_light)
            obs.append(dis_goal_ego)
            info = [ego_veh_x, ego_veh_y]

        else:
            obs = [self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,self.maxDistance, 0.0, 0.0, 0.0,\
                   self.maxDistance, 0.0, 0.0, 0.0,self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,\
                   0.0, 0.0, self.maxDistance, 0.0, self.max_dis_navigation]
            info = [0.0, 0.0]

        return obs, info

    def obs_to_state(self, vehicle_params):
        obs, info = self.raw_obs(vehicle_params)
        # print("raw_obs===>", obs)
        state = [obs[0]/self.maxDistance, obs[1]/self.max_angle, obs[2]/self.maxSpeed, obs[3]/self.max_angle,\
                 obs[4]/self.maxDistance, obs[5]/self.max_angle, obs[6]/self.maxSpeed, obs[7]/self.max_angle,\
                 obs[8]/self.maxDistance, obs[9]/self.max_angle, obs[10]/self.maxSpeed, obs[11]/self.max_angle,\
                 obs[12]/self.maxDistance, obs[13]/self.max_angle, obs[14]/self.maxSpeed, obs[15]/self.max_angle,\
                 obs[16]/self.maxDistance, obs[17]/self.max_angle, obs[18]/self.maxSpeed, obs[19]/self.max_angle,\
                 obs[20]/self.maxDistance, obs[21]/self.max_angle, obs[22]/self.maxSpeed, obs[23]/self.max_angle,\
                 obs[24]/self.maxSpeed, obs[25]/self.max_angle, obs[26]/self.maxDistance, obs[27], obs[28]/self.max_dis_navigation]

        return state, info

    def get_reward(self, vehicle_params):
        cost = 0.0
        infraction = 0.0
        infraction_check = False
        navigation_check = False
        done = False

        raw_obs, _ = self.raw_obs(vehicle_params)
        dis_fr = raw_obs[0]
        dis_f = raw_obs[4]
        dis_fl = raw_obs[8]
        dis_rl = raw_obs[12]
        dis_r = raw_obs[16]
        dis_rr = raw_obs[20]
        dis_sides = [dis_fr, dis_fl, dis_rl, dis_rr]
        v_ego = raw_obs[24]
        dis2light = raw_obs[26]
        red_light = raw_obs[27]
        dis_goal_ego = raw_obs[28]

        # efficiency
        reward = v_ego/5.0

        # safety
        collision_value = self.check_collision(dis_f, dis_r, dis_sides, vehicle_params)
        if collision_value is True:
            cost = 1.0
            done = True

        # infraction
        if (red_light == 1.0) and (dis2light < 15):
            infraction = 1.0
            infraction_check = True
            done = True
            print("+++>infraction:", infraction_check, red_light, dis2light)

        # navigation
        if dis_goal_ego < 15.0 or traci.vehicle.getLaneID(self.AutoCarID) == "E19_0":
            navigation = 100.0
            navigation_check = True
            done = True
            print(">>>>>>Touch down!!!")
        else:
            navigation = -np.log(1.0 + dis_goal_ego/self.max_dis_navigation) - 1.0

        return reward-cost-infraction+navigation, collision_value, cost, infraction_check, infraction, navigation_check, done

    def check_collision(self, dis_f, dis_r, dis_sides, vehicle_params):
        collision_value = False

        if (dis_f < 2.0) or (dis_r < 1.5) or (min(dis_sides) < 1.0):
            collision_value = True
            print("--->Checker-1: Collision!")
        elif self.AutoCarID not in vehicle_params:
            collision_value = True
            print("===>Checker-2: Collision!")

        return collision_value

    def step(self, action_a):
        acc, control_steer = action_a[0].item(), action_a[1].item()
        control_acc = self.max_acc * acc

        current_lane_id = traci.vehicle.getLaneID(self.AutoCarID)
        edge_id = traci.lane.getEdgeID(current_lane_id)
        max_lane_index = traci.edge.getLaneNumber(edge_id) - 1
        lane_index = traci.vehicle.getLaneIndex(self.AutoCarID)

        if -0.5 <= control_steer < 0.0:
            lane_change = 1  # left changing lane
        elif 0.0 <= control_steer < 0.5:
            lane_change = -1  # right changing lane
        else:
            lane_change = 0  # go straight

        if lane_index == max_lane_index and lane_change == 1:
            lane_change = 0

        if lane_index == 0 and lane_change == -1:
            lane_change = 0

        if edge_id == 'E18':
            traci.vehicle.changeLane(self.AutoCarID, lane_index + lane_change, 0)
            # print(">>>>>>", current_lane_id, edge_id)

        traci.vehicle.setSpeed(self.AutoCarID, max(traci.vehicle.getSpeed(self.AutoCarID) + control_acc, 0.001))
        traci.simulationStep()

        # Get the new vehicle parameters
        new_vehicle_params = traci.vehicle.getIDList()
        reward_cost, collision_value, cost, infraction_check, infraction, navigation_check, done = self.get_reward(new_vehicle_params)
        next_state, info = self.obs_to_state(new_vehicle_params)

        return reward_cost, next_state, collision_value, cost, infraction_check, infraction, navigation_check, done, info

    def reset(self):
        dom = xml.dom.minidom.parse(config_path)
        root = dom.documentElement
        random_seed_element = root.getElementsByTagName("seed")[0]

        if self.reset_times % 2 == 0:
            random_seed = "%d" % self.reset_times
            random_seed_element.setAttribute("value", random_seed)

        with open(config_path, "w") as file:
            dom.writexml(file)

        traci.load(["-c", config_path])
        print('Resetting the layout!!!!!!', self.reset_times)
        self.reset_times += 1

        AutoCarAvailable = False
        while AutoCarAvailable == False:
            traci.simulationStep()
            VehicleIds = traci.vehicle.getIDList()
            if self.AutoCarID in VehicleIds:
                AutoCarAvailable = True

        # Just check if the auto car still exisits and that there has not been any collision
        for VehId in VehicleIds:
            if VehId == self.AutoCarID:
                traci.vehicle.setSpeedMode(VehId, 22)
                traci.vehicle.setLaneChangeMode(VehId, 1)  # Disable automatic lane changing

        initial_state, _ = self.obs_to_state(VehicleIds)

        return initial_state

    def close(self):
        traci.close()

    def start(self, gui=False):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        traci.start([sumoBinary, "-c", config_path, "--collision.check-junctions", "true"])



