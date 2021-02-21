#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waypoint import Waypoints
from enemy_camera_detector import EnemyCameraDetector

from enum import Enum
import math
import os
import cv2
import numpy as np
import requests

import rospy
import tf
import actionlib
import angles


from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError

from std_srvs.srv import Empty, EmptyRequest, EmptyResponse


class ActMode(Enum):
    BASIC = 1
    ATTACK = 2
    ESCAPE = 3
    DEFENCE = 4

# --- target definition (r), refer http://localhost:5000/warstate ---
# the number means index of warstate json file.
# the target state is stored in all_field_score param (0:no one,1:mybot,2:enemy)
#
#      6   [zone3]  8
#            14
#      7            9
# [zone2] 16    15 [zone4]
#      10           12
#            17
#      11  [zone1]  13
# ----------------------------------------
#        Back 0                  Back 3
#   R 2[enemy_bot(b)]L 1   R 5[my_bot(r)]L 4
#        Front                   Front
# ----------------------------------------


class SeigoBot2:

    def __init__(self):

        def load_waypoint():
            path = os.environ['HOME'] + \
                '/catkin_ws/src/burger_war_dev/burger_war_dev/scripts/waypoints.csv'
            return Waypoints(path, self.my_side)

        self.listener = tf.TransformListener()

        self.move_base_client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction)
        if not self.move_base_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo('wait move base server')
        rospy.loginfo('server comes up!!')
        self.status = self.move_base_client.get_state()

        rospy.Subscriber('enemy_position', Odometry,
                         self.enemy_position_callback)
        self.enemy_position = Odometry()
        self.enemy_info = [0.0, 0.0]
        self.detect_counter = 0

        rospy.Subscriber('scan', LaserScan, self.lidar_callback)
        self.scan = LaserScan()

        # rospy.Subscriber('image_raw', Image, self.imageCallback)
        self.camera_detector = EnemyCameraDetector()
        self.is_camera_detect = False
        self.camera_detect_angle = -360

        self.game_timestamp = 0
        self.my_score = 0
        self.enemy_score = 0
        self.Is_lowwer_score = False
        self.all_field_score = np.ones([18])  # field score state
        self.all_field_score_prev = np.ones(
            [18])  # field score state (previous)
        self.enemy_get_target_no = -1
        self.enemy_get_target_no_timestamp = -1
        self.my_body_remain = 3
        self.enemy_body_remain = 3

        self.direct_twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        rospy.wait_for_service("/move_base/clear_costmaps")
        self.clear_costmap = rospy.ServiceProxy(
            "/move_base/clear_costmaps", Empty)

        self.act_mode = ActMode.BASIC
        self.get_rosparam()
        self.escape_mode_start_time = -1
        self.waypoint = load_waypoint()
        self.send_goal(self.waypoint.get_current_waypoint())
        # warstate callback should be called after all parameter is ready!!
        rospy.Timer(rospy.Duration(0.1), self.WarState_timerCallback)

    def get_rosparam(self):
        self.my_side = rospy.get_param('~side')
        self.robot_namespace = rospy.get_param('~robot_namespace')
        self.enemy_time_tolerance = rospy.get_param(
            '~detect_enemy_time_tolerance', default=0.5)
        self.snipe_th = rospy.get_param('~snipe_distance_th', default=0.8)
        self.distance_to_wall_th = rospy.get_param(
            '~distance_to_wall_th', default=0.15)
        self.counter_th = rospy.get_param('enemy_count_th', default=3)
        self.approach_distance_th = rospy.get_param(
            '~approach_distance_th', default=0.5)
        self.attack_angle_th = rospy.get_param(
            '~attack_angle_th', default=45*math.pi/180)
        self.camera_range_limit = rospy.get_param(
            '~camera_range_limit', default=[0.2, 0.5])
        self.camera_angle_limit = rospy.get_param(
            '~camera_angle_limit', default=30)*math.pi/180
        self.enable_escape_approach = rospy.get_param(
            '~enable_escape_approach', default="False")
        self.escape_approach_distance_th_min = rospy.get_param(
            '~escape_approach_distance_th_min', default=0.23)
        self.escape_approach_distance_th_max = rospy.get_param(
            '~escape_approach_distance_th_max', default=0.85)
        self.escape_approach_time_interval = rospy.get_param(
            '~escape_approach_time_interval', default=6)

    def imageCallback(self, data):
        #print("imageCallback+", rospy.Time.now())
        self.detect_from_camera(data)
        #print("imageCallback-", rospy.Time.now())

    def get_position_from_tf(self, c1, c2):
        trans = []
        rot = []
        try:
            (trans, rot) = self.listener.lookupTransform(
                c1, c2, rospy.Time(0))
            return trans, rot, True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('tf error')
            return trans, rot, False

    def enemy_position_callback(self, position):
        self.enemy_position = position

    def lidar_callback(self, scan):
        self.scan = scan

    def detect_enemy(self):
        exist, distance, direction_diff = self.detect_from_lidar()
        # もしカメラで確認できる範囲なら

        # if abs(direction_diff) < self.camera_angle_limit and distance > self.camera_range_limit[0] and distance < self.camera_range_limit[1]:
        #     exist = exist and self.is_camera_detect  # カメラとLidarのandをとる
        #     if exist == False:
        #         rospy.loginfo('detect enemy from LiDAR, but cannot detect from camera. So ignore')
        return exist, distance, direction_diff

    def detect_from_lidar(self):
        time_diff = rospy.Time.now().to_sec() - self.enemy_position.header.stamp.to_sec()
        if time_diff > self.enemy_time_tolerance:   # 敵情報が古かったら無視
            self.detect_counter = 0
            return False, 0.0, 0.0
        else:
            self.detect_counter = self.detect_counter+1
            if self.detect_counter < self.counter_th:
                return False, 0.0, 0.0

        map_topic = self.robot_namespace+"map"
        baselink_topic = self.robot_namespace+"base_link"
        trans, rot,  vaild = self.get_position_from_tf(
            map_topic, baselink_topic)
        if vaild == False:
            return False, 0.0, 0.0
        dx = self.enemy_position.pose.pose.position.x - trans[0]
        dy = self.enemy_position.pose.pose.position.y - trans[1]
        enemy_distance = math.sqrt(dx*dx+dy*dy)

        _, _, yaw = tf.transformations.euler_from_quaternion(rot)
        enemy_direction = math.atan2(dy, dx)
        enemy_direction_diff = angles.normalize_angle(enemy_direction-yaw)
        return True, enemy_distance, enemy_direction_diff

    def detect_from_camera(self, data):
        red_angle, green_angle, blue_angle = self.camera_detector.detect_enemy(
            data)
        if red_angle != -360:
            self.is_camera_detect = True
            self.camera_detect_angle = red_angle
            return
        else:
            if green_angle != -360:
                self.is_camera_detect = True
                self.camera_detect_angle = green_angle
            else:
                self.is_camera_detect = False
                self.camera_detect_angle = -360

    # RESPECT @koy_tak
    def detect_collision(self):
        front = False
        rear = False
        deg_90 = int((math.pi/2.0)/self.scan.angle_increment)

        front_count = len([i for i in self.scan.ranges[0:int(deg_90)] if i < self.distance_to_wall_th]) + \
            len([i for i in self.scan.ranges[int(deg_90)*3:-1]
                 if i < self.distance_to_wall_th])
        rear_count = len([i for i in self.scan.ranges[int(
            deg_90):int(deg_90)*3] if i < self.distance_to_wall_th])
        if front_count > 0 and rear_count == 0:
            front = True
            rospy.logwarn("front collision !!!")
        elif front_count == 0 and rear_count > 0:
            rear = True
            rospy.logwarn("rear collision !!!")
        elif front_count > 0 and rear_count > 0:
            front = front_count > rear_count
            rear = not front
            rospy.logwarn("both side collision !!!")

        # if (self.scan.ranges[0] != 0 and self.scan.ranges[0] < self.distance_to_wall_th) or (self.scan.ranges[10] != 0 and self.scan.ranges[10] < self.distance_to_wall_th) or (self.scan.ranges[350] != 0 and self.scan.ranges[350] < self.distance_to_wall_th):
        #     rospy.logwarn('front collision !!')
        #     front = True
        # if (self.scan.ranges[180] != 0 and self.scan.ranges[180] < self.distance_to_wall_th) or (self.scan.ranges[190] != 0 and self.scan.ranges[190] < self.distance_to_wall_th) or (self.scan.ranges[170] != 0 and self.scan.ranges[170] < self.distance_to_wall_th):
        #     rospy.logwarn('rear collision !!')
        #     rear = True
        return front, rear

    # RESPECT @F0CACC1A
    def WarState_timerCallback(self, state):
        self.getWarState()

    def getWarState(self):
        # get current state from judge server
        resp = requests.get(JUDGE_URL + "/warState")
        dic = resp.json()
        # get score
        if self.my_side == "r":  # red_bot
            self.my_score = int(dic["scores"]["r"])
            self.enemy_score = int(dic["scores"]["b"])
        else:  # blue_bot
            self.my_score = int(dic["scores"]["b"])
            self.enemy_score = int(dic["scores"]["r"])

        self.game_timestamp = int(dic["time"])

        # get warstate score state and compare previous value
        for idx in range(18):  # number of field targets, how to get the number?
            self.all_field_score_prev[idx] = self.all_field_score[idx]
            target_state = dic["targets"][idx]["player"]
            if target_state == "n":
                self.all_field_score[idx] = 1  # no one get this target
            elif target_state == self.my_side:
                self.all_field_score[idx] = 0  # my_bot get this target
            else:
                self.all_field_score[idx] = 2  # enemy get this target

            # check if field score is updated.
            if self.all_field_score[idx] != self.all_field_score_prev[idx]:
                if self.all_field_score[idx] == 2:
                    print("enemy get target No.", idx, self.game_timestamp)
                    self.enemy_get_target_no = idx
                    self.enemy_get_target_no_timestamp = self.game_timestamp
        # update field score state to check enemy get target

        self.waypoint.set_field_score(self.all_field_score[6:])

        # update body AR marker point
        if self.my_side == "b":
            self.my_body_remain = np.sum(self.all_field_score[0:3])
            self.enemy_body_remain = np.sum(self.all_field_score[3:6])
        elif self.my_side == "r":
            self.my_body_remain = np.sum(self.all_field_score[3:6])
            self.enemy_body_remain = np.sum(self.all_field_score[0:3])
        #print("my_body_remain: ", self.my_body_remain)
        #print("enemy_body_remain: ", self.enemy_body_remain)

        # update which bot is higher score
        if self.my_score <= self.enemy_score:
            self.Is_lowwer_score = True
        else:
            self.Is_lowwer_score = False
        #print("Is_lowwer_score", self.Is_lowwer_score)

    # ここで状態決定　
    def mode_decision(self):
        exist, distance, direction_diff = self.detect_enemy()
        if exist == False:  # いなかったら巡回
            return ActMode.BASIC
        else:
            # print self.enemy_body_remain, self.all_field_score[0:6]
            self.enemy_info = [distance, direction_diff]
            if self.my_body_remain == 0 and self.enemy_body_remain == 0:
                # 敵を検出していたとしても攻撃する必要がなければ巡回
                return ActMode.BASIC
            if self.enemy_body_remain <= 1 and distance < 1.0 and self.Is_lowwer_score == False:
                return ActMode.DEFENCE
            if distance < self.snipe_th and self.enable_escape_approach == True and self.Is_lowwer_score == True:
                # if low score, once attack then try to escape.
                return ActMode.ESCAPE
            if distance < self.snipe_th:  # 発見して近かったら攻撃
                return ActMode.ATTACK
            # rospy.loginfo('detect enemy but so far')
            return ActMode.BASIC

    def status_transition(self):
        def get_mode_txt(n):
            if n == ActMode.BASIC:
                return 'basic'
            elif n == ActMode.ATTACK:
                return 'attack'
            elif n == ActMode.ESCAPE:
                return 'escape'
            elif n == ActMode.DEFENCE:
                return 'defence'
            else:
                return 'unknown'

        pre_act_mode = self.act_mode
        self.act_mode = self.mode_decision()
        if self.act_mode == ActMode.BASIC:
            self.basic()
        elif self.act_mode == ActMode.ATTACK:
            self.attack()
        elif self.act_mode == ActMode.DEFENCE:
            self.defence()
        elif self.act_mode == ActMode.ESCAPE:
            if pre_act_mode != self.act_mode:
                self.escape_mode_start_time = rospy.Time.now().to_sec()
            self.escape()
        else:
            rospy.logwarn('unknown actmode !!!')

        if pre_act_mode != self.act_mode:
            text = 'change to ' + get_mode_txt(self.act_mode)
            rospy.loginfo(text)

    def basic(self):
        # vaild, vx = self.recovery()  # ぶつかってないか確認
        # if vaild == True:
        #     self.cancel_goal()
        #     cmd_vel = Twist()
        #     cmd_vel.linear.x = vx
        #     self.direct_twist_pub.publish(cmd_vel)
        #     return

        pre_status = self.status
        self.status = self.move_base_client.get_state()
        if pre_status != self.status:
            rospy.loginfo(self.move_base_client.get_goal_status_text())

        if self.status == actionlib.GoalStatus.ACTIVE:
            pass
        elif self.status == actionlib.GoalStatus.SUCCEEDED:
            # 早すぎてマーカー取れなかったらここでsleep
            rospy.sleep(0.3)
            point = self.waypoint.get_next_waypoint()
            self.send_goal(point)
        elif self.status == actionlib.GoalStatus.ABORTED:
            cmd_vel = Twist()
            valid, cmd_vel.linear.x = self.recovery()
            self.direct_twist_pub.publish(cmd_vel)
            # 必要ならsleep
        elif self.status == actionlib.GoalStatus.PENDING:
            self.send_goal(self.waypoint.get_current_waypoint())
        elif self.status == actionlib.GoalStatus.PREEMPTING or self.status == actionlib.GoalStatus.PREEMPTED:
            self.send_goal(self.waypoint.get_current_waypoint())
        else:
            return

    def send_goal(self, point):
        print self.clear_costmap.call()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.robot_namespace+"map"
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]

        q = tf.transformations.quaternion_from_euler(0, 0, point[2])
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        goal.target_pose.header.stamp = rospy.Time.now()

        self.move_base_client.send_goal(goal)
        rospy.sleep(0.5)

        rospy.loginfo('send goal')

    def cancel_goal(self):
        self.move_base_client.cancel_all_goals()
        return

    def get_move_base_status(self):
        return self.move_base_client.get_state()

    def attack(self):
        self.cancel_goal()
        cmd_vel = self.turn_to_enemy(self.enemy_info[1])
        # print(self.enemy_info[1]*180/math.pi)
        valid, vx = self.recovery()
        # print(valid, vx)
        if valid == True:
            cmd_vel.linear.x = vx
        else:
            if abs(self.enemy_info[1]) < self.attack_angle_th:
                cmd_vel.linear.x = self.enemy_info[0]-self.approach_distance_th
            else:
                cmd_vel.linear.x = 0.0
        self.direct_twist_pub.publish(cmd_vel)

    def escape(self):
        # if low score, once attack then try to escape.
        # change approach_distance_th by time_interval
        time_interval = self.escape_approach_time_interval
        number_of_time_interval = 3
        time_diff = rospy.Time.now().to_sec() - self.escape_mode_start_time
        time_diff = time_diff % (self.escape_approach_time_interval * number_of_time_interval)
        if time_diff < time_interval: # first interval
            local_approach_distance_th = self.approach_distance_th
        elif time_diff < time_interval*2: # second interval
            local_approach_distance_th = self.escape_approach_distance_th_min
        else: # third interval
            local_approach_distance_th = self.escape_approach_distance_th_max

        # same with attack mode --->
        self.cancel_goal()
        cmd_vel = self.turn_to_enemy(self.enemy_info[1])
        # print(self.enemy_info[1]*180/math.pi)
        valid, vx = self.recovery()
        # print(valid, vx)
        if valid == True:
            cmd_vel.linear.x = vx
        else:
            if abs(self.enemy_info[1]) < self.attack_angle_th:
                cmd_vel.linear.x = self.enemy_info[0]-local_approach_distance_th
            else:
                cmd_vel.linear.x = 0.0
        self.direct_twist_pub.publish(cmd_vel)
        return

    def defence(self):
        self.cancel_goal()
        cmd_vel = self.turn_to_enemy(self.enemy_info[1])
        print self.enemy_info
        valid, vx = self.recovery()
        if valid == True:
            cmd_vel.linear.x = vx
        else:
            cmd_vel.linear.x = 0.0
        self.direct_twist_pub.publish(cmd_vel)

    def turn_to_enemy(self, direction_diff):
        cmd_vel = Twist()
        if direction_diff > 60.0/180*math.pi:
            cmd_vel.angular.z = math.copysign(1.0, direction_diff) * 2.75
        else:
            cmd_vel.angular.z = direction_diff*2.0
        return cmd_vel

    def recovery(self):
        front, rear = self.detect_collision()
        if front == True:
            return True, -0.05
        elif rear == True:
            return True, 0.05
        else:
            return False, 0.0


def main():
    rospy.init_node('seigo_run2')
    node = SeigoBot2()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        node.status_transition()
        rate.sleep()


if __name__ == "__main__":
    JUDGE_URL = rospy.get_param('/send_id_to_judge/judge_url')
    main()
