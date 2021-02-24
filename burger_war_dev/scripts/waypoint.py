#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import math
import numpy as np


class Waypoints:

    def __init__(self, path, side):
        self.points = []
        self.number = 0
        self.Waypoints_Lap = 0
        self.next_target_idx = -1
        self.all_field_score = np.ones([18])  # field score state
        self._load_waypoints(path, side)
        print ('number of waypoints: '+str(len(self.points)))

    def _load_waypoints(self, path, side):
        with open(path) as f:
            lines = csv.reader(f)
            for l in lines:
                # x,y,radian,target_idx(refer main code)
                point = [float(n) for n in l]
                point[2] = point[2]*math.pi/180.0
                if side == 'r':
                    point[3] = int(point[3])
                else:
                    point[3] = int(point[4])
                print point
                self.points.append(point[0:4])

    def get_next_waypoint(self):
        self.number = self.number+1
        if self.number == len(self.points):
            self.Waypoints_Lap = self.Waypoints_Lap+1
            print("next lap!!!!!!")
            self.number = 0

        if self.Waypoints_Lap == 0:
            return self.points[self.number][0:3]
        elif self.Waypoints_Lap > 0:
            print("search target !!!!!!", self.all_field_score)
            for i in range(self.number, len(self.points))+range(self.number):
                score_num = self.points[i][3]
                print score_num
                if score_num == -1:
                    # 得点と関係ないwaypoint
                    continue

                if self.all_field_score[score_num-6] == 0:
                    continue
                else:
                    print i
                    self.number = i
                    return self.points[i][0:3]

            print("got all field score !!!")
            return self.points[self.number][0:3]
        return self.points[self.number][0:3]

        # disable
        # check if better target to get score.
        #self.better_number = self.number
        # for i in range(len(self.points)):
        #    # check next target status
        #    self.next_target_idx = self.points[self.better_number][3]
        #    if self.all_field_score[self.next_target_idx] != 0:
        #        print("better target to get score", self.number)
        #        self.number = self.better_number
        #        break
        #
        #    self.better_number = self.better_number+1
        #    if self.better_number == len(self.points):
        #        self.better_number = 0

    def get_current_waypoint(self):
        return self.points[self.number]

    def get_current_target_number(self):
        # target No.
        return self.points[self.number][3]

    def get_any_waypoint(self, n):
        return self.points[n]

    def set_number(self, n):
        self.number = n

    def set_field_score(self, n):
        self.all_field_score = n
        # print(self.all_field_score)


# if __name__ == "__main__":
    # Waypoints('waypoints.csv')
