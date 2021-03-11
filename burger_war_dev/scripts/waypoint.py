#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import math
import numpy as np

FIELD_SCORE_NUM_OFFSET=6

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

        print("search target !!!!!!", self.all_field_score)
        for i in range(self.number, len(self.points))+range(self.number):
            score_num = self.points[i][3]
            print score_num

            # 得点と関係ないwaypoint
            if score_num == -1:
                ## 1週目は得点と関係ないwaypointも辿る。
                #if self.Waypoints_Lap == 0:
                #    return self.points[self.number][0:3]
                #continue
                # 常に得点と関係ないwaypointは必ず辿る（常に外周を走行するため）
                return self.points[self.number][0:3]

            # 得点と関係あるwaypoint
            if self.all_field_score[score_num - FIELD_SCORE_NUM_OFFSET] == 0:
                # if already get score, skip search
                continue
            else:
                # if not get score, go to target
                print i
                self.number = i
                return self.points[i][0:3]

        print("got all field score !!!")
        return self.points[self.number][0:3]

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

    def check_if_get_field_score(self, n):
        score_num = n
        if self.all_field_score[score_num - FIELD_SCORE_NUM_OFFSET] == 0:
            return True
        else:
            return False


# if __name__ == "__main__":
    # Waypoints('waypoints.csv')
