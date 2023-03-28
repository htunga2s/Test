import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


plt.figure(figsize=(12,12))
class Line:
    """An object used to create a line from two points

    :param start: First point used to generate the line. It's an
                  array of form [x,y].
    :type start: numpy.ndarray
    :param end: Second point used to generate the line. It's an
                array of form [x,y].
    :type end: numpy.ndarray
    """
    def __init__(self, start: np.ndarray, end: np.ndarray):
        if np.shape(start)!= (2,):
            raise ValueError("Start point must have the shape (2,)")
        if np.shape(end) != (2,):
            raise ValueError("End point must have the shape (2,)")
        if (start==end).all():
            raise ValueError("Start and end points must be different")
        
        # Calculate useful properties of the line
        self.start = start
        self.line = end - start
        self.length = np.linalg.norm(self.line)
        self.unit_line = self.line / self.length
        
    def point_dist(self, point: np.ndarray):
        """
        To calculate the closest distance to a line, we calculate
        the orthogonal distance from the point to the line.

        :param point: Numpy array of form [x,y], describing the point.
        :type point: numpy.ndarray
        :return: The minimum distance to a point.
        :rtype: float
        """
        if np.shape(point)!= (2,):
            raise ValueError("Start point must have the shape (2,)")
        # compute the perpendicular distance to the theoretical infinite line
        return np.linalg.norm(np.cross(self.line, self.start - point)) /self.length
    
    def equation(self):
        """Calculate the basic linear equation parameters useful for plotting

        :return: (m, c) where m is the gradient of the line
                 and c is the y intercept
        :rtype: tuple of floats
        """
        m = self.line[1]/self.line[0]
        c = self.start[1] - m*self.start[0]
        return (m, c)


def RANSAC(points: list, dist_thresh: int, k: int, thresh_count:int):
    """
    RANSAC algorithm to fit a line to a set of 2D points

    :param points: The list of points to fit a line where each point
                   is an numpy array in the form array of form [x,y].
    :type points: numpy.ndarray
    :param dist_thresh: The parameter used to determine the furthest a point
                        can be from the line and still be considered an inlier
    :type dist_thresh: int
    :param k: The number of iterations the RANSAC algorithm will run
    :type k: int
    :param thresh_count: number of minimum points required to form a line
    :type thresh_count: int
    :return: A tuple of two numpy arrays which are the two points which produce
             a line with the most inliers from all iterations
    :rtype: tuple
    """
    if len(points) == 0:
        return False
    
    indexes = list(range(0, len(points)))
    random.seed(111)
    inliers = dict()
    for _ in range(0,k):
        sample_points = indexes.copy()
        random.shuffle(sample_points)
        start = sample_points.pop()
        end = sample_points.pop()

        # selecting the pair of points which were not selected in earlier samples
        if((start,end) not in inliers) and ((end,start) not in inliers):
            inliers[(start,end)]=[]
            line = Line(points[start],points[end])
            for point_idx in sample_points:
                if line.point_dist(points[point_idx]) < dist_thresh:
                    inliers[(start,end)].append(points[point_idx])
        else:
            print("current sample points were sampled before")

    best_pair = None
    for cur_pair in inliers:
        if best_pair == None:
            best_pair = cur_pair
        elif len(inliers[cur_pair]) > len(inliers[best_pair]):
            best_pair = cur_pair

    if len(inliers[best_pair]) < thresh_count:
        print("Number of inliers is ", len(inliers[best_pair]),", which is lesser than the threshold")
        return False
    
    print("num inliers", len(inliers[best_pair]))
    best_point_1 = points[best_pair[0]]
    best_point_2 = points[best_pair[1]]    
    return (best_point_1, best_point_2, inliers[best_pair])

def get_line_params(laser_sub_cartesian: np.ndarray ,dist_thresh: int, k: int, thresh_count: int):
    """
    Function to extract line parameters from laser scan data
    :param laser_sub_cartesian: 2D array representing laser scan data in cartesian coordinates
    :param type: np.ndarray
    :return m_c_start_end: tuple of slope, constant of line equation, start and end points of line as 1D arrays 
    :param type: tuple
    :param dist_thresh: The parameter used to determine the furthest a point
                        can be from the line and still be considered an inlier
    :type dist_thresh: int
    :param k: The number of iterations the RANSAC algorithm will run
    :type k: int
    :param thresh_count: number of minimum points required to form a line
    :type thresh_count: int
    :return: A list of multiple line parameters, which by itself is a list consisting of 
    line parameters
    :rtype: list    
    """
    points = [point for point in laser_sub_cartesian]
    m_c_start_end = []
    # m_c_start_end: list of tuples. Each tuple: (slope, constant, end_point_array)
    while True:
        params = RANSAC(points, dist_thresh, k, thresh_count)
        if params:
            p1, p2, best_inliers = params
            # dtype(best_inliers): list of array of individual points
        else:
            break
            
        # updating points list by removing all inliers that were found
        for point in best_inliers:
            points = [x for x in points if not (x == point).all()]
        best_inliers = np.array(best_inliers)
        best_line = Line(p1, p2)
        m, c = best_line.equation()
        end_points = None
        min_x, max_x = min(best_inliers[:, 0]), max(best_inliers[:, 0])
        min_y, max_y = min(best_inliers[:, 1]), max(best_inliers[:, 1])
        if abs(min_x-max_x) >= abs(min_y-max_y):
            end_points = np.array([[min_x, m*min_x+c], [max_x, m*max_x+c]])
        else:
            end_points = np.array(
                [[(min_y-c)/m, min_y], [(max_y-c)/m, max_y]])
        m_c_start_end.append([m, c, end_points])

    return m_c_start_end

def np_polar2rect(np_array):
    center = np.array([0,0])
    r = np_array.T[0,]
    theta = np_array.T[1,]
    x = r*np.sin(np.deg2rad(theta))
    y = r*np.cos(np.deg2rad(theta))
    return np.array([x, y]).T
        
def reduction_filter(data_points, sigma, k):
    '''
    Method to reduce number of laser scan points by replacing cluster of points
    with their representative.
    :param data_points: 2D array representing laser scan data in cartesian coordinates
    :param type: np.ndarray
    :param sigma: maximum distance between two consecutive points to consider them 
    in same cluster
    :param type: float    
    :param k: maximum number of points allowed in a cluster
    :param type: int    
    '''
    points_list = list(data_points)
    output = []
    while len(points_list) > 0:
        a_i = points_list.pop(0)
        cur_sum = np.array(a_i)
        i = 1
        while len(points_list) > 0 and abs(a_i[0] - points_list[0][0]) < sigma \
        and i <= k:
            cur_sum += np.array(points_list.pop(0))
            i += 1
        output.append(cur_sum/i)
    return np.array(output)

def exec_ranssac(data):
    print("Reduction filter : ")
    ind = range(1,len(data)+1)
    df = pd.DataFrame(data=data,index=ind,columns=['distance'])    # 1st column as index
    df["angle"] = np.linspace(np.rad2deg(-1.91),np.rad2deg(1.91),len(data))
    data_points_unfiltered = np.array(df[['distance', 'angle']])
    data_points = np.array(data_points_unfiltered)
    reduced_points = np_polar2rect(reduction_filter(data_points, 40, 6))
    print(reduced_points)
    array = np_polar2rect(reduction_filter(data_points, 40, 6))
    points = [point for point in array]
    plot_points = np.array(points)

    m_c_start_end = get_line_params(laser_sub_cartesian=array, 
                                dist_thresh=10, k=10, thresh_count=10)
   
    plt.scatter(plot_points[:,0], plot_points[:,1])
    for line_num in range(len(m_c_start_end)):
        _,_,end_points = m_c_start_end[line_num]
        plt.plot(end_points[:,0], end_points[:,1], label='line '+str(line_num+1))
    plt.grid()
    plt.title("RANSAC - finding line features")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend()

#subscriber
def scancb(msg):
    global section
    points = np.array(msg.ranges)
    return exec_ranssac(points)
    # len_of_scan = len(ranges)

    # midle_point = len_of_scan/2
def main():
    scan_sub = rospy.Subscriber('/scan_filtered', LaserScan, scancb)
    rospy.spin()
    plt.show()
    
if __name__ == '__main__':
    rospy.init_node('wallfollowing', anonymous=True)
    main()
    
    # cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)