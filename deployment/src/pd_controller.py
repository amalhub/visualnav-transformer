import numpy as np
import yaml
from typing import Tuple

# ROS2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
						 REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1 / robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1  # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi / 4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None


def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi


def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx)) / DT
	elif np.abs(dx) < EPS:
		v = 0
		w = np.sign(dy) * np.pi / (2 * DT)
	else:
		v = dx / DT
		w = np.arctan(dy / dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


class PDControllerNode(Node):
	def __init__(self):
		super().__init__("pd_controller")
		self.vel_msg = Twist()
		self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
		self.reached_goal = False
		self.reverse_mode = False

		self.waypoint_sub = self.create_subscription(
			Float32MultiArray, WAYPOINT_TOPIC, self.callback_drive, 1
		)
		self.reached_goal_sub = self.create_subscription(
			Bool, REACHED_GOAL_TOPIC, self.callback_reached_goal, 1
		)
		self.vel_out = self.create_publisher(Twist, VEL_TOPIC, 1)
		self.timer = self.create_timer(1.0 / RATE, self.timer_callback)

		self.get_logger().info("Node initialized. Waiting for waypoints...")

	def callback_drive(self, waypoint_msg: Float32MultiArray):
		"""Callback function for the waypoint subscriber"""
		self.get_logger().info("Setting waypoint")
		self.waypoint.set(waypoint_msg.data)

	def callback_reached_goal(self, reached_goal_msg: Bool):
		"""Callback function for the reached goal subscriber"""
		self.reached_goal = reached_goal_msg.data

	def timer_callback(self):
		"""Main control loop"""
		self.vel_msg = Twist()
		if self.reached_goal:
			self.vel_out.publish(self.vel_msg)
			self.get_logger().info("Reached goal! Stopping...")
			return
		elif self.waypoint.is_valid(verbose=True):
			v, w = pd_controller(self.waypoint.get())
			if self.reverse_mode:
				v *= -1
			self.vel_msg.linear.x = v
			self.vel_msg.angular.z = w
			self.get_logger().info(f"Publishing new velocity: {v}, {w}")
		self.vel_out.publish(self.vel_msg)


def main(args=None):
	rclpy.init(args=args)
	node = PDControllerNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
