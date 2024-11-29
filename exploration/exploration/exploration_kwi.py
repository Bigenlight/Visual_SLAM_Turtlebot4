import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
import math

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.exploring = False
        self.max_frontier_distance = 5.0  # 최대 탐사 거리 (미터 단위)
        self.min_frontier_distance = 0.5  # 최소 목표 거리 (미터 단위)
        self.max_retries = 3  # 목표 재시도 횟수
        self.retry_count = 0
        self.goal_timeout = 60.0  # 목표 도달 타임아웃 (초 단위)

        # 퍼블리셔 추가: cmd_vel 토픽에 정지 명령을 보내기 위해
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Set up TF listener to get robot's current pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait for the action server to be ready
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('NavigateToPose action server ready.')

    def map_callback(self, msg):
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        if not self.exploring:
            self.exploring = True
            self.find_and_navigate_to_frontier()

    def find_and_navigate_to_frontier(self):
        # Frontier detection logic
        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info('Exploration complete!')
            self.exploring = False
            self.stop_robot()
            return

        # Select the closest valid frontier within min and max distance
        goal_position = self.select_frontier(frontiers)

        if goal_position is None:
            self.get_logger().info('No reachable frontiers found within the specified distance range.')
            self.exploring = False
            self.stop_robot()
            return

        # Create and send a navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position = goal_position
        goal_msg.pose.pose.orientation.w = 1.0  # Facing forward

        self.get_logger().info(f'Navigating to frontier at ({goal_position.x:.2f}, {goal_position.y:.2f})')

        send_goal_future = self.navigator.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

        # Start a timer for goal timeout
        self.create_timer(self.goal_timeout, self.goal_timeout_callback, callback_group=None)

    def detect_frontiers(self):
        # Implement frontier detection logic
        # Return a list of Point objects representing frontiers

        unknown = np.argwhere(self.map_data == -1)
        frontiers = []

        # For each unknown cell, check if it is adjacent to a free cell
        for cell in unknown:
            y, x = cell
            neighbors = self.get_neighbors(x, y)
            for nx, ny in neighbors:
                if self.map_data[ny, nx] == 0:
                    # This is a frontier cell
                    mx, my = self.grid_to_map(x, y)
                    frontiers.append(Point(x=mx, y=my, z=0.0))
                    break  # No need to check other neighbors

        return frontiers

    def select_frontier(self, frontiers):
        # Select the closest reachable frontier within min and max frontier distance

        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warning('Could not get robot position. Selecting the first frontier.')
            return frontiers[0] if frontiers else None

        # Compute distances and filter frontiers within min and max distance
        valid_frontiers = []
        distances = []
        for frontier in frontiers:
            dx = frontier.x - robot_position.x
            dy = frontier.y - robot_position.y
            distance = math.hypot(dx, dy)
            if self.min_frontier_distance <= distance <= self.max_frontier_distance:
                valid_frontiers.append(frontier)
                distances.append(distance)

        if not valid_frontiers:
            return None

        # Select the frontier with the minimum distance
        min_index = np.argmin(distances)
        return valid_frontiers[min_index]

    def get_neighbors(self, x, y):
        # Get valid neighboring cells (8-connected grid for better frontier detection)
        neighbors = []
        width = self.map_data.shape[1]
        height = self.map_data.shape[0]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))
        return neighbors

    def grid_to_map(self, x, y):
        # Convert grid coordinates to map coordinates
        mx = self.map_info.origin.position.x + (x + 0.5) * self.map_info.resolution
        my = self.map_info.origin.position.y + (y + 0.5) * self.map_info.resolution
        return mx, my

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return trans.transform.translation
        except Exception as e:
            self.get_logger().error(f'Could not get robot pose: {e}')
            return None

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self.exploring = False
            self.stop_robot()
            return

        self.get_logger().info('Goal accepted :)')
        self.current_goal = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
            self.retry_count = 0  # Reset retry count on success
        else:
            self.get_logger().info(f'Goal failed with status: {status}')
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.get_logger().warn('Maximum retries reached. Stopping exploration.')
                self.exploring = False
                self.stop_robot()
                return
            else:
                self.get_logger().info('Retrying with a new frontier.')

        # After reaching the goal or failing, look for the next frontier
        self.find_and_navigate_to_frontier()

    def feedback_callback(self, feedback_msg):
        # Process feedback from the action server if needed
        pass

    def goal_timeout_callback(self):
        if self.current_goal is not None:
            self.get_logger().warn('Goal timeout reached. Cancelling the current goal.')
            self.navigator.async_cancel_goal(self.current_goal)  # 수정된 메서드 호출
            self.current_goal = None
            self.exploring = False  # Reset exploring flag to allow retry
            self.stop_robot()
            self.find_and_navigate_to_frontier()

    def stop_robot(self):
        # Send zero velocity to stop the robot
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_msg)
        self.get_logger().info('Sent stop command to the robot.')

def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        explorer.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        explorer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
