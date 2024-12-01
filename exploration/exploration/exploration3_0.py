#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import math
from sklearn.cluster import DBSCAN
from std_msgs.msg import Int32

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('exploration3_0')

        # ==== Hyperparameters with ROS2 Parameters ====
        # Frontier detection parameters
        self.declare_parameter('min_frontier_size', 5)
        self.declare_parameter('dbscan_eps', 0.7)
        self.declare_parameter('dbscan_min_samples', 2)

        # Goal selection parameters
        self.declare_parameter('min_frontier_distance', 0.36)
        self.declare_parameter('goal_reached_tolerance', 0.3)

        # Retrieve parameter values
        self.min_frontier_size = self.get_parameter('min_frontier_size').get_parameter_value().integer_value
        self.dbscan_eps = self.get_parameter('dbscan_eps').get_parameter_value().double_value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').get_parameter_value().integer_value
        self.min_frontier_distance = self.get_parameter('min_frontier_distance').get_parameter_value().double_value
        self.goal_reached_tolerance = self.get_parameter('goal_reached_tolerance').get_parameter_value().double_value

        # ==== Publishers and Subscribers ====
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.num_frontiers_publisher = self.create_publisher(Int32, '/num_of_frontiers', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # ==== Initialize NavigateToPose action client ====
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ==== Initialize variables ====
        self.map_data = None
        self.map_info = None
        self.current_goal_handle = None
        self.exploring = False          # True when moving to a goal
        self.current_goal_position = None

        # ==== Initialize TF2 Buffer and Listener ====
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ==== Wait for action server to be ready ====
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('NavigateToPose action server is ready.')

        # ==== Create a timer to check if goal is reached ====
        self.goal_check_timer = self.create_timer(1.0, self.check_goal_reached)

    def map_callback(self, msg):
        """
        Callback function for /map topic.
        Updates map data and starts exploration if not already exploring.
        """
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        # Debugging map info
        self.get_logger().debug(f'Map resolution: {self.map_info.resolution}, Origin: ({self.map_info.origin.position.x}, {self.map_info.origin.position.y})')

        if not self.exploring:
            self.get_logger().info('Starting frontier detection.')
            # 1. Detect frontiers
            frontiers = self.detect_frontiers()

            # Publish number of frontiers
            num_frontiers_msg = Int32()
            num_frontiers_msg.data = len(frontiers)
            self.num_frontiers_publisher.publish(num_frontiers_msg)

            if not frontiers:
                self.get_logger().info('No frontiers detected.')
                # No frontiers remaining, exploration complete
                self.shutdown()
                return

            # 2. Select the nearest frontier
            goal_position = self.select_frontier(frontiers)

            if goal_position is None:
                self.get_logger().info('No valid frontiers found.')
                self.shutdown()
                return

            # 3. Send navigation goal
            self.send_navigation_goal(goal_position)

    def detect_frontiers(self):
        """
        Detects frontiers in the map.
        Returns a list of frontiers, each frontier is a numpy array of points (x, y).
        """
        if self.map_data is None:
            self.get_logger().warn('Map data is not available.')
            return []

        # Frontier cells are free cells (0) adjacent to unknown cells (-1)
        height, width = self.map_data.shape
        frontier_points = []

        # For all free cells, check if they have at least one unknown neighbor
        for y in range(height):
            for x in range(width):
                if self.map_data[y, x] == 0:
                    neighbors = self.get_neighbors(x, y)
                    for nx, ny in neighbors:
                        if self.map_data[ny, nx] == -1:
                            # Convert grid coordinates to map coordinates
                            mx, my = self.grid_to_map(x, y)
                            frontier_points.append([mx, my])
                            break  # No need to check other neighbors

        self.get_logger().info(f'Detected {len(frontier_points)} frontier points.')

        if not frontier_points:
            self.get_logger().debug('No frontier points detected after initial scanning.')
            return []

        # Cluster frontier points to group them into frontiers
        frontiers = self.cluster_frontiers(frontier_points)

        return frontiers

    def cluster_frontiers(self, frontier_points):
        """
        Clusters frontier points into frontiers.
        Returns a list of frontiers, each frontier is a numpy array of points.
        """
        if not frontier_points:
            return []

        # Use DBSCAN to cluster frontier points
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(frontier_points)
        labels = clustering.labels_

        # Number of clusters in labels, ignoring noise if present
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label

        frontiers = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            cluster = np.array(frontier_points)[class_member_mask]
            if len(cluster) >= self.min_frontier_size:
                frontiers.append(cluster)
                self.get_logger().debug(f'Frontier cluster {k} accepted with size {len(cluster)}.')
            else:
                self.get_logger().debug(f'Frontier cluster {k} discarded due to small size {len(cluster)}.')

        self.get_logger().info(f'Number of frontiers after clustering and filtering: {len(frontiers)}')

        return frontiers

    def select_frontier(self, frontiers):
        """
        Selects the nearest frontier and returns the middle point.
        """
        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warn('Cannot get robot position.')
            return None

        # Compute distances to frontiers
        frontier_distances = []
        for frontier in frontiers:
            # Compute centroid of the frontier
            centroid = np.mean(frontier, axis=0)
            dx = centroid[0] - robot_position.x
            dy = centroid[1] - robot_position.y
            distance = math.hypot(dx, dy)
            frontier_distances.append((distance, centroid))

        if not frontier_distances:
            self.get_logger().info('No frontiers to select.')
            return None

        # Sort frontiers by distance
        frontier_distances.sort()
        nearest_frontier_distance, nearest_frontier_centroid = frontier_distances[0]

        self.get_logger().debug(f'Nearest frontier distance: {nearest_frontier_distance:.2f} meters.')

        # Check if distance is within the allowed range
        if nearest_frontier_distance < self.min_frontier_distance:
            self.get_logger().info(f'Nearest frontier is too close ({nearest_frontier_distance:.2f}m). Skipping.')
            return None

        # Create Point for the goal position
        goal_position = Point()
        goal_position.x = nearest_frontier_centroid[0]
        goal_position.y = nearest_frontier_centroid[1]
        goal_position.z = 0.0

        self.get_logger().info(f'Selected frontier at ({goal_position.x:.2f}, {goal_position.y:.2f})')

        return goal_position

    def send_navigation_goal(self, goal_position):
        """
        Sends a navigation goal to the specified position.
        """
        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position = goal_position
        goal_msg.pose.pose.orientation.w = 1.0  # Forward-facing orientation

        self.get_logger().info(f'Navigating to position: ({goal_position.x:.2f}, {goal_position.y:.2f})')

        # Send goal
        self.exploring = True  # Set exploring to True to stop frontier detection
        self.current_goal_position = goal_position

        send_goal_future = self.navigator.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def get_robot_pose(self):
        """
        Retrieves the robot's current position in the 'map' frame.
        """
        try:
            # Ensure the transform is available
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.get_logger().debug(f'Robot position: ({trans.transform.translation.x:.2f}, {trans.transform.translation.y:.2f})')
            return trans.transform.translation
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Unable to get robot position: {e}')
            return None

    def get_neighbors(self, x, y):
        """
        Returns the 4-connected neighbors of a given cell.
        """
        neighbors = []
        width = self.map_data.shape[1]
        height = self.map_data.shape[0]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
        return neighbors

    def grid_to_map(self, x, y):
        """
        Converts grid coordinates to map coordinates.
        """
        mx = self.map_info.origin.position.x + (x + 0.5) * self.map_info.resolution
        my = self.map_info.origin.position.y + (y + 0.5) * self.map_info.resolution
        return mx, my

    def goal_response_callback(self, future):
        """
        Callback function to handle the response from the action server after sending a goal.
        """
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')
            self.exploring = False
            return

        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected by the action server.')
            self.exploring = False
            return

        self.get_logger().info('Goal accepted by the action server.')
        self.current_goal_handle = goal_handle

        # Get the result asynchronously
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Callback function to handle the result from the action server after goal completion.
        """
        try:
            result = future.result()
            status = result.status
        except Exception as e:
            self.get_logger().error(f'Failed to get result of the goal: {e}')
            self.exploring = False
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Reached the goal successfully!')
        else:
            self.get_logger().info(f'Goal failed with status: {status}')

        # Reset exploring flag to allow the next exploration
        self.exploring = False

    def feedback_callback(self, feedback_msg):
        """
        Callback function to handle feedback from the action server.
        Currently unused but can be implemented as needed.
        """
        pass

    def check_goal_reached(self):
        """
        Checks if the robot is close enough to the goal.
        """
        if not self.exploring:
            return  # Not currently moving to a goal

        robot_position = self.get_robot_pose()
        if robot_position is None or self.current_goal_position is None:
            return

        dx = robot_position.x - self.current_goal_position.x
        dy = robot_position.y - self.current_goal_position.y
        distance = math.hypot(dx, dy)
        self.get_logger().debug(f'Distance to goal: {distance:.2f} meters.')

        if distance <= self.goal_reached_tolerance:
            self.get_logger().info('Goal reached within tolerance.')
            self.cancel_current_goal()
            self.exploring = False  # Allow frontier detection to proceed

    def cancel_current_goal(self):
        """
        Cancels the current goal.
        """
        if self.current_goal_handle is not None:
            self.get_logger().info('Cancelling current goal.')
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done)
            self.current_goal_handle = None
            self.current_goal_position = None

    def cancel_done(self, future):
        self.get_logger().info('Goal cancel request completed.')

    def shutdown(self):
        """
        Stops exploration and gracefully shuts down the node.
        """
        self.get_logger().info('Exploration complete. Shutting down the node.')

        # Stop the robot
        self.stop_robot()

        # Destroy the node and shutdown
        self.destroy_node()
        rclpy.shutdown()

    def stop_robot(self):
        """
        Publishes zero velocities to stop the robot.
        """
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
        self.get_logger().info('Published stop command to the robot.')


def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()

    # Use a MultiThreadedExecutor to allow timers and callbacks to run in parallel
    executor = rclpy.executors.MultiThreadedExecutor()
    try:
        rclpy.spin(explorer, executor=executor)
    except KeyboardInterrupt:
        explorer.get_logger().info('Keyboard interrupt received. Shutting down.')
        explorer.shutdown()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
