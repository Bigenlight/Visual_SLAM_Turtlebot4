import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
import math
from sklearn.cluster import DBSCAN

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Set logging level to DEBUG
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # Subscription to the /map topic
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # ActionClient for NavigateToPose
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize variables
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.exploring = False
        self.max_frontier_distance = 20.0  # Adjusted maximum exploration distance
        self.min_frontier_distance = 0.25  # Adjusted minimum goal distance
        self.safety_distance = 0.1  # Adjusted safety distance
        self.max_retries = 3  # Maximum number of goal retries
        self.retry_count = 0
        self.goal_timeout = 30.0  # Goal reach timeout in seconds

        # Publisher to cmd_vel to stop the robot
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # TF2 Buffer and Listener
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

        # Cluster the frontiers
        clustered_frontiers = self.cluster_frontiers(frontiers)

        # Select the closest valid frontier within min and max distance
        goal_position = self.select_frontier(clustered_frontiers)

        if goal_position is None:
            self.get_logger().info('No reachable and safe frontiers found within the specified distance range.')
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
        self.goal_timer = self.create_timer(self.goal_timeout, self.goal_timeout_callback, callback_group=None)

    def detect_frontiers(self):
        # Find free cells that are adjacent to unknown cells
        if self.map_data is None:
            return []

        height, width = self.map_data.shape
        frontier_points = []

        # Iterate over all free cells
        free_cells = np.argwhere(self.map_data == 0)

        for cell in free_cells:
            y, x = cell
            neighbors = self.get_neighbors(x, y)
            for nx, ny in neighbors:
                if self.map_data[ny, nx] == -1:
                    # This free cell is adjacent to an unknown cell; it's a frontier
                    mx, my = self.grid_to_map(x, y)
                    frontier_points.append([mx, my])
                    break  # No need to check other neighbors

        self.get_logger().info(f'Detected {len(frontier_points)} frontier points.')
        return frontier_points

    def cluster_frontiers(self, frontiers):
        # Cluster the frontier points using DBSCAN
        if not frontiers:
            return []

        clustering = DBSCAN(eps=0.5, min_samples=2).fit(frontiers)
        labels = clustering.labels_

        unique_labels = set(labels)
        clustered_frontiers = []

        for label in unique_labels:
            if label == -1:
                # Noise points; you can choose to include or exclude them
                continue
            indices = np.where(labels == label)[0]
            cluster = np.array(frontiers)[indices]
            # Compute the centroid of the cluster
            centroid = np.mean(cluster, axis=0)
            point = Point(x=centroid[0], y=centroid[1], z=0.0)
            clustered_frontiers.append(point)

        self.get_logger().info(f'Clustered into {len(clustered_frontiers)} frontiers.')
        return clustered_frontiers

    def select_frontier(self, frontiers):
        # Select the closest frontier within the specified distance range and ensure it's safe
        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warning('Could not get robot position. Selecting the first frontier.')
            return frontiers[0] if frontiers else None

        valid_frontiers = []
        distances = []
        for frontier in frontiers:
            dx = frontier.x - robot_position.x
            dy = frontier.y - robot_position.y
            distance = math.hypot(dx, dy)
            if self.min_frontier_distance <= distance <= self.max_frontier_distance:
                # Check if the goal is safe
                if self.is_goal_safe(frontier.x, frontier.y, self.safety_distance):
                    valid_frontiers.append(frontier)
                    distances.append(distance)
                    self.get_logger().info(f'Valid frontier at ({frontier.x:.2f}, {frontier.y:.2f}), distance: {distance:.2f}m')

        if not valid_frontiers:
            return None

        # Select the frontier with the minimum distance
        min_index = np.argmin(distances)
        selected_frontier = valid_frontiers[min_index]
        self.get_logger().info(f'Selected frontier at ({selected_frontier.x:.2f}, {selected_frontier.y:.2f})')
        return selected_frontier

    def is_goal_safe(self, goal_x, goal_y, safety_distance=0.5):
        """
        Check if the goal position is safe by ensuring there are no obstacles within the safety distance.
        
        :param goal_x: Goal x position in meters
        :param goal_y: Goal y position in meters
        :param safety_distance: Safety distance in meters
        :return: True if safe, False otherwise
        """
        if self.map_info is None or self.map_data is None:
            return False

        # Calculate the number of cells corresponding to the safety distance
        num_cells = int(math.ceil(safety_distance / self.map_info.resolution))

        # Convert goal position to grid coordinates
        goal_grid_x = int((goal_x - self.map_info.origin.position.x) / self.map_info.resolution)
        goal_grid_y = int((goal_y - self.map_info.origin.position.y) / self.map_info.resolution)

        # Define the grid range to check
        min_x = max(goal_grid_x - num_cells, 0)
        max_x = min(goal_grid_x + num_cells, self.map_data.shape[1] - 1)
        min_y = max(goal_grid_y - num_cells, 0)
        max_y = min(goal_grid_y + num_cells, self.map_data.shape[0] - 1)

        # Check each cell within the safety distance
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if self.map_data[y, x] > 50:  # Threshold for obstacle, adjust as needed
                    return False
        return True

    def get_neighbors(self, x, y):
        # Get 8-connected neighbors
        neighbors = []
        width = self.map_data.shape[1]
        height = self.map_data.shape[0]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = x + dx
                ny = y + dy
                if (dx != 0 or dy != 0) and 0 <= nx < width and 0 <= ny < height:
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
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Could not get robot pose: {e}')
            return None

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Goal failed to reach server: {e}')
            self.exploring = False
            self.stop_robot()
            return

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self.exploring = False
            self.stop_robot()
            return

        self.get_logger().info('Goal accepted :)')
        self.current_goal = goal_handle

        # Cancel the goal timer since the goal was accepted
        if hasattr(self, 'goal_timer'):
            self.goal_timer.cancel()

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to get result: {e}')
            self.handle_goal_failure()
            return

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
            self.retry_count = 0  # Reset retry count on success
        else:
            self.get_logger().info(f'Goal failed with status: {status}')
            self.handle_goal_failure()

        # Look for the next frontier
        self.find_and_navigate_to_frontier()

    def handle_goal_failure(self):
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.get_logger().warn('Maximum retries reached. Stopping exploration.')
            self.exploring = False
            self.stop_robot()
        else:
            self.get_logger().info('Retrying with a new frontier.')
            self.find_and_navigate_to_frontier()

    def feedback_callback(self, feedback_msg):
        # Optional: Process feedback from the action server
        pass

    def goal_timeout_callback(self):
        if self.current_goal is not None:
            self.get_logger().warn('Goal timeout reached. Cancelling the current goal.')
            try:
                cancel_future = self.navigator.cancel_goal_async(self.current_goal)
                cancel_future.add_done_callback(self.cancel_goal_response_callback)
            except AttributeError as e:
                self.get_logger().error(f'Failed to cancel goal: {e}')

            # Reset variables
            self.current_goal = None
            self.exploring = False  # Allow retry
            self.stop_robot()

    def cancel_goal_response_callback(self, future):
        try:
            response = future.result()
            if len(response.goals_canceling) > 0:
                self.get_logger().info('Goal successfully cancelled.')
            else:
                self.get_logger().info('No goals were cancelled.')
        except Exception as e:
            self.get_logger().error(f'Failed to cancel goal: {e}')

    def stop_robot(self):
        # Publish zero velocities to stop the robot
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