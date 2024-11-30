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
from visualization_msgs.msg import Marker, MarkerArray


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Set logging level to DEBUG for detailed logs
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
        self.max_frontier_distance = 20.0  # Maximum exploration distance in meters
        self.min_frontier_distance = 0.41  # Minimum goal distance in meters (tolerance)
        self.safety_distance = 0.1  # Safety distance in meters
        self.max_retries = 3  # Maximum number of goal retries
        self.retry_count = 0
        self.goal_timeout = 30.0  # Goal reach timeout in seconds

        # Movement monitoring variables
        self.last_moving_position = None
        self.last_moving_time = None
        self.movement_check_interval = 1.0  # Check every 1 second
        self.movement_threshold = 0.10  # 10 cm
        self.movement_timeout = 8.0  # seconds without movement

        # No-frontier timer variables
        self.no_frontier_timer = None
        self.no_frontier_duration = 60.0  # seconds without frontiers

        # Publisher to cmd_vel to stop the robot
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait for the action server to be ready
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('NavigateToPose action server ready.')

        # Initialize visited frontiers list
        self.visited_frontiers = []
        self.frontier_distance_threshold = 0.25  # 25 cm to consider a frontier as visited

        # Publisher for RViz markers
        self.marker_publisher = self.create_publisher(MarkerArray, 'frontier_markers', 10)

    def map_callback(self, msg):
        """
        Callback function for the /map topic.
        Updates the map data and initiates exploration if not already exploring.
        """
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        if not self.exploring:
            self.exploring = True
            self.find_and_navigate_to_frontier()

    def find_and_navigate_to_frontier(self):
        """
        Detects frontiers, clusters them, selects a valid frontier, and sends a navigation goal.
        """
        self.get_logger().debug('Starting find_and_navigate_to_frontier.')

        # Frontier detection logic
        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info('No frontiers detected.')
            # Start the no-frontier timer if not already started
            if self.no_frontier_timer is None:
                self.get_logger().info('Starting no-frontier timer.')
                self.no_frontier_timer = self.create_timer(self.no_frontier_duration, self.no_frontier_timer_callback)
            self.exploring = False  # Allow exploration to be re-triggered
            return  # Exit without stopping exploration yet

        # If frontiers are found and the no-frontier timer is running, cancel it
        if self.no_frontier_timer is not None:
            self.no_frontier_timer.cancel()
            self.no_frontier_timer = None
            self.get_logger().info('Frontiers detected. No-frontier timer cancelled.')

        # Cluster the frontiers
        clustered_frontiers = self.cluster_frontiers(frontiers)

        if not clustered_frontiers:
            self.get_logger().info('No valid frontiers after clustering.')
            # Start the no-frontier timer if not already started
            if self.no_frontier_timer is None:
                self.get_logger().info('Starting no-frontier timer.')
                self.no_frontier_timer = self.create_timer(self.no_frontier_duration, self.no_frontier_timer_callback)
            self.exploring = False  # Allow exploration to be re-triggered
            return  # Exit without stopping exploration yet

        # Select the closest valid frontier within min and max distance
        goal_position = self.select_frontier(clustered_frontiers)

        if goal_position is None:
            self.get_logger().info('No reachable and safe frontiers found within the specified distance range.')
            # Start the no-frontier timer if not already started
            if self.no_frontier_timer is None:
                self.get_logger().info('Starting no-frontier timer.')
                self.no_frontier_timer = self.create_timer(self.no_frontier_duration, self.no_frontier_timer_callback)
            self.exploring = False  # Allow exploration to be re-triggered
            return  # Exit without stopping exploration yet

        # If we have a goal, cancel the no-frontier timer if running
        if self.no_frontier_timer is not None:
            self.no_frontier_timer.cancel()
            self.no_frontier_timer = None
            self.get_logger().info('Valid frontier selected. No-frontier timer cancelled.')

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

        # Movement monitoring will start after the goal is accepted

    def no_frontier_timer_callback(self):
        """
        Callback function for the no-frontier timer.
        Stops exploration and shuts down the node after the specified duration without frontiers.
        """
        self.get_logger().info('The map is closed! No new frontiers detected for 40 seconds.')
        self.stop_robot()
        self.shutdown()

    def shutdown(self):
        """
        Stops exploration and shuts down the node gracefully.
        """
        self.get_logger().info('Shutting down the exploration node.')
        # Cancel any running timers
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.get_logger().debug('Movement monitoring timer cancelled during shutdown.')
        if self.no_frontier_timer is not None:
            self.no_frontier_timer.cancel()
            self.get_logger().debug('No-frontier timer cancelled during shutdown.')
        # Destroy the node
        self.destroy_node()
        rclpy.shutdown()

    def detect_frontiers(self):
        """
        Detects frontier points in the map.
        A frontier is a free cell adjacent to at least two unknown cells.
        """
        if self.map_data is None:
            self.get_logger().warn('Map data is not available.')
            return []

        height, width = self.map_data.shape
        frontier_points = []

        # Iterate over all free cells
        free_cells = np.argwhere(self.map_data == 0)

        for cell in free_cells:
            y, x = cell
            neighbors = self.get_neighbors(x, y)
            unknown_neighbors = 0
            for nx, ny in neighbors:
                if self.map_data[ny, nx] == -1:
                    unknown_neighbors += 1
            if unknown_neighbors >= 2:  # Require at least two unknown neighbors
                mx, my = self.grid_to_map(x, y)
                frontier_points.append([mx, my])

        self.get_logger().info(f'Detected {len(frontier_points)} frontier points.')
        return frontier_points

    def cluster_frontiers(self, frontiers):
        """
        Clusters frontier points using DBSCAN and filters out small clusters.
        Returns the centroids of valid clusters.
        Also publishes markers for visualization in RViz.
        """
        if not frontiers:
            self.get_logger().warn('No frontiers to cluster.')
            return []

        clustering = DBSCAN(eps=0.3, min_samples=5).fit(frontiers)  # min_samples increased to filter small clusters
        labels = clustering.labels_

        unique_labels = set(labels)
        clustered_frontiers = []

        min_cluster_size = 5  # Minimum number of points in a cluster

        for label in unique_labels:
            if label == -1:
                # Noise points; exclude them
                continue
            indices = np.where(labels == label)[0]
            cluster = np.array(frontiers)[indices]
            cluster_size = len(cluster)

            # Filter out clusters that are too small
            if cluster_size < min_cluster_size:
                self.get_logger().debug(f'Ignoring small cluster with label {label} of size {cluster_size}')
                continue

            # Compute the centroid of the cluster
            centroid = np.mean(cluster, axis=0)
            point = Point(x=centroid[0], y=centroid[1], z=0.0)
            clustered_frontiers.append(point)

        self.get_logger().info(f'Clustered into {len(clustered_frontiers)} valid frontiers after filtering.')

        # Publish markers for visualization
        self.publish_frontier_markers(clustered_frontiers)

        return clustered_frontiers

    def select_frontier(self, frontiers):
        """
        Selects the closest valid frontier that hasn't been visited and is safe.
        """
        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warning('Could not get robot position. Selecting the first frontier.')
            return frontiers[0] if frontiers else None

        valid_frontiers = []
        distances = []
        for frontier in frontiers:
            # Check if this frontier has been visited
            if self.is_frontier_visited(frontier):
                self.get_logger().debug(f'Frontier at ({frontier.x:.2f}, {frontier.y:.2f}) has already been visited.')
                continue

            dx = frontier.x - robot_position.x
            dy = frontier.y - robot_position.y
            distance = math.hypot(dx, dy)
            self.get_logger().debug(f'Frontier at ({frontier.x:.2f}, {frontier.y:.2f}), distance: {distance:.2f}m')

            if self.min_frontier_distance <= distance <= self.max_frontier_distance:
                # Check if the goal is safe
                is_safe = self.is_goal_safe(frontier.x, frontier.y, self.safety_distance)
                self.get_logger().debug(f'Frontier at ({frontier.x:.2f}, {frontier.y:.2f}) is within distance range and safety check result: {is_safe}')
                if is_safe:
                    valid_frontiers.append(frontier)
                    distances.append(distance)
                    self.get_logger().info(f'Valid frontier at ({frontier.x:.2f}, {frontier.y:.2f}), distance: {distance:.2f}m')
            else:
                self.get_logger().debug(f'Frontier at ({frontier.x:.2f}, {frontier.y:.2f}) is out of distance range.')

        if not valid_frontiers:
            self.get_logger().info('No valid frontiers found after filtering.')
            return None

        # Select the frontier with the minimum distance
        min_index = np.argmin(distances)
        selected_frontier = valid_frontiers[min_index]
        self.visited_frontiers.append(selected_frontier)  # Add to visited frontiers
        self.get_logger().info(f'Selected frontier at ({selected_frontier.x:.2f}, {selected_frontier.y:.2f})')
        return selected_frontier

    def is_goal_safe(self, goal_x, goal_y, safety_distance=0.5):
        """
        Checks if the goal position is safe by ensuring there are no obstacles within the safety distance.
        """
        if self.map_info is None or self.map_data is None:
            self.get_logger().warn('Map information or data is not available for safety check.')
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
                cell_value = self.map_data[y, x]
                if cell_value > 50:  # Threshold for obstacle, adjust as needed
                    self.get_logger().debug(f'Obstacle found at grid ({x}, {y}) with value {cell_value}')
                    return False
        self.get_logger().debug(f'Goal at ({goal_x:.2f}, {goal_y:.2f}) is safe.')
        return True

    def get_neighbors(self, x, y):
        """
        Returns the 8-connected neighbors of a given cell.
        """
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
        """
        Converts grid coordinates to map coordinates.
        """
        mx = self.map_info.origin.position.x + (x + 0.5) * self.map_info.resolution
        my = self.map_info.origin.position.y + (y + 0.5) * self.map_info.resolution
        return mx, my

    def get_robot_pose(self):
        """
        Retrieves the robot's current pose in the map frame.
        """
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.get_logger().debug(f'Robot position: ({trans.transform.translation.x:.2f}, {trans.transform.translation.y:.2f})')
            return trans.transform.translation
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Could not get robot pose: {e}')
            return None

    def is_frontier_visited(self, frontier):
        """
        Checks if a frontier has already been visited based on the distance threshold.
        """
        for visited in self.visited_frontiers:
            dx = frontier.x - visited.x
            dy = frontier.y - visited.y
            distance = math.hypot(dx, dy)
            if distance < self.frontier_distance_threshold:
                return True
        return False

    def goal_response_callback(self, future):
        """
        Callback for handling the response from the action server after sending a goal.
        """
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

        # Start movement monitoring only after the goal is accepted
        self.start_movement_monitoring()

        # Set up the result callback
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Callback for handling the result from the action server after the goal is processed.
        """
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

        # Stop the movement monitoring timer
        self.stop_movement_monitoring()

        # Reset exploring flag to allow new explorations
        self.exploring = False

        # Look for the next frontier
        self.find_and_navigate_to_frontier()

    def handle_goal_failure(self):
        """
        Handles goal failure by retrying or stopping exploration after maximum retries.
        """
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.get_logger().warn('Maximum retries reached. Stopping exploration.')
            self.exploring = False
            self.stop_robot()
        else:
            self.get_logger().info('Retrying with a new frontier.')
            self.exploring = False  # Allow new exploration
            self.find_and_navigate_to_frontier()

    def feedback_callback(self, feedback_msg):
        """
        Callback for processing feedback from the action server.
        Currently unused but can be implemented as needed.
        """
        pass

    def goal_timeout_callback(self):
        """
        Callback for handling goal timeouts. Cancels the current goal if timeout is reached.
        """
        if self.current_goal is not None:
            self.get_logger().warn('Goal timeout reached. Cancelling the current goal.')
            self.cancel_current_goal()
            # Start looking for a new frontier
            self.find_and_navigate_to_frontier()

    def cancel_goal_response_callback(self, future):
        """
        Callback for handling the response from the action server after cancelling a goal.
        """
        try:
            response = future.result()
            if len(response.goals_canceling) > 0:
                self.get_logger().info('Goal successfully cancelled.')
            else:
                self.get_logger().info('No goals were cancelled.')
        except Exception as e:
            self.get_logger().error(f'Failed to cancel goal: {e}')
            return  # Exit if cancellation failed

        # Start looking for a new frontier after the goal is cancelled
        self.find_and_navigate_to_frontier()

    def cancel_current_goal(self):
        """
        Cancels the current navigation goal and resets exploration state.
        """
        if self.current_goal is not None:
            try:
                cancel_future = self.navigator.cancel_goal_async(self.current_goal)
                cancel_future.add_done_callback(self.cancel_goal_response_callback)
            except Exception as e:
                self.get_logger().error(f'Failed to cancel goal: {e}')
            self.current_goal = None
            self.stop_robot()
            self.stop_movement_monitoring()
            # Reset movement monitoring variables
            self.last_moving_position = None
            self.last_moving_time = None

    def stop_robot(self):
        """
        Publishes zero velocities to stop the robot.
        """
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_msg)
        self.get_logger().info('Sent stop command to the robot.')

    # Movement monitoring methods
    def start_movement_monitoring(self):
        """
        Initializes movement monitoring by recording the current position and starting a timer.
        Resets movement monitoring variables to prevent immediate retriggering.
        """
        # Cancel any existing movement monitoring timer
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.get_logger().debug('Existing movement monitoring timer cancelled.')

        self.last_moving_position = self.get_robot_pose()
        self.last_moving_time = None  # Will be set in check_movement_callback
        self.movement_timer = self.create_timer(self.movement_check_interval, self.check_movement_callback)
        self.get_logger().debug('Started movement monitoring.')

    def stop_movement_monitoring(self):
        """
        Stops the movement monitoring timer.
        """
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.movement_timer = None
            self.get_logger().debug('Stopped movement monitoring.')

    def check_movement_callback(self):
        """
        Periodically checks if the robot has moved significantly. If not, considers the robot stuck.
        """
        current_time = self.get_clock().now()
        current_position = self.get_robot_pose()
        if current_position is None or self.last_moving_position is None:
            return  # Can't get position

        # Compute distance moved since last moving position
        dx = current_position.x - self.last_moving_position.x
        dy = current_position.y - self.last_moving_position.y
        distance_moved = math.hypot(dx, dy)

        if distance_moved >= self.movement_threshold:
            # Robot has moved more than threshold, update last moving position and reset last_moving_time
            self.last_moving_position = current_position
            self.last_moving_time = None  # Reset last_moving_time
            self.get_logger().debug('Robot has moved significantly.')
        else:
            if self.last_moving_time is None:
                # Robot hasn't moved, start timing
                self.last_moving_time = current_time
                self.get_logger().debug('Robot hasn\'t moved significantly. Starting movement timeout timer.')
                return

            # Calculate time since last significant movement
            time_since_last_move = (current_time - self.last_moving_time).nanoseconds / 1e9  # seconds
            self.get_logger().debug(f'Robot has not moved significantly for {time_since_last_move:.2f} seconds.')

            if time_since_last_move >= self.movement_timeout:
                # Robot hasn't moved threshold distance in movement_timeout seconds, consider stuck
                self.get_logger().warn('Robot is stuck, cancelling goal.')
                self.stop_movement_monitoring()
                self.cancel_current_goal()
                # Exploration will be re-triggered in cancel_goal_response_callback()

    # Visualization methods
    def publish_frontier_markers(self, frontiers):
        """
        Publishes frontier markers for visualization in RViz.
        """
        marker_array = MarkerArray()
        for idx, frontier in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontiers'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = frontier
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)
        self.get_logger().debug('Published frontier markers for RViz.')


def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()

    # Use a MultiThreadedExecutor to allow timers and callbacks to run in parallel
    executor = rclpy.executors.MultiThreadedExecutor()
    try:
        rclpy.spin(explorer, executor=executor)
    except KeyboardInterrupt:
        explorer.get_logger().info('Keyboard interrupt, shutting down.')
        explorer.shutdown()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
