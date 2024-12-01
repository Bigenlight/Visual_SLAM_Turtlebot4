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
from std_msgs.msg import Bool
from collections import deque
from slam_toolbox.srv import SaveMap  # Map saving service import
from std_msgs.msg import String

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Set logging level to DEBUG for detailed logs
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # Subscribe to /map topic
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # Initialize NavigateToPose action client
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize map saving service client
        self.save_map_client = self.create_client(SaveMap, '/slam_toolbox/save_map')

        # Initialize variables
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.exploring = False
        self.max_frontier_distance = 20.0  # Maximum exploration distance (meters)
        self.min_frontier_distance = 0.36  # Minimum goal distance (meters)
        self.safety_distance = 0.17  # Safety distance (meters)
        self.max_retries = 3  # Maximum goal retry attempts
        self.retry_count = 0
        self.goal_timeout = 30.0  # Goal timeout (seconds)

        # Movement monitoring variables
        self.last_moving_position = None
        self.last_moving_time = None
        self.movement_check_interval = 1.0  # Check every 1 second
        self.movement_threshold = 0.10  # 15 cm
        self.movement_timeout = 12.0  # 10 seconds of no movement triggers reset

        # Publisher to cmd_vel to stop the robot
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initialize TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait for action server to be ready
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('NavigateToPose action server is ready.')

        # Initialize lists to keep track of visited and failed frontiers
        self.visited_frontiers = []
        self.failed_frontiers = []  # List to record failed frontiers
        self.frontier_distance_threshold = 0.1  # Consider frontiers within 10 cm as visited

        # Current goal position
        self.current_goal_position = None

        # Publisher to signal cleaning algorithm
        self.cleaning_publisher = self.create_publisher(Bool, '/cleaning', 10)

 
    def map_callback(self, msg):
        """
        Callback function for /map topic.
        Updates map data and starts exploration if not already exploring.
        """
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        if not self.exploring:
            self.find_and_navigate_to_frontier()

    def find_and_navigate_to_frontier(self):
        """
        Detects and navigates to frontiers. If no frontiers remain,
        saves the map and shuts down.
        """
        if self.exploring:
            self.get_logger().debug('Already exploring. Skipping find_and_navigate_to_frontier.')
            return

        self.exploring = True  # Indicate that exploration is in progress

        self.get_logger().debug('Starting find_and_navigate_to_frontier.')

        # 1. Detect frontiers
        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info('No frontiers detected.')

            # Check if any frontiers remain in the entire map
            if not self.are_frontiers_remaining():
                self.get_logger().info('No frontiers remain in the map. Exploration complete.')
                self.stop_robot()
                self.shutdown()
            else:
                self.get_logger().info('Frontiers remain, but none are currently detectable.')
                self.get_logger().info('Retrying exploration...')
                self.exploring = False  # Allow another attempt
            return  # Exit the method without shutting down

        # 2. Cluster frontiers
        clustered_frontiers = self.cluster_frontiers(frontiers)

        if not clustered_frontiers:
            self.get_logger().info('No valid frontiers after clustering.')

            # Check if any frontiers remain in the entire map
            if not self.are_frontiers_remaining():
                self.get_logger().info('No frontiers remain in the map. Exploration complete.')
                self.stop_robot()
                self.shutdown()
            else:
                self.get_logger().info('Frontiers remain, but none are currently valid.')
                self.get_logger().info('Retrying exploration...')
                self.exploring = False  # Allow another attempt
            return  # Exit the method without shutting down

        # 3. Select a frontier
        goal_position = self.select_frontier(clustered_frontiers)

        if goal_position is None:
            self.get_logger().info('No valid frontiers found within distance constraints.')

            # Check if any frontiers remain in the entire map
            if not self.are_frontiers_remaining():
                self.get_logger().info('No frontiers remain in the map. Exploration complete.')
                self.stop_robot()
                self.shutdown()
            else:
                self.get_logger().info('Frontiers remain, but none are currently valid.')
                self.get_logger().info('Retrying exploration...')
                self.exploring = False  # Allow another attempt
            return  # Exit the method without shutting down

        # Proceed to send the goal
        self.current_goal_position = goal_position  # Save current goal position

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position = goal_position
        goal_msg.pose.pose.orientation.w = 1.0  # Forward-facing orientation

        self.get_logger().info(f'Navigating to frontier: ({goal_position.x:.2f}, {goal_position.y:.2f})')

        # Send goal
        send_goal_future = self.navigator.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

        # Movement monitoring starts upon goal acceptance

    def are_frontiers_remaining(self):
        """
        Checks if there are any frontiers (cells with values 0 or -1)
        larger than 10 cm in the entire map.
        """
        if self.map_data is None:
            self.get_logger().warn('Map data is not available.')
            return True  # Assume frontiers remain if map data is unavailable

        # Combine free space and unknown space
        frontier_mask = np.logical_or(self.map_data == 0, self.map_data == -1)

        # Label connected regions
        from scipy.ndimage import label

        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, num_features = label(frontier_mask, structure=structure)

        self.get_logger().info(f'Found {num_features} potential frontiers.')

        # Calculate the area (number of cells) of each connected region
        sizes = np.bincount(labeled.flatten())

        # The background (value 0) is not a frontier
        sizes[0] = 0

        # Convert 10 cm to number of cells
        min_cells = int(np.ceil(0.1 / self.map_info.resolution))

        # Check if any region is larger than min_cells
        for size in sizes:
            if size >= min_cells:
                self.get_logger().info(f'Frontier of size {size} cells found.')
                return True  # Frontiers remain

        self.get_logger().info('No frontiers larger than 10 cm remain.')
        return False  # No frontiers remain

    # Remove or comment out the is_map_explored method
    # def is_map_explored(self):
    #     """
    #     This method is no longer used.
    #     """
    #     pass

    # Rest of your code remains the same
    # ... (Include the rest of the methods as in your current code)

        

    def count_accessible_unknown_cells(self):
        """
        Calculates the number of accessible unknown cells using BFS from the robot's current position.
        """
        if self.map_data is None or self.map_info is None:
            return 0

        # Convert robot's current position to grid coordinates
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().warn('Unable to get robot position, cannot calculate accessible unknown cells.')
            return 0

        robot_grid_x = int((robot_pose.x - self.map_info.origin.position.x) / self.map_info.resolution)
        robot_grid_y = int((robot_pose.y - self.map_info.origin.position.y) / self.map_info.resolution)

        height, width = self.map_data.shape
        if not (0 <= robot_grid_x < width and 0 <= robot_grid_y < height):
            self.get_logger().warn('Robot grid position is out of map bounds.')
            return 0

        visited = np.zeros((height, width), dtype=bool)
        queue = deque()
        queue.append((robot_grid_x, robot_grid_y))
        visited[robot_grid_y, robot_grid_x] = True

        accessible_unknown = 0

        while queue:
            x, y = queue.popleft()

            neighbors = self.get_neighbors(x, y)
            for nx, ny in neighbors:
                if not visited[ny, nx]:
                    if self.map_data[ny, nx] == 0:  # Free space
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                    elif self.map_data[ny, nx] == -1:  # Unknown space
                        accessible_unknown += 1
                        visited[ny, nx] = True  # Mark as visited to prevent double-counting

        self.get_logger().info(f'Accessible unknown cells: {accessible_unknown}')
        return accessible_unknown

    def detect_frontiers(self):
        """
        Detects frontier points in the map.
        A frontier is a free cell adjacent to at least one unknown cell.
        """
        if self.map_data is None:
            self.get_logger().warn('Map data is not available.')
            return []

        height, width = self.map_data.shape
        frontier_points = []

        # Iterate through all free cells
        free_cells = np.argwhere(self.map_data == 0)

        for cell in free_cells:
            y, x = cell
            neighbors = self.get_neighbors(x, y)
            unknown_neighbors = 0
            for nx, ny in neighbors:
                if self.map_data[ny, nx] == -1:
                    unknown_neighbors += 1
            if unknown_neighbors >= 1:  # At least one unknown neighbor
                mx, my = self.grid_to_map(x, y)
                frontier_points.append([mx, my])

        self.get_logger().info(f'Detected {len(frontier_points)} frontier points.')
        return frontier_points

    def cluster_frontiers(self, frontiers):

        if not frontiers:
            self.get_logger().warn('No frontiers to cluster.')
            return []

        # DBSCAN parameters
        eps = 0.15  # 15 cm
        min_samples = 2

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(frontiers)
        labels = clustering.labels_

        unique_labels = set(labels)
        clustered_frontiers = []

        min_cluster_size = 2  # Minimum number of points to form a cluster

        for label in unique_labels:
            if label == -1:
                # Ignore noise points
                continue
            indices = np.where(labels == label)[0]
            cluster = np.array(frontiers)[indices]
            cluster_size = len(cluster)

            # Ignore small clusters
            if cluster_size < min_cluster_size:
                self.get_logger().debug(f'Ignoring small cluster with label {label} of size {cluster_size}')
                continue

            # Calculate centroid of the cluster
            centroid = np.mean(cluster, axis=0)
            point = Point(x=centroid[0], y=centroid[1], z=0.0)
            clustered_frontiers.append(point)

        self.get_logger().info(f'Clustered into {len(clustered_frontiers)} valid frontiers after filtering.')


        return clustered_frontiers

    def select_frontier(self, frontiers):
        """
        Selects the closest unvisited frontier based on the robot's current position.
        """
        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warning('Unable to get robot position, selecting first frontier.')
            return frontiers[0] if frontiers else None

        valid_frontiers = []
        distances = []
        for frontier in frontiers:
            # Check if the frontier has been visited or failed
            if self.is_frontier_visited(frontier) or self.is_frontier_failed(frontier):
                self.get_logger().debug(f'Frontier ({frontier.x:.2f}, {frontier.y:.2f}) has already been visited or failed.')
                continue

            dx = frontier.x - robot_position.x
            dy = frontier.y - robot_position.y
            distance = math.hypot(dx, dy)
            self.get_logger().debug(f'Frontier ({frontier.x:.2f}, {frontier.y:.2f}), Distance: {distance:.2f}m')

            if distance >= self.min_frontier_distance:
                valid_frontiers.append(frontier)
                distances.append(distance)
                self.get_logger().info(f'Valid frontier ({frontier.x:.2f}, {frontier.y:.2f}), Distance: {distance:.2f}m')
            else:
                self.get_logger().debug(f'Frontier ({frontier.x:.2f}, {frontier.y:.2f}) is too close.')

        if not valid_frontiers:
            self.get_logger().info('No valid frontiers found.')
            return None

        # Select the closest frontier
        min_index = np.argmin(distances)
        selected_frontier = valid_frontiers[min_index]
        self.visited_frontiers.append(selected_frontier)  # Mark as visited
        self.get_logger().info(f'Selected frontier ({selected_frontier.x:.2f}, {selected_frontier.y:.2f})')
        return selected_frontier

    def is_goal_safe(self, goal_x, goal_y, safety_distance=0.5):
        """
        Checks if the goal position is safe by ensuring no obstacles within the safety distance.
        """
        # This method is not used anymore since we removed obstacle checks.
        pass

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
        Retrieves the robot's current position in the 'odom' frame.
        """
        try:
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.get_logger().debug(f'Robot position: ({trans.transform.translation.x:.2f}, {trans.transform.translation.y:.2f})')
            return trans.transform.translation
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Unable to get robot position: {e}')
            return None

    def is_frontier_visited(self, frontier):
        """
        Checks if the frontier has already been visited or is being attempted.
        """
        for visited in self.visited_frontiers:
            dx = frontier.x - visited.x
            dy = frontier.y - visited.y
            distance = math.hypot(dx, dy)
            if distance < self.frontier_distance_threshold:
                return True
        return False

    def is_frontier_failed(self, frontier):
        """
        Checks if the frontier has been previously attempted and failed.
        """
        for failed in self.failed_frontiers:
            dx = frontier.x - failed.x
            dy = frontier.y - failed.y
            distance = math.hypot(dx, dy)
            if distance < self.frontier_distance_threshold:
                return True
        return False

    def goal_response_callback(self, future):
        """
        Callback function to handle the response from the action server after sending a goal.
        """
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to reach the goal server: {e}')
            self.exploring = False
            self.stop_robot()
            return

        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected by the action server.')
            self.exploring = False
            self.stop_robot()
            return

        self.get_logger().info('Goal accepted by the action server.')
        self.current_goal = goal_handle

        # Get the result asynchronously
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

        # Start movement monitoring
        self.start_movement_monitoring()

    def get_result_callback(self, future):
        """
        Callback function to handle the result from the action server after goal completion.
        """
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to get result of the goal: {e}')
            self.handle_goal_failure()
            return

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Reached the goal successfully!')
            self.retry_count = 0  # Reset retry count on success
        else:
            self.get_logger().info(f'Goal failed with status: {status}')
            self.handle_goal_failure()

        # Stop movement monitoring
        self.stop_movement_monitoring()

        # Reset exploration flag to allow the next exploration
        self.exploring = False

        # Proceed to find and navigate to the next frontier
        self.find_and_navigate_to_frontier()

    def handle_goal_failure(self):
        """
        Handles goal failure by retrying or moving to the next frontier.
        """
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.get_logger().warn('Maximum retry attempts reached. Recording frontier as failed and moving to next frontier.')
            if self.current_goal_position is not None:
                self.failed_frontiers.append(self.current_goal_position)  # Record as failed
            self.retry_count = 0  # Reset retry count
            self.current_goal_position = None  # Reset current goal position
            self.exploring = False
            self.stop_robot()
            # Attempt to navigate to the next frontier
            self.find_and_navigate_to_frontier()
        else:
            self.get_logger().info('Retrying the same frontier.')
            self.exploring = False  # Reset flag to allow retry
            self.find_and_navigate_to_frontier()

    def feedback_callback(self, feedback_msg):
        """
        Callback function to handle feedback from the action server.
        Currently unused but can be implemented as needed.
        """
        pass

    def cancel_current_goal(self):
        """
        Resets the current goal and exploration state.
        """
        self.get_logger().info('Resetting current goal and exploration state.')
        self.current_goal = None
        self.stop_robot()
        self.stop_movement_monitoring()
        self.last_moving_position = None
        self.last_moving_time = None
        self.exploring = False  # Allow find_and_navigate_to_frontier to proceed
        # Proceed to find and navigate to a new frontier
        self.find_and_navigate_to_frontier()

    def cancel_goal_response_callback(self, future):
        """
        Processes the response to the goal cancellation request.
        """
        try:
            response = future.result()
            if response.goals_canceling:
                self.get_logger().info('All active goals have been canceled.')
            else:
                self.get_logger().info('No active goals were found to cancel.')
        except Exception as e:
            self.get_logger().error(f'Failed to cancel goals: {e}')

        # Reset exploration flag and attempt to navigate to the next frontier
        self.exploring = False
        self.find_and_navigate_to_frontier()

    def shutdown(self):
        """
        Stops exploration and gracefully shuts down the node.
        """
        self.get_logger().info('Shutting down the exploration node.')

        try:
            # Call the map saving service
            self.save_map()
        except Exception as e:
            self.get_logger().error(f'Exception occurred while saving the map: {e}')

        # Publish signal to start the cleaning algorithm
        cleaning_msg = Bool()
        cleaning_msg.data = True
        self.cleaning_publisher.publish(cleaning_msg)
        self.get_logger().info('Published cleaning start signal.')

        # Cancel all active timers
        self.cancel_all_timers()

        # Stop the robot
        self.stop_robot()

        # Destroy the node and shutdown
        self.destroy_node()
        rclpy.shutdown()

    def save_map(self):
        """
        Calls the /slam_toolbox/save_map service to save the current map.
        """
        self.get_logger().info('Calling map saving service.')

        # Wait for the service to be available
        if not self.save_map_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/slam_toolbox/save_map service is not available.')
            return

        # Create service request
        request = SaveMap.Request()

        # Properly set the 'name' field as a std_msgs/String message
        request.name = String()
        request.name.data = 'map_test'

        # Call the service asynchronously
        future = self.save_map_client.call_async(request)

        # Wait for the service response
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('Map saved successfully.')
            else:
                self.get_logger().error(f"Failed to save map: {future.result().message}")
        else:
            self.get_logger().error('Map saving service call failed.')

    def cancel_all_timers(self):
        """
        Cancels all active timers to ensure clean shutdown.
        """
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.get_logger().debug('Canceled movement monitoring timer.')

        # Cancel other timers if any
        # e.g., self.another_timer.cancel()

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
        self.get_logger().info('Published stop command to the robot.')

    # Movement monitoring methods
    def start_movement_monitoring(self):
        """
        Starts a timer to monitor the robot's movement.
        """
        # Cancel existing timer if any
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.get_logger().debug('Canceled existing movement monitoring timer.')

        self.last_moving_position = self.get_robot_pose()
        self.last_moving_time = None  # Reset

        self.movement_timer = self.create_timer(self.movement_check_interval, self.check_movement_callback)
        self.get_logger().debug('Started movement monitoring timer.')

    def stop_movement_monitoring(self):
        """
        Stops the movement monitoring timer.
        """
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.movement_timer = None
            self.get_logger().debug('Stopped movement monitoring timer.')

    def check_movement_callback(self):
        """
        Periodically checks if the robot has moved sufficiently.
        Resets the goal and seeks a new frontier if not.
        """
        self.get_logger().debug('check_movement_callback called')
        current_time = self.get_clock().now()
        current_position = self.get_robot_pose()
        if current_position is None or self.last_moving_position is None:
            self.get_logger().debug('Cannot get current position or last moving position is None.')
            return  # Cannot retrieve position

        # Calculate distance moved since last check
        dx = current_position.x - self.last_moving_position.x
        dy = current_position.y - self.last_moving_position.y
        distance_moved = math.hypot(dx, dy)
        self.get_logger().debug(f'Distance moved: {distance_moved:.4f} meters.')

        if distance_moved >= self.movement_threshold:
            # Sufficient movement detected; update position and reset timer
            self.last_moving_position = current_position
            self.last_moving_time = None  # Reset
            self.get_logger().debug('The robot has moved sufficiently.')
        else:
            if self.last_moving_time is None:
                # Start the movement timeout timer
                self.last_moving_time = current_time
                self.get_logger().debug('The robot has not moved sufficiently. Starting movement timeout timer.')
                return

            # Calculate elapsed time since last movement
            time_since_last_move = (current_time - self.last_moving_time).nanoseconds / 1e9  # in seconds
            self.get_logger().debug(f'The robot has not moved sufficiently for {time_since_last_move:.2f} seconds.')

            if time_since_last_move >= self.movement_timeout:
                # Movement timeout exceeded; reset goal and seek new frontier
                self.get_logger().warn('The robot is stuck. Resetting goal and searching for next frontier.')
                self.cancel_current_goal()

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
