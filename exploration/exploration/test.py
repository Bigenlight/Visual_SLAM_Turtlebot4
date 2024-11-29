import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener
import math
import tf2_ros
from rclpy.duration import Duration
from skimage import measure
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Path

class BoustrophedonExplorer(Node):
    def __init__(self):
        super().__init__('boustrophedon_explorer')
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.exploring = False

        # Set up TF listener to get robot's current pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait for the action server to be ready
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('Action server available.')

        # Create a service client for path planning
        self.get_plan_client = self.create_client(GetPlan, 'planner_server/get_plan')
        while not self.get_plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for GetPlan service...')
        self.get_logger().info('GetPlan service available.')

        # Parameters
        self.declare_parameter('safe_distance', 0.3)  # meters
        self.safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value

        self.declare_parameter('recovery_behavior_enabled', True)
        self.recovery_behavior_enabled = self.get_parameter('recovery_behavior_enabled').get_parameter_value().bool_value

    def map_callback(self, msg):
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        if not self.exploring:
            self.exploring = True
            self.plan_and_execute_coverage()

    def plan_and_execute_coverage(self):
        # Boustrophedon decomposition and path planning
        free_space = self.map_data == 0  # Free cells
        labeled_map, num_features = measure.label(free_space, connectivity=1, return_num=True)

        if num_features == 0:
            self.get_logger().info('No free space to explore.')
            self.exploring = False
            rclpy.shutdown()
            return

        self.get_logger().info(f'Number of regions to explore: {num_features}')

        # Generate coverage path for each region
        for region_label in range(1, num_features + 1):
            region = labeled_map == region_label
            waypoints = self.generate_boustrophedon_path(region)
            if waypoints:
                self.follow_waypoints(waypoints)

        self.get_logger().info('Coverage complete!')
        self.exploring = False
        rclpy.shutdown()

    def generate_boustrophedon_path(self, region):
        # Get the indices of the free cells in the region
        indices = np.argwhere(region)
        if len(indices) == 0:
            return []

        # Extract the bounding rectangle of the region
        min_y, min_x = indices.min(axis=0)
        max_y, max_x = indices.max(axis=0)

        waypoints = []
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        # Generate boustrophedon path within the bounding rectangle
        for y in range(min_y, max_y + 1):
            row_indices = indices[indices[:, 0] == y]
            if len(row_indices) == 0:
                continue
            x_values = row_indices[:, 1]
            x_start = x_values.min()
            x_end = x_values.max()

            # Depending on the row, alternate the direction
            if (y - min_y) % 2 == 0:
                x_range = range(x_start, x_end + 1)
            else:
                x_range = range(x_end, x_start - 1, -1)

            for x in x_range:
                if region[y, x]:
                    mx = origin_x + (x + 0.5) * resolution
                    my = origin_y + (y + 0.5) * resolution

                    # Check if the waypoint is at a safe distance from obstacles
                    if self.is_safe_point(x, y):
                        waypoint = Point(x=mx, y=my, z=0.0)
                        waypoints.append(waypoint)
                    else:
                        self.get_logger().debug(f'Waypoint at ({mx:.2f}, {my:.2f}) is too close to an obstacle.')

        self.get_logger().info(f'Generated {len(waypoints)} waypoints for the region.')
        return waypoints

    def is_safe_point(self, x, y):
        # Check surrounding cells to ensure the point is at a safe distance from obstacles
        safe_distance_cells = int(self.safe_distance / self.map_info.resolution)
        for dy in range(-safe_distance_cells, safe_distance_cells + 1):
            for dx in range(-safe_distance_cells, safe_distance_cells + 1):
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= self.map_data.shape[1] or ny < 0 or ny >= self.map_data.shape[0]:
                    return False  # Out of bounds
                if self.map_data[ny, nx] == 100:
                    return False  # Obstacle nearby
        return True

    def follow_waypoints(self, waypoints):
        for idx, waypoint in enumerate(waypoints):
            next_waypoint = waypoints[idx + 1] if idx + 1 < len(waypoints) else None
            goal_pose = self.create_pose_stamped(waypoint, next_position=next_waypoint)
            self.get_logger().info(f'Navigating to waypoint at ({waypoint.x:.2f}, {waypoint.y:.2f})')

            # Check if the robot can plan to the waypoint
            if not self.can_plan_to_goal(goal_pose):
                self.get_logger().warning(f'Cannot plan to waypoint at ({waypoint.x:.2f}, {waypoint.y:.2f}), skipping.')
                continue

            # Attempt to reach the waypoint with retries
            success = self.navigate_to_pose_with_retries(goal_pose, retries=3)
            if not success and self.recovery_behavior_enabled:
                self.perform_recovery_behavior()

    def navigate_to_pose_with_retries(self, goal_pose, retries=3):
        attempt = 0
        while attempt < retries:
            attempt += 1
            self.get_logger().info(f'Attempt {attempt} to navigate to waypoint.')
            send_goal_future = self.navigator.send_goal_async(NavigateToPose.Goal(pose=goal_pose), feedback_callback=self.feedback_callback)
            send_goal_future.add_done_callback(self.goal_response_callback)

            # Wait for the result with a timeout
            result_future = self.result_future
            try:
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
                status = result_future.result().status
                if status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info('Waypoint reached!')
                    return True
                else:
                    self.get_logger().warning(f'Failed to reach waypoint. Status: {status}')
            except Exception as e:
                self.get_logger().error(f'Exception while waiting for goal result: {e}')

        self.get_logger().error('Failed to reach waypoint after multiple attempts.')
        return False

    def perform_recovery_behavior(self):
        self.get_logger().info('Performing recovery behavior.')
        # Example recovery behavior: rotate in place
        self.rotate_in_place()

    def rotate_in_place(self):
        # Implement rotation in place
        # This is a simple example using a Twist publisher
        from geometry_msgs.msg import Twist
        cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        twist = Twist()
        twist.angular.z = 0.5  # Rotate at 0.5 rad/s
        duration = 5.0  # Rotate for 5 seconds
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        # Stop rotation
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        self.get_logger().info('Recovery behavior completed.')

    def can_plan_to_goal(self, goal_pose):
        # Use the GetPlan service to check if a path exists to the goal
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().warning('Cannot get robot pose.')
            return False

        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.header.stamp = self.get_clock().now().to_msg()
        start_pose.pose.position = robot_pose
        start_pose.pose.orientation.w = 1.0

        req = GetPlan.Request()
        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = 0.5  # meters

        future = self.get_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            path = future.result().plan
            if len(path.poses) > 0:
                return True
            else:
                return False
        else:
            self.get_logger().warning('Failed to call GetPlan service.')
            return False

    def create_pose_stamped(self, position, next_position=None):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = position

        if next_position is not None:
            dx = next_position.x - position.x
            dy = next_position.y - position.y
            yaw = math.atan2(dy, dx)
            quaternion = self.euler_to_quaternion(0, 0, yaw)
            pose.pose.orientation = quaternion
        else:
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        return pose

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - \
             math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + \
             math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - \
             math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + \
             math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                now,
                timeout=Duration(seconds=1.0))
            return trans.transform.translation
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'Could not get robot pose: {e}')
            return None

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info('Goal rejected :(')
                self.result_future = None
                return

            self.get_logger().info('Goal accepted :)')
            self.result_future = goal_handle.get_result_async()
        except Exception as e:
            self.get_logger().error(f'Exception in goal_response_callback: {e}')
            self.result_future = None

    def feedback_callback(self, feedback_msg):
        # Optionally process feedback here
        pass

def main(args=None):
    rclpy.init(args=args)
    explorer = BoustrophedonExplorer()
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        pass
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()