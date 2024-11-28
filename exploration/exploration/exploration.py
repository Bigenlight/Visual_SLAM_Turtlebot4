import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

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

        # Set up TF listener to get robot's current pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait for the action server to be ready
        self.navigator.wait_for_server()

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
            return

        # Select the closest frontier
        goal_position = self.select_frontier(frontiers)

        # Create and send a navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position = goal_position
        goal_msg.pose.pose.orientation.w = 1.0  # Facing forward

        self.get_logger().info(f'Navigating to frontier at ({goal_position.x}, {goal_position.y})')

        send_goal_future = self.navigator.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

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
        # Select the closest frontier to the robot's current position

        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warning('Could not get robot position. Selecting the first frontier.')
            return frontiers[0]

        # Compute distances
        distances = []
        for frontier in frontiers:
            dx = frontier.x - robot_position.x
            dy = frontier.y - robot_position.y
            distance = np.hypot(dx, dy)
            distances.append(distance)

        # Select the frontier with the minimum distance
        min_index = np.argmin(distances)
        return frontiers[min_index]

    def get_neighbors(self, x, y):
        # Get valid neighboring cells (4-connected grid)
        neighbors = []
        width = self.map_data.shape[1]
        height = self.map_data.shape[0]
        if x > 0:
            neighbors.append((x - 1, y))
        if x < width - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < height - 1:
            neighbors.append((x, y + 1))
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
            return

        self.get_logger().info('Goal accepted :)')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
        else:
            self.get_logger().info(f'Goal failed with status: {status}')

        # After reaching the goal, look for the next frontier
        self.find_and_navigate_to_frontier()

    def feedback_callback(self, feedback_msg):
        # Process feedback from the action server if needed
        pass

def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
