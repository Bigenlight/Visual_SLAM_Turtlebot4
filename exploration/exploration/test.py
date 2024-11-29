import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener
import math
import tf2_ros
from rclpy.duration import Duration
from skimage import measure
import scipy.ndimage
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray


class BoustrophedonExplorer(Node):
    def __init__(self):
        super().__init__('boustrophedon_explorer')

        # 맵 구독자 설정
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # NavigateToPose 액션 클라이언트 설정
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 초기 변수 설정
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.exploring = False
        self.starting_pose = None  # 시작 위치 저장

        # TF 리스너 설정
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 액션 서버 대기
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('Action server available.')

        # 파라미터 선언
        self.declare_parameter('safe_distance', 0.5)  # 미터 단위
        self.safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value

        self.declare_parameter('recovery_behavior_enabled', True)
        self.recovery_behavior_enabled = self.get_parameter('recovery_behavior_enabled').get_parameter_value().bool_value

        self.declare_parameter('stuck_timeout', 10.0)  # 초 단위
        self.stuck_timeout = self.get_parameter('stuck_timeout').get_parameter_value().double_value

        self.declare_parameter('waypoint_spacing', 0.5)  # 미터 단위
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').get_parameter_value().double_value

        self.declare_parameter('robot_width', 0.35)  # 로봇의 폭 (미터 단위)
        self.robot_width = self.get_parameter('robot_width').get_parameter_value().double_value

        # cmd_vel 퍼블리셔 설정 (회복 행동용)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 웨이포인트 마커 퍼블리셔 설정
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints_markers', 10)

        # 시작 위치 저장
        self.starting_pose = self.get_robot_pose()
        if self.starting_pose is None:
            self.get_logger().error('Could not get starting position.')
        else:
            self.get_logger().info(f'Starting position recorded at ({self.starting_pose.x:.2f}, {self.starting_pose.y:.2f})')

    def map_callback(self, msg):
        # OccupancyGrid 데이터를 NumPy 배열로 변환
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        # Distance Transform 계산 (장애물로부터의 거리 맵)
        free_space = self.map_data == 0
        obstacles = self.map_data == 100
        self.distance_transform = scipy.ndimage.distance_transform_edt(free_space) * self.map_info.resolution

        if not self.exploring:
            self.exploring = True
            self.plan_and_execute_coverage()

    def plan_and_execute_coverage(self):
        # Boustrophedon 분할 및 경로 계획
        free_space = self.map_data == 0  # 자유 공간
        labeled_map, num_features = measure.label(free_space, connectivity=1, return_num=True)

        if num_features == 0:
            self.get_logger().info('No free space to explore.')
            self.exploring = False
            self.return_to_start()
            return

        self.get_logger().info(f'Number of regions to explore: {num_features}')

        # 각 영역에 대한 웨이포인트 생성 및 탐색
        for region_label in range(1, num_features + 1):
            region = labeled_map == region_label
            waypoints = self.generate_boustrophedon_path(region)
            if waypoints:
                self.follow_waypoints(waypoints)

        self.get_logger().info('Coverage complete! Returning to starting position...')
        self.return_to_start()
        self.exploring = False
        rclpy.shutdown()

    def generate_boustrophedon_path(self, region):
        # Distance Transform 계산
        distance_transform = scipy.ndimage.distance_transform_edt(region) * self.map_info.resolution

        # 중심선 추출: 최대 거리의 80% 이상인 지점 선택
        max_distance = distance_transform.max()
        if max_distance == 0:
            self.get_logger().warning('Max distance in region is zero. Skipping region.')
            return []

        threshold = max_distance * 0.8
        centerline = distance_transform >= threshold

        # 중심선을 따라 웨이포인트 생성
        indices = np.argwhere(centerline)
        if len(indices) == 0:
            self.get_logger().warning('No centerline found for the region.')
            return []

        # 좌표 변환 (x, y)
        points = indices[:, [1, 0]].astype(float)  # x, y 순서로 변경
        points = points * self.map_info.resolution + np.array([self.map_info.origin.position.x, self.map_info.origin.position.y])

        # DBSCAN 클러스터링 적용
        clustering = DBSCAN(eps=self.waypoint_spacing, min_samples=1).fit(points)
        cluster_centers = []
        for cluster_label in np.unique(clustering.labels_):
            cluster_points = points[clustering.labels_ == cluster_label]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)

        # 웨이포인트 생성
        waypoints = []
        for center in cluster_centers:
            if self.is_safe_point_world(center[0], center[1]):
                waypoint = Point(x=center[0], y=center[1], z=0.0)
                waypoints.append(waypoint)
            else:
                self.get_logger().debug(f'Waypoint at ({center[0]:.2f}, {center[1]:.2f}) is too close to an obstacle.')

        # 웨이포인트 시각화
        self.visualize_waypoints(waypoints)

        self.get_logger().info(f'Generated {len(waypoints)} waypoints for the region.')
        return waypoints

    def is_safe_point_world(self, mx, my):
        # 월드 좌표를 맵 좌표로 변환
        x = int((mx - self.map_info.origin.position.x) / self.map_info.resolution)
        y = int((my - self.map_info.origin.position.y) / self.map_info.resolution)

        if x < 0 or x >= self.map_data.shape[1] or y < 0 or y >= self.map_data.shape[0]:
            return False  # 맵 외부

        distance = self.distance_transform[y, x]
        return distance >= self.safe_distance

    def visualize_waypoints(self, waypoints):
        marker_array = MarkerArray()
        for idx, wp in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'waypoints'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = wp
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def follow_waypoints(self, waypoints):
        for idx, waypoint in enumerate(waypoints):
            goal_pose = self.create_pose_stamped(waypoint)
            self.get_logger().info(f'Navigating to waypoint at ({waypoint.x:.2f}, {waypoint.y:.2f})')

            # 타임아웃과 회복 행동을 포함한 목표 이동 시도
            success = self.navigate_to_pose_with_timeout(goal_pose, timeout=self.stuck_timeout)
            if not success and self.recovery_behavior_enabled:
                self.perform_recovery_behavior()

    def navigate_to_pose_with_timeout(self, goal_pose, timeout=60.0):
        self.get_logger().info('Sending goal to the action server.')
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose.pose

        send_goal_future = self.navigator.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Goal rejected by the action server.')
            return False

        self.get_logger().info('Goal accepted. Monitoring progress...')
        start_time = self.get_clock().now()
        last_position = self.get_robot_pose()
        if last_position is None:
            self.get_logger().warning('Cannot get initial robot position. Proceeding without movement check.')

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=1.0)
            status = goal_handle.goal_status.status
            if status in [GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_REJECTED, GoalStatus.STATUS_CANCELLED]:
                if status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info('Waypoint reached!')
                    return True
                else:
                    self.get_logger().warning(f'Failed to reach waypoint. Status: {status}')
                    return False

            # 타임아웃 확인
            current_time = self.get_clock().now()
            if (current_time - start_time).nanoseconds > timeout * 1e9:
                self.get_logger().warning('Timeout reached while navigating to waypoint. Cancelling goal.')
                goal_handle.cancel_goal_async()
                return False

            # 로봇의 이동 거리 확인 (움직임이 없으면 멈춘 것으로 간주)
            current_position = self.get_robot_pose()
            if current_position is not None and last_position is not None:
                distance_moved = math.hypot(
                    current_position.x - last_position.x,
                    current_position.y - last_position.y
                )
                if distance_moved < 0.01:  # 1cm 미만 이동 시
                    self.get_logger().warning('Robot seems to be stuck. Cancelling goal.')
                    goal_handle.cancel_goal_async()
                    return False
                else:
                    last_position = current_position

        return False

    def perform_recovery_behavior(self):
        self.get_logger().info('Performing recovery behavior.')
        self.rotate_in_place()
        self.get_logger().info('Recovery behavior completed.')

    def rotate_in_place(self):
        # 제자리에서 회전하여 장애물 회피 시도
        twist = Twist()
        twist.angular.z = 0.5  # 시계방향으로 0.5 rad/s 회전
        duration = 5.0  # 5초 동안 회전
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        # 회전 정지
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def return_to_start(self):
        if self.starting_pose is None:
            self.get_logger().error('Starting position is unknown. Cannot return to start.')
            return

        starting_goal_pose = self.create_pose_stamped(self.starting_pose)
        self.get_logger().info(f'Navigating back to starting position at ({self.starting_pose.x:.2f}, {self.starting_pose.y:.2f})')
        success = self.navigate_to_pose_with_timeout(starting_goal_pose, timeout=60.0)
        if success:
            self.get_logger().info('Returned to starting position successfully!')
        else:
            self.get_logger().warning('Failed to return to starting position.')

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
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - \
             math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + \
             math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - \
             math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + \
             math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
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

    def distance_between(self, p1, p2):
        return math.hypot(p2.x - p1.x, p2.y - p1.y)

    def feedback_callback(self, feedback_msg):
        # 피드백 처리 가능 (현재는 생략)
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
