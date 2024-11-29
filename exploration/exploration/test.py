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
from nav_msgs.msg import MapMetaData


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
        self.last_map_data = None  # 이전 맵 데이터 저장

        # TF 리스너 설정
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 액션 서버 대기
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('Action server available.')

        # 파라미터 선언
        self.declare_parameter('safe_distance', 0.7)  # 안전 거리 증가
        self.safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value

        self.declare_parameter('recovery_behavior_enabled', True)
        self.recovery_behavior_enabled = self.get_parameter('recovery_behavior_enabled').get_parameter_value().bool_value

        self.declare_parameter('stuck_timeout', 30.0)  # 초 단위
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
        self.timer = self.create_timer(1.0, self.initialize_starting_pose)

    def initialize_starting_pose(self):
        self.starting_pose = self.get_robot_pose()
        if self.starting_pose is not None:
            self.get_logger().info(f'Starting position recorded at ({self.starting_pose.x:.2f}, {self.starting_pose.y:.2f})')
            self.destroy_timer(self.timer)
        else:
            self.get_logger().warning('Waiting for robot pose to initialize starting position.')

    def map_callback(self, msg):
        # OccupancyGrid 데이터를 NumPy 배열로 변환
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        # Distance Transform 계산 (장애물로부터의 거리 맵)
        free_space = self.map_data == 0
        obstacles = self.map_data == 100
        unknown = self.map_data == -1

        # 탐색되지 않은 영역이 남아있는지 확인
        if not np.any(unknown):
            if self.exploring:
                self.get_logger().info('Exploration complete! Returning to starting position...')
                self.return_to_start()
                self.exploring = False
                rclpy.shutdown()
            else:
                self.get_logger().info('Map is fully explored.')
            return

        if not self.exploring:
            self.exploring = True
            self.plan_and_execute_coverage()

    def plan_and_execute_coverage(self):
        # 미탐색 영역 (unknown) 주변의 자유 공간을 탐색
        unknown = self.map_data == -1
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated_unknown = scipy.ndimage.binary_dilation(unknown, structure=kernel)
        exploration_frontier = dilated_unknown & (self.map_data == 0)

        if not np.any(exploration_frontier):
            self.get_logger().info('No exploration frontier found.')
            self.exploring = False
            self.return_to_start()
            rclpy.shutdown()
            return

        # 탐색 전선 영역에서 웨이포인트 생성
        waypoints = self.generate_waypoints(exploration_frontier)
        if waypoints:
            self.follow_waypoints(waypoints)
        else:
            self.get_logger().info('No valid waypoints found.')
            self.exploring = False
            self.return_to_start()
            rclpy.shutdown()

    def generate_waypoints(self, frontier):
        # Distance Transform 계산
        distance_transform = scipy.ndimage.distance_transform_edt(self.map_data != 0) * self.map_info.resolution

        # 안전 거리 이상인 지점 선택
        safe_region = distance_transform >= self.safe_distance

        # 탐색 전선과 안전 영역의 교집합
        valid_points = frontier & safe_region

        # 유효한 포인트 인덱스 추출
        indices = np.argwhere(valid_points)
        if len(indices) == 0:
            self.get_logger().warning('No valid points found in frontier.')
            return []

        # Distance Transform 값에 기반하여 우선순위 설정
        valid_distances = distance_transform[valid_points]
        sorted_indices = indices[np.argsort(-valid_distances)]

        # 좌표 변환 (x, y)
        points = sorted_indices[:, [1, 0]].astype(float)  # x, y 순서로 변경
        points = points * self.map_info.resolution + np.array([self.map_info.origin.position.x, self.map_info.origin.position.y])

        # 클러스터링 적용
        clustering = DBSCAN(eps=self.waypoint_spacing, min_samples=1).fit(points)
        cluster_centers = []
        for cluster_label in np.unique(clustering.labels_):
            cluster_points = points[clustering.labels_ == cluster_label]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)

        # 웨이포인트 생성
        waypoints = []
        for center in cluster_centers:
            waypoint = Point(x=center[0], y=center[1], z=0.0)
            waypoints.append(waypoint)

        # 웨이포인트 시각화
        self.visualize_waypoints(waypoints)

        self.get_logger().info(f'Generated {len(waypoints)} waypoints for exploration.')
        return waypoints

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
                # 회복 후 재시도
                success = self.navigate_to_pose_with_timeout(goal_pose, timeout=self.stuck_timeout)
                if not success:
                    self.get_logger().warning('Failed to reach waypoint after recovery. Skipping to next waypoint.')
                    continue

    def navigate_to_pose_with_timeout(self, goal_pose, timeout=60.0):
        self.get_logger().info('Sending goal to the action server.')
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

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

        # 이동하지 않은 시간 측정을 위한 변수 초기화
        time_not_moved = 0.0
        last_movement_time = self.get_clock().now()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=1.0)
            status = goal_handle.status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Waypoint reached!')
                return True
            elif status in [GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED]:
                self.get_logger().warning(f'Failed to reach waypoint. Status: {status}')
                return False

            # 타임아웃 확인
            current_time = self.get_clock().now()
            if (current_time - start_time).nanoseconds > timeout * 1e9:
                self.get_logger().warning('Timeout reached while navigating to waypoint. Cancelling goal.')
                goal_handle.cancel_goal_async()
                return False

            # 로봇의 이동 거리 확인
            current_position = self.get_robot_pose()
            if current_position is not None and last_position is not None:
                distance_moved = math.hypot(
                    current_position.x - last_position.x,
                    current_position.y - last_position.y
                )
                if distance_moved < 0.01:  # 1cm 미만 이동 시
                    time_not_moved = (current_time - last_movement_time).nanoseconds / 1e9
                    if time_not_moved >= 1.5:
                        self.get_logger().warning('Robot has not moved for 1.5 seconds. Cancelling goal.')
                        goal_handle.cancel_goal_async()
                        return False
                else:
                    last_position = current_position
                    last_movement_time = current_time
                    time_not_moved = 0.0

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
            # 현재 방향 유지
            current_orientation = self.get_robot_orientation()
            if current_orientation is not None:
                pose.pose.orientation = current_orientation
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
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            return trans.transform.translation
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'Could not get robot pose: {e}')
            return None

    def get_robot_orientation(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            return trans.transform.rotation
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

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
