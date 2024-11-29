import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_ros  # tf2_ros 모듈 import 추가
import math
from rclpy.duration import Duration
import scipy.ndimage
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
import os
import json
import datetime


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # 맵 구독자 설정
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # NavigateToPose 액션 클라이언트 설정
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 초기 변수 설정
        self.map_data = None
        self.map_info = None
        self.exploring = False
        self.waypoints = []
        self.current_waypoint_index = 0
        self.goal_handle = None

        # TF 리스너 설정
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 액션 서버 대기
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self.navigator.wait_for_server()
        self.get_logger().info('Action server available.')

        # 파라미터 선언 및 초기화
        self.declare_parameter('safe_distance', 0.2)  # 안전 거리 조정
        self.safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value

        self.declare_parameter('recovery_behavior_enabled', True)
        self.recovery_behavior_enabled = self.get_parameter('recovery_behavior_enabled').get_parameter_value().bool_value

        self.declare_parameter('stuck_timeout', 4.0)  # 4초 동안 움직이지 않으면 스턱으로 간주
        self.stuck_timeout = self.get_parameter('stuck_timeout').get_parameter_value().double_value

        self.declare_parameter('waypoint_spacing', 0.3)  # 웨이포인트 간격을 0.3미터로 설정
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').get_parameter_value().double_value

        self.declare_parameter('robot_width', 0.35)
        self.robot_width = self.get_parameter('robot_width').get_parameter_value().double_value

        self.declare_parameter('map_save_directory', '/tmp/maps')  # 맵 저장 디렉토리
        self.map_save_directory = self.get_parameter('map_save_directory').get_parameter_value().string_value
        os.makedirs(self.map_save_directory, exist_ok=True)

        # cmd_vel 퍼블리셔 설정 (회복 행동용)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 웨이포인트 마커 퍼블리셔 설정
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints_markers', 10)

        # 현재 목표 상태 추적
        self.current_goal_active = False

        # 로봇의 마지막 위치 및 시간 추적
        self.last_pose = None
        self.last_movement_time = self.get_clock().now()
        self.movement_threshold = 0.1  # 미터 단위
        self.stuck = False

        # 로봇 이동 모니터링을 위한 타이머 설정 (1초 간격)
        self.timer = self.create_timer(1.0, self.monitor_robot_movement)

    def map_callback(self, msg):
        # OccupancyGrid 데이터를 NumPy 배열로 변환
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        # 맵 데이터를 파일로 저장
        self.save_map_to_file(msg)

        # 탐색되지 않은 영역이 남아있는지 확인
        unknown = self.map_data == -1
        if not np.any(unknown):
            if self.exploring:
                self.get_logger().info('Exploration complete! All areas have been explored.')
                self.exploring = False
                self.shutdown_node()
            else:
                self.get_logger().info('Map is fully explored.')
            return

        if not self.exploring and not self.current_goal_active:
            self.exploring = True
            self.plan_and_execute_exploration()

    def plan_and_execute_exploration(self):
        # 탐색 전선 검출
        frontiers = self.detect_frontiers()
        if frontiers is None or frontiers.size == 0:
            self.get_logger().info('No frontiers detected. Waiting for map updates...')
            self.exploring = False  # 다음 맵 업데이트에서 다시 탐색 시도
            return

        # 웨이포인트 생성
        waypoints = self.generate_waypoints(frontiers)
        if waypoints:
            self.waypoints = waypoints
            self.current_waypoint_index = 0
            self.send_next_waypoint()
        else:
            self.get_logger().info('No valid waypoints found. Waiting for map updates...')
            self.exploring = False
            return

    def detect_frontiers(self):
        # 맵 데이터에서 탐색 전선 검출
        occupied = self.map_data == 100
        free = self.map_data == 0
        unknown = self.map_data == -1

        # 알려진 공간과 미탐색 공간의 경계 찾기
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = scipy.ndimage.binary_dilation(unknown, structure=kernel)
        border = unknown_dilated & free

        # 탐색 전선의 좌표 추출
        frontiers = np.argwhere(border)
        self.get_logger().info(f'Number of frontier points detected: {len(frontiers)}')

        return frontiers

    def generate_waypoints(self, frontiers):
        # 좌표 변환 (x, y)
        points = frontiers[:, [1, 0]].astype(float)  # x, y 순서로 변경
        points = points * self.map_info.resolution + np.array([self.map_info.origin.position.x, self.map_info.origin.position.y])

        if points.size == 0:
            self.get_logger().warning('No points available for clustering.')
            return []

        # 클러스터링 적용
        clustering = DBSCAN(eps=self.waypoint_spacing, min_samples=3).fit(points)
        cluster_centers = []
        for cluster_label in np.unique(clustering.labels_):
            if cluster_label == -1:
                continue  # 노이즈 제거
            cluster_points = points[clustering.labels_ == cluster_label]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)

        if not cluster_centers:
            self.get_logger().warning('No clusters found for frontiers.')
            return []

        # 웨이포인트 생성
        waypoints = []
        for center in cluster_centers:
            if self.is_safe_point(center[0], center[1]):
                waypoint = Point(x=center[0], y=center[1], z=0.0)
                waypoints.append(waypoint)
            else:
                self.get_logger().debug(f'Waypoint at ({center[0]:.2f}, {center[1]:.2f}) is not safe.')

        # 웨이포인트 시각화
        self.visualize_waypoints(waypoints)

        self.get_logger().info(f'Generated {len(waypoints)} waypoints for exploration.')
        return waypoints

    def is_safe_point(self, x_world, y_world):
        # 월드 좌표를 맵 인덱스로 변환
        x_idx = int((x_world - self.map_info.origin.position.x) / self.map_info.resolution)
        y_idx = int((y_world - self.map_info.origin.position.y) / self.map_info.resolution)

        # 맵 인덱스가 유효한지 확인
        if x_idx < 0 or x_idx >= self.map_info.width or y_idx < 0 or y_idx >= self.map_info.height:
            return False

        # 해당 위치가 자유 공간인지 확인
        if self.map_data[y_idx, x_idx] != 0:
            return False

        # 거리 변환(Distance Transform)을 이용하여 주변 장애물과의 거리 확인
        occupied = self.map_data == 100
        distance_map = scipy.ndimage.distance_transform_edt(~occupied) * self.map_info.resolution
        distance = distance_map[y_idx, x_idx]

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

    def send_next_waypoint(self):
        if self.current_waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_index]
            goal_pose = self.create_pose_stamped(waypoint)
            self.get_logger().info(f'Sending goal to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} at ({waypoint.x:.2f}, {waypoint.y:.2f})')
            self.send_goal(goal_pose)
        else:
            self.get_logger().info('All current waypoints have been processed. Checking for new frontiers...')
            self.exploring = False
            # 새로운 맵 업데이트를 기다리기 위해 노드를 종료하지 않고 맵 콜백에서 다시 탐색 시도
            # 즉, map_callback이 다시 호출될 때 plan_and_execute_exploration()이 실행됨

    def send_goal(self, goal_pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info('Sending goal to the action server.')
        send_goal_future = self.navigator.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)
        self.current_goal_active = True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Goal rejected by the action server.')
            self.current_goal_active = False
            self.current_waypoint_index += 1
            self.send_next_waypoint()
            return

        self.get_logger().info('Goal accepted by the action server.')
        self.goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        try:
            result = future.result().result
            status = future.result().status

            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Waypoint reached successfully!')
            else:
                self.get_logger().warning(f'Failed to reach waypoint. Status: {status}')
        except Exception as e:
            self.get_logger().error(f'Error in get_result_callback: {e}')

        # 다음 웨이포인트로 이동
        self.current_goal_active = False
        self.current_waypoint_index += 1
        self.send_next_waypoint()

    def perform_recovery_behavior(self):
        self.get_logger().info('Performing recovery behavior.')
        # 현재 목표가 활성화되어 있다면 취소
        if self.goal_handle is not None:
            self.get_logger().info('Cancelling current goal due to being stuck.')
            cancel_future = self.navigator.cancel_goal_async(self.goal_handle)
            cancel_future.add_done_callback(self.cancel_goal_callback)
        else:
            # 목표가 없다면 회전 후 다음 웨이포인트 시도
            self.rotate_in_place()
            self.get_logger().info('Recovery behavior completed.')
            self.send_next_waypoint()

    def cancel_goal_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Current goal successfully canceled.')
        else:
            self.get_logger().warning('Failed to cancel current goal.')

        self.goal_handle = None
        self.current_goal_active = False
        self.send_next_waypoint()

    def rotate_in_place(self):
        # 제자리에서 회전하여 장애물 회피 시도
        twist = Twist()
        twist.angular.z = 0.5  # 시계방향으로 0.5 rad/s 회전
        duration = 5.0  # 5초 동안 회전
        start_time = self.get_clock().now()
        self.get_logger().info('Rotating in place for recovery.')

        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

        # 회전 정지
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info('Rotation complete.')

    def create_pose_stamped(self, position):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = position

        # 로봇이 웨이포인트를 향하도록 방향 설정
        robot_pose = self.get_robot_pose()
        if robot_pose is not None:
            dx = position.x - robot_pose.x
            dy = position.y - robot_pose.y
            yaw = math.atan2(dy, dx)
            quaternion = self.euler_to_quaternion(0, 0, yaw)
            pose.pose.orientation = quaternion
        else:
            # 기본 방향 설정
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
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            return trans.transform.translation
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f'Could not get robot pose in map frame: {e}')
            return None

    def feedback_callback(self, feedback_msg):
        # 피드백 처리 가능 (현재는 생략)
        pass

    def shutdown_node(self):
        # 노드가 안전하게 종료되도록 처리
        if rclpy.ok():
            self.get_logger().info('Shutting down the node safely.')
            self.destroy_node()
            rclpy.shutdown()

    def save_map_to_file(self, map_msg):
        # 맵 데이터를 JSON 파일로 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        map_filename = os.path.join(self.map_save_directory, f'map_{timestamp}.json')
        map_dict = {
            'header': {
                'stamp': {
                    'sec': map_msg.header.stamp.sec,
                    'nanosec': map_msg.header.stamp.nanosec
                },
                'frame_id': map_msg.header.frame_id
            },
            'info': {
                'map_load_time': {
                    'sec': map_msg.info.map_load_time.sec,
                    'nanosec': map_msg.info.map_load_time.nanosec
                },
                'resolution': map_msg.info.resolution,
                'width': map_msg.info.width,
                'height': map_msg.info.height,
                'origin': {
                    'position': {
                        'x': map_msg.info.origin.position.x,
                        'y': map_msg.info.origin.position.y,
                        'z': map_msg.info.origin.position.z
                    },
                    'orientation': {
                        'x': map_msg.info.origin.orientation.x,
                        'y': map_msg.info.origin.orientation.y,
                        'z': map_msg.info.origin.orientation.z,
                        'w': map_msg.info.origin.orientation.w
                    }
                }
            },
            'data': map_msg.data.tolist()
        }

        try:
            with open(map_filename, 'w') as f:
                json.dump(map_dict, f, indent=4)
            self.get_logger().info(f'Map saved to {map_filename}')
        except Exception as e:
            self.get_logger().error(f'Failed to save map to file: {e}')

    def monitor_robot_movement(self):
        current_pose = self.get_robot_pose()
        if current_pose is None:
            return

        if self.last_pose is not None:
            dx = current_pose.x - self.last_pose.x
            dy = current_pose.y - self.last_pose.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance >= self.movement_threshold:
                self.last_movement_time = self.get_clock().now()
                self.stuck = False
            else:
                elapsed = (self.get_clock().now() - self.last_movement_time).nanoseconds / 1e9
                if elapsed >= self.stuck_timeout and not self.stuck:
                    self.get_logger().warning('Robot is stuck. Initiating recovery behavior.')
                    self.stuck = True
                    self.perform_recovery_behavior()
        else:
            # Initialize last_pose
            self.last_pose = current_pose
            self.last_movement_time = self.get_clock().now()

        self.last_pose = current_pose


def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        explorer.get_logger().info('Keyboard Interrupt (SIGINT)')
    except Exception as e:
        explorer.get_logger().error(f'Exception in main: {e}')
    finally:
        explorer.shutdown_node()


if __name__ == '__main__':
    main()
