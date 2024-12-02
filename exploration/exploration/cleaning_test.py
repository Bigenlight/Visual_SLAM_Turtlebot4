#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSPresetProfiles
import numpy as np
import math
import yaml
import os
from PIL import Image
from irobot_create_msgs.action import WallFollow, Dock
import transforms3d
from action_msgs.msg import GoalStatus  # 추가
import time  # 추가


class CleaningNode(Node):
    def __init__(self):
        super().__init__('cleaning_node')

        # 로깅 레벨 설정
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # /odom 토픽 구독 (QoS 설정: SENSOR_DATA)
        qos_odom = QoSPresetProfiles.get_from_short_key('sensor_data')
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_odom)
        self.get_logger().info('Subscribed to /odom topic.')

        # "wall_follow" 액션 클라이언트 초기화
        self.wall_follow_client = ActionClient(self, WallFollow, 'wall_follow')
        self.get_logger().info('Initialized wall_follow action client.')

        # "navigate_to_pose" 액션 클라이언트 초기화
        self.navigate_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Initialized navigate_to_pose action client.')

        # "dock" 액션 클라이언트 초기화
        self.dock_client = ActionClient(self, Dock, 'dock')
        self.get_logger().info('Initialized dock action client.')

        # 변수 초기화
        self.map_data = None
        self.map_info = None
        self.cleaning_started = False

        # /initialpose 퍼블리셔 초기화
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.get_logger().info('Initialized /initialpose publisher.')

        # 타이머를 사용하여 초기 위치 퍼블리시 (블로킹 방지)
        self.create_timer(2.0, self.publish_initial_pose)  # 2초 후 초기 위치 퍼블리시

        self.state = 'idle'  # 현재 상태: idle, wall_following, coverage_cleaning, docking
        self.coverage_waypoints = []  # 커버리지 경로 웨이포인트 리스트
        self.current_waypoint_index = 0  # 현재 진행 중인 웨이포인트 인덱스

        # 위치 추적을 위한 변수
        self.current_position = None
        self.start_position = None
        self.previous_position = None
        self.total_distance = 0.0

        # 벽 따라기 종료 조건 파라미터
        self.return_distance_threshold = 0.5  # 시작 지점으로부터의 거리 임계값 (미터)
        self.max_wall_follow_distance = 20.0  # 벽 따라기 최대 누적 이동 거리 (미터)

        # TF2 Buffer 및 Listener 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info('Initialized TF2 Buffer and Listener.')

        # RViz 시각화를 위한 마커 퍼블리셔 초기화
        qos_marker = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.marker_publisher = self.create_publisher(MarkerArray, 'cleaning_markers', qos_marker)
        self.get_logger().info('Initialized cleaning_markers publisher.')

        # OccupancyGrid 퍼블리셔 초기화
        qos_map = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, 'map', qos_map)
        self.get_logger().info('Initialized /map publisher.')

        # 맵 파일 경로 설정
        self.map_file_path = '/home/rokey/4_ws/map_test.yaml'  # 실제 맵 파일 경로로 수정하세요
        self.get_logger().info(f'Loading map from {self.map_file_path}')

        # 맵 데이터 로드
        self.load_map(self.map_file_path)

        # 맵 퍼블리시
        self.publish_map()

    def set_initial_pose(self):
        """
        코드 내에서 직접 초기 위치를 설정합니다.
        """
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.pose.position.x = 7.607888698577881
        initial_pose.pose.position.y = 3.164588689804077
        initial_pose.pose.position.z = 0.0
        initial_pose.pose.orientation.x = 0.0
        initial_pose.pose.orientation.y = 0.0
        initial_pose.pose.orientation.z = -0.3586598015541359
        initial_pose.pose.orientation.w = 0.9334683426603967
        self.get_logger().info(f'초기 위치 설정: (x={initial_pose.pose.position.x:.2f}, y={initial_pose.pose.position.y:.2f})')
        return initial_pose

    def publish_initial_pose(self):
        """
        AMCL에게 초기 위치를 알려주기 위해 /initialpose 토픽에 퍼블리시합니다.
        """
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.pose.pose = self.set_initial_pose().pose

        # Covariance 설정 (모든 0을 0.0으로 변경)
        initial_pose_msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.25, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.06853891945200942, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.06853891945200942, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942
        ]

        self.initial_pose_publisher.publish(initial_pose_msg)
        self.get_logger().info('AMCL에 초기 위치를 퍼블리시하였습니다.')

        # 초기 위치 퍼블리시 후 청소 시작
        self.start_cleaning()

    def load_map(self, map_file_path):
        """
        지정된 경로에서 맵 파일을 로드하여 self.map_data와 self.map_info를 설정합니다.
        """
        if not os.path.exists(map_file_path):
            self.get_logger().error(f'맵 파일을 찾을 수 없습니다: {map_file_path}')
            return

        try:
            with open(map_file_path, 'r') as file:
                map_yaml = yaml.safe_load(file)
        except Exception as e:
            self.get_logger().error(f'맵 YAML 파일을 로드하는 중 오류 발생: {e}')
            return

        # YAML에서 필요한 정보 추출
        try:
            image_path = map_yaml['image']
            resolution = float(map_yaml['resolution'])
            origin = map_yaml['origin']
            negate = map_yaml.get('negate', 0)
            occupied_thresh = map_yaml.get('occupied_thresh', 0.65)
            free_thresh = map_yaml.get('free_thresh', 0.196)
        except KeyError as e:
            self.get_logger().error(f'맵 YAML 파일에서 키를 찾을 수 없습니다: {e}')
            return
        except ValueError as e:
            self.get_logger().error(f'맵 YAML 파일의 값 형식이 올바르지 않습니다: {e}')
            return

        # 이미지 파일의 절대 경로 계산
        map_dir = os.path.dirname(map_file_path)
        image_full_path = os.path.join(map_dir, image_path)

        if not os.path.exists(image_full_path):
            self.get_logger().error(f'맵 이미지 파일을 찾을 수 없습니다: {image_full_path}')
            return

        try:
            # 이미지 로드 (흑백 이미지로 가정)
            img = Image.open(image_full_path)
            img = img.convert('L')  # 흑백으로 변환
            map_array = np.array(img)

            # 맵을 수직으로 뒤집기 (flipud)
            map_array = np.flipud(map_array)

            # 픽셀 값을 OccupancyGrid 값으로 변환
            # 일반적으로 0 (흰색) = free, 100 (검은색) = occupied, -1 = unknown
            map_data = []
            for row in map_array:
                for pixel in row:
                    if pixel == 255:
                        map_data.append(0)
                    elif pixel == 0:
                        map_data.append(100)
                    else:
                        map_data.append(-1)

            height, width = map_array.shape

            # OccupancyGrid 메타데이터 설정
            map_info = MapMetaData()
            map_info.map_load_time = self.get_clock().now().to_msg()
            map_info.resolution = resolution
            map_info.width = width
            map_info.height = height
            # origin을 float으로 변환
            map_info.origin.position.x = float(origin[0])
            map_info.origin.position.y = float(origin[1])
            map_info.origin.position.z = 0.0
            map_info.origin.orientation.x = 0.0
            map_info.origin.orientation.y = 0.0
            map_info.origin.orientation.z = 0.0
            map_info.origin.orientation.w = 1.0

            self.map_data = map_data
            self.map_info = map_info

            self.get_logger().info(f'맵을 성공적으로 로드하였습니다: {width}x{height}, 해상도={resolution}m/pix')
            self.get_logger().info(f'Origin: x={map_info.origin.position.x}, y={map_info.origin.position.y}, z={map_info.origin.position.z}')

        except Exception as e:
            self.get_logger().error(f'맵 이미지를 처리하는 중 오류 발생: {e}')
            return

    def publish_map(self):
        """
        OccupancyGrid 메시지를 퍼블리시합니다.
        """
        if self.map_data is None or self.map_info is None:
            self.get_logger().error('맵 데이터가 로드되지 않았습니다. 퍼블리시할 수 없습니다.')
            return

        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.frame_id = 'map'
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.info = self.map_info
        occupancy_grid.data = self.map_data

        self.map_publisher.publish(occupancy_grid)
        self.get_logger().info('OccupancyGrid 맵을 퍼블리시하였습니다.')

    def start_cleaning(self):
        """
        맵 로드 및 초기 위치 퍼블리시 후 청소 작업을 시작합니다.
        """
        self.get_logger().info('청소 작업을 시작합니다.')
        self.state = 'wall_following'
        self.start_wall_follow()

    def odom_callback(self, msg):
        """
        /odom 토픽의 콜백 함수.
        로봇의 현재 위치를 저장하고, 벽 따라기 종료 조건을 확인합니다.
        """
        # 현재 위치 업데이트
        self.current_position = msg.pose.pose.position

        if self.state == 'wall_following':
            if self.start_position is None:
                # 벽 따라기 시작 시 시작 위치 저장
                self.start_position = Point()
                self.start_position.x = self.current_position.x
                self.start_position.y = self.current_position.y
                self.start_position.z = self.current_position.z
                self.get_logger().info('벽 따라기를 시작합니다. 시작 위치를 저장하였습니다.')
            else:
                # 누적 이동 거리 계산
                distance = self.calculate_distance(self.previous_position, self.current_position)
                self.total_distance += distance

                # 시작 위치로부터의 거리 계산
                return_distance = self.calculate_distance(self.start_position, self.current_position)

                # 종료 조건 확인
                if return_distance <= self.return_distance_threshold and self.total_distance > 1.0:
                    self.get_logger().info('시작 위치로 돌아왔습니다. 벽 따라기를 종료합니다.')
                    self.cancel_wall_follow()
                elif self.total_distance >= self.max_wall_follow_distance:
                    self.get_logger().info('최대 이동 거리를 초과하였습니다. 벽 따라기를 종료합니다.')
                    self.cancel_wall_follow()

        # 이전 위치 업데이트
        self.previous_position = self.current_position

    def calculate_distance(self, pos1, pos2):
        """
        두 위치 간의 거리를 계산합니다.
        """
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx * dx + dy * dy)

    def start_wall_follow(self):
        """
        벽 따라가기(wall_follow)를 시작합니다.
        """
        # 초기화
        self.start_position = None
        self.previous_position = None
        self.total_distance = 0.0

        # "wall_follow" 액션 서버가 준비될 때까지 대기
        self.get_logger().info('"wall_follow" 액션 서버를 기다리는 중...')
        if not self.wall_follow_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('"wall_follow" 액션 서버가 준비되지 않았습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return
        self.get_logger().info('"wall_follow" 액션 서버가 준비되었습니다.')

        # "wall_follow" 액션 목표 생성
        goal_msg = WallFollow.Goal()
        goal_msg.follow_side = WallFollow.Goal.FOLLOW_LEFT  # 왼쪽 벽 따라가기
        # 최대 실행 시간을 충분히 크게 설정 (벽 따라기 종료는 Odom 콜백에서 처리)
        goal_msg.max_runtime = rclpy.duration.Duration(seconds=3600).to_msg()

        # 액션 요청 보내기
        self.get_logger().info('"wall_follow" 액션을 시작합니다.')
        send_goal_future = self.wall_follow_client.send_goal_async(
            goal_msg,
            feedback_callback=self.wall_follow_feedback_callback
        )
        send_goal_future.add_done_callback(self.wall_follow_goal_response_callback)

    def cancel_wall_follow(self):
        """
        벽 따라기 액션을 취소합니다.
        """
        if hasattr(self, 'wall_follow_goal_handle') and self.wall_follow_goal_handle:
            self.get_logger().info('벽 따라기 액션을 취소합니다.')
            self.wall_follow_goal_handle.cancel_goal_async()
        else:
            self.get_logger().warn('벽 따라기 액션 핸들이 존재하지 않습니다.')

    def wall_follow_goal_response_callback(self, future):
        """
        "wall_follow" 액션 서버로부터 목표 수락 응답을 처리하는 콜백 함수입니다.
        """
        self.wall_follow_goal_handle = future.result()
        if not self.wall_follow_goal_handle.accepted:
            self.get_logger().error('"wall_follow" 액션 목표가 거부되었습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return

        self.get_logger().info('"wall_follow" 액션 목표가 수락되었습니다.')

        # 목표 결과를 비동기로 가져옴
        get_result_future = self.wall_follow_goal_handle.get_result_async()
        get_result_future.add_done_callback(self.wall_follow_get_result_callback)

    def wall_follow_get_result_callback(self, future):
        """
        "wall_follow" 액션 서버로부터 결과를 처리하는 콜백 함수입니다.
        """
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED or status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('"wall_follow" 액션이 완료되었습니다.')
            # 커버리지 청소 시작
            self.state = 'coverage_cleaning'
            self.start_coverage_cleaning()
        else:
            self.get_logger().warn(f'"wall_follow" 액션이 실패하였습니다. 상태 코드: {status}')
            self.cleaning_started = False
            self.state = 'idle'

    def wall_follow_feedback_callback(self, feedback_msg):
        """
        "wall_follow" 액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        """
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'벽 따라가기 진행 중: 현재 거리 = {feedback.current_distance:.2f}m')

    def start_coverage_cleaning(self):
        """
        커버리지 청소를 시작합니다.
        """
        self.get_logger().info('커버리지 청소를 시작합니다.')
        # 커버리지 경로 생성
        self.generate_coverage_path()

        if not self.coverage_waypoints:
            self.get_logger().error('커버리지 경로 생성에 실패하였습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return

        self.get_logger().info(f'{len(self.coverage_waypoints)}개의 웨이포인트가 생성되었습니다.')
        self.publish_cleaning_markers(self.coverage_waypoints)

        # 첫 번째 웨이포인트로 이동 시작
        self.current_waypoint_index = 0
        self.navigate_to_waypoint(self.coverage_waypoints[self.current_waypoint_index])

    def generate_coverage_path(self):
        """
        지그재그 패턴의 커버리지 경로를 생성합니다.
        """
        self.coverage_waypoints = []

        # 맵 정보를 사용하여 경로 생성
        resolution = self.map_info.resolution
        width = self.map_info.width
        height = self.map_info.height
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        # 맵 데이터를 2D numpy 배열로 변환
        map_array = np.array(self.map_data).reshape((height, width))

        # 자유 공간 좌표 추출
        free_spaces = np.argwhere(map_array == 0)

        if free_spaces.size == 0:
            self.get_logger().error('맵에 자유 공간이 없습니다.')
            return

        # 자유 공간의 최소 및 최대 좌표 계산
        min_y, min_x = free_spaces.min(axis=0)
        max_y, max_x = free_spaces.max(axis=0)

        # 실제 좌표로 변환
        min_x = origin_x + min_x * resolution
        max_x = origin_x + max_x * resolution
        min_y = origin_y + min_y * resolution
        max_y = origin_y + max_y * resolution

        # 커버리지 경로 생성 (지그재그 패턴)
        step_size = 0.5  # 로봇의 폭보다 약간 작은 값으로 설정
        x = min_x
        direction = 1  # 위로 이동

        while x <= max_x:
            if direction == 1:
                y_start = min_y
                y_end = max_y
            else:
                y_start = max_y
                y_end = min_y

            # 시작점과 끝점을 생성
            start_pose = PoseStamped()
            start_pose.header.frame_id = 'map'
            start_pose.header.stamp = self.get_clock().now().to_msg()
            start_pose.pose.position.x = x
            start_pose.pose.position.y = y_start
            start_pose.pose.position.z = 0.0
            start_pose.pose.orientation = self.yaw_to_quaternion(0.0 if direction == 1 else math.pi)

            end_pose = PoseStamped()
            end_pose.header.frame_id = 'map'
            end_pose.header.stamp = self.get_clock().now().to_msg()
            end_pose.pose.position.x = x
            end_pose.pose.position.y = y_end
            end_pose.pose.position.z = 0.0
            end_pose.pose.orientation = self.yaw_to_quaternion(0.0 if direction == 1 else math.pi)

            self.coverage_waypoints.append(start_pose)
            self.coverage_waypoints.append(end_pose)

            # 다음 열로 이동
            x += step_size
            direction *= -1  # 방향 반전

    def yaw_to_quaternion(self, yaw):
        """
        yaw 값을 Quaternion으로 변환합니다.
        """
        quaternion = transforms3d.euler.euler2quat(0, 0, yaw, axes='sxyz')  # [w, x, y, z]
        return Quaternion(
            x=quaternion[1],
            y=quaternion[2],
            z=quaternion[3],
            w=quaternion[0]
        )

    def navigate_to_waypoint(self, waypoint):
        """
        지정된 웨이포인트로 이동합니다.
        """
        self.get_logger().info(f'웨이포인트로 이동: (x={waypoint.pose.position.x:.2f}, y={waypoint.pose.position.y:.2f})')

        # "navigate_to_pose" 액션 서버가 준비될 때까지 대기
        if not self.navigate_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('"navigate_to_pose" 액션 서버가 준비되지 않았습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return

        # "navigate_to_pose" 액션 목표 생성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = waypoint

        # 액션 요청 보내기
        send_goal_future = self.navigate_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigate_feedback_callback
        )
        send_goal_future.add_done_callback(self.navigate_goal_response_callback)

    def navigate_goal_response_callback(self, future):
        """
        "navigate_to_pose" 액션 서버로부터 목표 수락 응답을 처리하는 콜백 함수입니다.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('"navigate_to_pose" 액션 목표가 거부되었습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return

        self.get_logger().info('"navigate_to_pose" 액션 목표가 수락되었습니다.')

        # 목표 결과를 비동기로 가져옴
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.navigate_get_result_callback)

    def navigate_get_result_callback(self, future):
        """
        "navigate_to_pose" 액션 서버로부터 결과를 처리하는 콜백 함수입니다.
        """
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('웨이포인트에 도착하였습니다.')
            if self.state == 'coverage_cleaning':
                # 다음 웨이포인트로 이동
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.coverage_waypoints):
                    self.navigate_to_waypoint(self.coverage_waypoints[self.current_waypoint_index])
                else:
                    self.get_logger().info('커버리지 청소를 완료하였습니다.')
                    # 도킹 시작
                    self.state = 'docking'
                    self.start_docking()
            elif self.state == 'docking':
                self.get_logger().info('도킹을 완료하였습니다.')
                self.cleaning_started = False
                self.state = 'idle'
        else:
            self.get_logger().warn(f'"navigate_to_pose" 액션이 실패하였습니다. 상태 코드: {status}')
            self.cleaning_started = False
            self.state = 'idle'

    def navigate_feedback_callback(self, feedback_msg):
        """
        "navigate_to_pose" 액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        """
        # 필요에 따라 피드백 처리
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose.position
        self.get_logger().debug(f'현재 위치: x={current_pose.x:.2f}, y={current_pose.y:.2f}')

    def start_docking(self):
        """
        도킹 액션을 시작합니다.
        """
        self.get_logger().info('"dock" 액션 서버를 기다리는 중...')
        if not self.dock_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('"dock" 액션 서버가 준비되지 않았습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return
        self.get_logger().info('"dock" 액션 서버가 준비되었습니다.')

        # "dock" 액션 목표 생성
        goal_msg = Dock.Goal()
        # Dock 액션에 필요한 추가 필드가 있다면 설정하세요

        # 액션 요청 보내기
        send_goal_future = self.dock_client.send_goal_async(
            goal_msg,
            feedback_callback=self.dock_feedback_callback
        )
        send_goal_future.add_done_callback(self.dock_goal_response_callback)

    def dock_goal_response_callback(self, future):
        """
        "dock" 액션 서버로부터 목표 수락 응답을 처리하는 콜백 함수입니다.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('"dock" 액션 목표가 거부되었습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return

        self.get_logger().info('"dock" 액션 목표가 수락되었습니다.')

        # 목표 결과를 비동기로 가져옴
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.dock_get_result_callback)

    def dock_get_result_callback(self, future):
        """
        "dock" 액션 서버로부터 결과를 처리하는 콜백 함수입니다.
        """
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('"dock" 액션이 성공적으로 완료되었습니다.')
            self.cleaning_started = False
            self.state = 'idle'
        else:
            self.get_logger().warn(f'"dock" 액션이 실패하였습니다. 상태 코드: {status}')
            self.cleaning_started = False
            self.state = 'idle'

    def dock_feedback_callback(self, feedback_msg):
        """
        "dock" 액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        """
        # 필요에 따라 피드백 처리
        feedback = feedback_msg.feedback
        self.get_logger().debug('도킹 진행 중...')

    def publish_cleaning_markers(self, waypoints):
        """
        청소 진행을 시각화하기 위한 마커를 퍼블리시합니다.
        """
        marker_array = MarkerArray()
        for idx, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'cleaning_path'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = waypoint.pose.position
            marker.pose.orientation = waypoint.pose.orientation
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)
        self.get_logger().info('청소 경로 마커를 퍼블리시하였습니다.')


def main(args=None):
    rclpy.init(args=args)
    cleaning_node = CleaningNode()

    try:
        rclpy.spin(cleaning_node)
    except KeyboardInterrupt:
        cleaning_node.get_logger().info('키보드 인터럽트가 발생하여 종료합니다.')
    finally:
        cleaning_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
