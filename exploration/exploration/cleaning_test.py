#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped, Quaternion, Pose
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSPresetProfiles
import numpy as np
import math
import yaml
import os
from PIL import Image
import transforms3d
from action_msgs.msg import GoalStatus
from irobot_create_msgs.action import WallFollow, Dock
from threading import Event

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

        # /amcl_pose 토픽 구독 (AMCL 초기화 확인용)
        self.amcl_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        self.amcl_pose_received = Event()

        # "navigate_to_pose" 액션 클라이언트 초기화
        self.navigate_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Initialized navigate_to_pose action client.')

        # "wall_follow" 액션 클라이언트 초기화
        self.wall_follow_client = ActionClient(self, WallFollow, 'wall_follow')
        self.get_logger().info('Initialized wall_follow action client.')

        # "dock" 액션 클라이언트 초기화
        self.dock_client = ActionClient(self, Dock, 'dock')
        self.get_logger().info('Initialized dock action client.')

        # 변수 초기화
        self.cleaning_started = False  # 청소 시작 플래그 초기화

        # /initialpose 퍼블리셔 초기화
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.get_logger().info('Initialized /initialpose publisher.')

        self.state = 'idle'  # 현재 상태: idle, coverage_cleaning, wall_following, docking
        self.coverage_waypoints = []  # 커버리지 경로 웨이포인트 리스트
        self.current_waypoint_index = 0  # 현재 진행 중인 웨이포인트 인덱스

        # 위치 추적을 위한 변수
        self.current_position = None

        # RViz 시각화를 위한 마커 퍼블리셔 초기화
        qos_marker = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.marker_publisher = self.create_publisher(MarkerArray, 'cleaning_markers', qos_marker)
        self.get_logger().info('Initialized cleaning_markers publisher.')

        # 맵 데이터 로드
        self.map_data = None
        self.map_info = None
        self.load_map()

        # 맵 로드 여부 확인
        if self.map_info is None:
            self.get_logger().error('맵 데이터를 로드하지 못했습니다. 노드를 종료합니다.')
            rclpy.shutdown()
            return

        # 청소된 영역 추적을 위한 맵 초기화
        self.cleaned_map = np.zeros((self.map_info.height, self.map_info.width), dtype=np.uint8)

        # 초기 위치가 설정되었는지 확인하는 플래그
        self.initial_pose_published = False

        # 초기 위치 퍼블리시를 위한 타이머 설정
        self.initial_pose_timer = self.create_timer(1.0, self.publish_initial_pose)
        self.get_logger().info('Timer to publish initial pose has been set.')

    def amcl_pose_callback(self, msg):
        """
        AMCL의 위치 추정을 받았을 때 호출되는 콜백 함수입니다.
        """
        if not self.amcl_pose_received.is_set():
            self.amcl_pose_received.set()
            self.get_logger().info('AMCL이 초기화되었습니다.')

    def load_map(self):
        """
        맵 파일을 로드하여 맵 데이터를 설정합니다.
        """
        # 맵 파일 경로 설정 (파라미터로 받아오기)
        self.declare_parameter('map_file_path', '/path/to/your/map.yaml')  # 실제 맵 경로로 수정하세요
        map_file_path = self.get_parameter('map_file_path').get_parameter_value().string_value

        if not os.path.exists(map_file_path):
            self.get_logger().error(f'맵 파일을 찾을 수 없습니다: {map_file_path}')
            self.map_info = None
            return

        try:
            with open(map_file_path, 'r') as file:
                map_yaml = yaml.safe_load(file)
        except Exception as e:
            self.get_logger().error(f'맵 YAML 파일을 로드하는 중 오류 발생: {e}')
            self.map_info = None
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
            self.map_info = None
            return
        except ValueError as e:
            self.get_logger().error(f'맵 YAML 파일의 값 형식이 올바르지 않습니다: {e}')
            self.map_info = None
            return

        # 이미지 파일의 절대 경로 계산
        map_dir = os.path.dirname(map_file_path)
        image_full_path = os.path.join(map_dir, image_path)

        if not os.path.exists(image_full_path):
            self.get_logger().error(f'맵 이미지 파일을 찾을 수 없습니다: {image_full_path}')
            self.map_info = None
            return

        try:
            # 이미지 로드 (흑백 이미지로 가정)
            img = Image.open(image_full_path)
            img = img.convert('L')  # 흑백으로 변환
            map_array = np.array(img)

            # 맵을 수직으로 뒤집기 (flipud)
            map_array = np.flipud(map_array)

            # OccupancyGrid 값으로 변환
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

        except Exception as e:
            self.get_logger().error(f'맵 이미지를 처리하는 중 오류 발생: {e}')
            self.map_info = None
            return

    def set_initial_pose(self):
        """
        코드 내에서 직접 초기 위치를 설정합니다.
        """
        initial_pose = Pose()
        initial_pose.position.x = 0.0  # 실제 초기 위치로 수정하세요
        initial_pose.position.y = 0.0  # 실제 초기 위치로 수정하세요
        initial_pose.position.z = 0.0
        # 초기 방향 설정 (여기서는 0도로 설정)
        quat = transforms3d.euler.euler2quat(0, 0, 0, axes='sxyz')
        initial_pose.orientation.x = quat[1]
        initial_pose.orientation.y = quat[2]
        initial_pose.orientation.z = quat[3]
        initial_pose.orientation.w = quat[0]
        self.get_logger().info(f'초기 위치 설정: (x={initial_pose.position.x:.2f}, y={initial_pose.position.y:.2f})')
        return initial_pose

    def publish_initial_pose(self):
        """
        AMCL에게 초기 위치를 알려주기 위해 /initialpose 토픽에 퍼블리시합니다.
        """
        if self.initial_pose_published:
            return

        # AMCL이 초기화될 때까지 대기
        if not self.amcl_pose_received.is_set():
            self.get_logger().info('AMCL이 초기화될 때까지 대기 중입니다...')
            return

        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.pose.pose = self.set_initial_pose()

        # Covariance 설정
        initial_pose_msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.25, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0685, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0685, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685
        ]

        self.initial_pose_publisher.publish(initial_pose_msg)
        self.get_logger().info('AMCL에 초기 위치를 퍼블리시하였습니다.')

        # 초기 위치가 퍼블리시되었음을 표시
        self.initial_pose_published = True

        # 초기 위치 퍼블리시 타이머 종료
        self.initial_pose_timer.cancel()

        # 청소 작업 시작
        self.start_cleaning()
        self.cleaning_started = True  # 청소 시작 플래그 설정

    def start_cleaning(self):
        """
        초기 위치 퍼블리시 후 청소 작업을 시작합니다.
        """
        self.get_logger().info('청소 작업을 시작합니다.')
        self.state = 'coverage_cleaning'
        self.start_coverage_cleaning()

    def odom_callback(self, msg):
        """
        /odom 토픽의 콜백 함수.
        로봇의 현재 위치를 저장하고, 청소된 영역을 업데이트합니다.
        """
        # 현재 위치 업데이트
        self.current_position = msg.pose.pose.position

        # 로봇의 위치를 맵 좌표계로 변환하여 청소된 영역 업데이트
        if self.map_info is not None and self.current_position is not None:
            map_x = int((self.current_position.x - self.map_info.origin.position.x) / self.map_info.resolution)
            map_y = int((self.current_position.y - self.map_info.origin.position.y) / self.map_info.resolution)

            if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
                self.cleaned_map[map_y, map_x] = 1  # 청소된 위치로 표시

    def calculate_coverage(self):
        """
        청소된 영역의 비율을 계산합니다.
        """
        total_free = np.count_nonzero(np.array(self.map_data) == 0)
        total_cleaned = np.count_nonzero(self.cleaned_map)
        coverage = (total_cleaned / total_free) * 100 if total_free > 0 else 0
        self.get_logger().info(f'현재 청소 커버리지: {coverage:.2f}%')
        return coverage

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
        맵 데이터를 기반으로 커버리지 경로를 생성합니다.
        """
        self.coverage_waypoints = []

        if self.map_info is None:
            self.get_logger().error('맵 정보가 없습니다. 커버리지 경로를 생성할 수 없습니다.')
            return

        # 로봇의 폭을 고려하여 그리드 크기 설정
        robot_radius = 0.34  # 로봇의 반경 (미터)
        grid_size = robot_radius * 1.5  # 로봇이 지나갈 수 있는 여유를 두기 위해 설정

        resolution = self.map_info.resolution
        width = self.map_info.width
        height = self.map_info.height
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        # 맵 데이터를 2D numpy 배열로 변환
        map_array = np.array(self.map_data).reshape((height, width))

        # 자유 공간 좌표 추출
        free_space = np.where(map_array == 0, 1, 0)

        # 그리드 기반 경로 생성
        num_cells_x = int((width * resolution) / grid_size)
        num_cells_y = int((height * resolution) / grid_size)

        x_coords = np.linspace(origin_x, origin_x + (width - 1) * resolution, num_cells_x)
        y_coords = np.linspace(origin_y, origin_y + (height - 1) * resolution, num_cells_y)

        direction = 1  # 방향 제어 변수

        for idx_y, y in enumerate(y_coords):
            if direction == 1:
                x_iter = x_coords
            else:
                x_iter = x_coords[::-1]

            for x in x_iter:
                # 해당 좌표가 자유 공간인지 확인
                map_x = int((x - origin_x) / resolution)
                map_y = int((y - origin_y) / resolution)

                if 0 <= map_x < width and 0 <= map_y < height:
                    if free_space[map_y, map_x]:
                        # 웨이포인트 생성
                        pose = PoseStamped()
                        pose.header.frame_id = 'map'
                        pose.header.stamp = self.get_clock().now().to_msg()
                        pose.pose.position.x = x
                        pose.pose.position.y = y
                        pose.pose.position.z = 0.0
                        pose.pose.orientation = self.yaw_to_quaternion(0.0)
                        self.coverage_waypoints.append(pose)

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

            # 커버리지 업데이트
            coverage = self.calculate_coverage()

            if self.state == 'coverage_cleaning':
                # 다음 웨이포인트로 이동
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.coverage_waypoints):
                    self.navigate_to_waypoint(self.coverage_waypoints[self.current_waypoint_index])
                else:
                    self.get_logger().info('커버리지 청소를 완료하였습니다.')
                    if coverage >= 95.0:
                        # 도킹 시작
                        self.state = 'docking'
                        self.start_docking()
                    else:
                        # 벽 따라기 시작
                        self.state = 'wall_following'
                        self.start_wall_follow()
            elif self.state == 'wall_following':
                self.get_logger().info('벽 따라기를 완료하였습니다.')
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

    def start_wall_follow(self):
        """
        벽 따라기를 시작합니다.
        """
        self.get_logger().info('벽 따라기를 시작합니다.')

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
        goal_msg.follow_side = 1  # FOLLOW_LEFT
        goal_msg.max_runtime = rclpy.duration.Duration(seconds=60).to_msg()  # max_runtime을 60초로 설정

        # 액션 요청 보내기
        self.get_logger().info(f'"wall_follow" 액션을 시작합니다. Goal: follow_side={goal_msg.follow_side}, max_runtime={goal_msg.max_runtime}')
        send_goal_future = self.wall_follow_client.send_goal_async(
            goal_msg,
            feedback_callback=self.wall_follow_feedback_callback
        )
        send_goal_future.add_done_callback(self.wall_follow_goal_response_callback)

    def wall_follow_goal_response_callback(self, future):
        """
        "wall_follow" 액션 서버로부터 목표 수락 응답을 처리하는 콜백 함수입니다.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('"wall_follow" 액션 목표가 거부되었습니다.')
            self.cleaning_started = False
            self.state = 'idle'
            return

        self.get_logger().info('"wall_follow" 액션 목표가 수락되었습니다.')
        self.wall_follow_goal_handle = goal_handle

        # 목표 결과를 비동기로 가져옴
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.wall_follow_get_result_callback)

    def wall_follow_get_result_callback(self, future):
        """
        "wall_follow" 액션 서버로부터 결과를 처리하는 콜백 함수입니다.
        """
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED or status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('"wall_follow" 액션이 완료되었습니다.')
            # 도킹 시작
            self.state = 'docking'
            self.start_docking()
        else:
            self.get_logger().warn(f'"wall_follow" 액션이 실패하였습니다. 상태 코드: {status}')
            self.cleaning_started = False
            self.state = 'idle'

    def wall_follow_feedback_callback(self, feedback_msg):
        """
        "wall_follow" 액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        """
        # 필요에 따라 피드백 처리
        self.get_logger().debug('벽 따라기 진행 중...')

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
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
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
