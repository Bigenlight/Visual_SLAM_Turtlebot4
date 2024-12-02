#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, Pose
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import numpy as np
import math
import yaml
import os
from PIL import Image
import transforms3d
from action_msgs.msg import GoalStatus
from irobot_create_msgs.action import WallFollow, Dock
from threading import Event
from std_srvs.srv import Empty
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped

class CleaningNode(Node):
    def __init__(self):
        super().__init__('cleaning_node')

        # 로깅 레벨 설정
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # /odom 토픽 구독 (QoS 설정: SENSOR_DATA)
        qos_odom = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_odom)
        self.get_logger().info('Subscribed to /odom topic.')

        # /amcl_pose 토픽 구독 (AMCL 위치 추정용)
        self.amcl_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        self.amcl_pose_received = False

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

        self.state = 'idle'  # 현재 상태: idle, coverage_cleaning, wall_following, docking
        self.coverage_waypoints = []  # 커버리지 경로 웨이포인트 리스트
        self.current_waypoint_index = 0  # 현재 진행 중인 웨이포인트 인덱스

        # 위치 추적을 위한 변수
        self.current_position = None

        # 로봇의 초기 위치
        self.robot_initial_pose = None

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

        # TF 버퍼 및 리스너 초기화
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # /initialpose 퍼블리셔 초기화
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.get_logger().info('Initialized /initialpose publisher.')

        # 초기 위치 퍼블리시를 위한 타이머 설정
        self.initial_pose_published = False
        self.initial_pose_timer = self.create_timer(1.0, self.publish_initial_pose)
        self.get_logger().info('Timer to publish initial pose has been set.')

        # AMCL 초기화 대기를 위한 타이머 설정
        self.amcl_wait_timer = self.create_timer(1.0, self.check_amcl_initialization)
        self.get_logger().info('Timer to check AMCL initialization has been set.')

        # /global_localization 서비스 클라이언트 생성
        self.global_localization_client = self.create_client(Empty, '/global_localization')

        # 서비스 서버가 준비될 때까지 대기
        while not self.global_localization_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/global_localization 서비스 대기 중...')

        # 글로벌 로컬라이제이션 서비스 요청
        self.call_global_localization()

    def call_global_localization(self):
        """
        /global_localization 서비스를 호출하여 AMCL이 글로벌 로컬라이제이션을 수행하도록 합니다.
        """
        self.get_logger().info('/global_localization 서비스를 호출합니다.')
        req = Empty.Request()
        future = self.global_localization_client.call_async(req)
        future.add_done_callback(self.global_localization_response_callback)

    def global_localization_response_callback(self, future):
        self.get_logger().info('글로벌 로컬라이제이션 서비스 호출이 완료되었습니다.')

    def publish_initial_pose(self):
        """
        AMCL에게 초기 위치를 알려주기 위해 /initialpose 토픽에 퍼블리시합니다.
        """
        if self.initial_pose_published:
            return

        # 초기 위치 설정 (로봇의 실제 초기 위치로 수정하세요)
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.header.frame_id = 'map'
        initial_pose.pose.pose.position.x = 0.0  # 실제 초기 위치로 수정하세요
        initial_pose.pose.pose.position.y = 0.0  # 실제 초기 위치로 수정하세요
        initial_pose.pose.pose.position.z = 0.0
        quat = transforms3d.euler.euler2quat(0, 0, 0, axes='sxyz')
        initial_pose.pose.pose.orientation.x = quat[1]
        initial_pose.pose.pose.orientation.y = quat[2]
        initial_pose.pose.pose.orientation.z = quat[3]
        initial_pose.pose.pose.orientation.w = quat[0]
        # 공분산 설정
        initial_pose.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.25, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0685, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0685, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685
        ]

        self.initial_pose_publisher.publish(initial_pose)
        self.get_logger().info('AMCL에 초기 위치를 퍼블리시하였습니다.')

        self.initial_pose_published = True
        self.initial_pose_timer.cancel()

    def amcl_pose_callback(self, msg):
        """
        AMCL의 위치 추정을 받았을 때 호출되는 콜백 함수입니다.
        """
        if not self.amcl_pose_received:
            self.amcl_pose_received = True
            self.robot_initial_pose = msg.pose.pose
            self.get_logger().info('AMCL에서 로봇의 초기 위치를 받았습니다.')
            self.get_logger().info(f'로봇 초기 위치: x={self.robot_initial_pose.position.x:.2f}, y={self.robot_initial_pose.position.y:.2f}')
            # AMCL 초기화 대기 타이머 취소
            self.amcl_wait_timer.cancel()
            # 청소 작업 시작
            self.start_cleaning()

    def check_amcl_initialization(self):
        """
        AMCL이 초기화되었는지 확인합니다.
        """
        if not self.amcl_pose_received:
            self.get_logger().info('AMCL에서 위치 추정을 기다리는 중입니다...')
        else:
            # 이미 AMCL 위치를 받았으므로 타이머를 취소합니다.
            self.amcl_wait_timer.cancel()

    def load_map(self):
        """
        맵 파일을 로드하여 맵 데이터를 설정합니다.
        """
        # 맵 파일 경로 설정 (파라미터로 받아오기)
        self.declare_parameter('map_file_path', '/home/theo/4_ws/map_test.yaml')  # 실제 맵 경로로 수정하세요
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

            # OccupancyGrid 값으로 변환
            map_data = []
            for row in map_array:
                for pixel in row:
                    if pixel == 254 or pixel == 255:
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

    def start_cleaning(self):
        """
        AMCL로부터 위치를 받은 후 청소 작업을 시작합니다.
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
        맵 데이터를 기반으로 Boustrophedon Path Planning 알고리즘을 사용하여 커버리지 경로를 생성합니다.
        """
        self.coverage_waypoints = []

        if self.map_info is None:
            self.get_logger().error('맵 정보가 없습니다. 커버리지 경로를 생성할 수 없습니다.')
            return

        # 로봇의 폭을 고려하여 셀 크기 설정
        robot_width = 0.68  # 로봇의 폭 (미터)
        cell_size = robot_width * 0.8  # 셀 크기를 로봇 폭의 80%로 설정

        resolution = self.map_info.resolution
        width = self.map_info.width
        height = self.map_info.height
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        # 맵 데이터를 2D numpy 배열로 변환
        map_array = np.array(self.map_data).reshape((height, width))
        map_array = np.flipud(map_array)  # 좌표계를 맞추기 위해 상하 반전

        # 자유 공간 좌표 추출
        free_space = np.where(map_array == 0, 1, 0)

        # 셀 크기에 따른 그리드 생성
        num_cells_x = int((width * resolution) / cell_size)
        num_cells_y = int((height * resolution) / cell_size)

        x_coords = np.linspace(0, width - 1, num_cells_x, dtype=int)
        y_coords = np.linspace(0, height - 1, num_cells_y, dtype=int)

        direction = 1  # 이동 방향 제어 변수

        for idx_y, y in enumerate(y_coords):
            if direction == 1:
                x_iter = x_coords
            else:
                x_iter = x_coords[::-1]

            for x in x_iter:
                if free_space[y, x]:
                    # 맵 좌표를 실제 좌표로 변환
                    real_x = origin_x + x * resolution
                    real_y = origin_y + y * resolution

                    # 웨이포인트 생성
                    pose = PoseStamped()
                    pose.header.frame_id = 'map'
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.pose.position.x = real_x
                    pose.pose.position.y = real_y
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
        self.dock_goal_handle = goal_handle

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
