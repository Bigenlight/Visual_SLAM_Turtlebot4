import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from action_msgs.msg import GoalStatus
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import yaml
import os
from PIL import Image

# "wall_follow" 액션 메시지 임포트 (실제 패키지와 액션 이름으로 수정 필요)
# 예시: from your_wall_follow_package.action import WallFollow
# 아래는 예시로 정의한 액션 메시지입니다. 실제 환경에 맞게 수정하세요.
from irobot_create_msgs.action import WallFollow  # 실제 패키지와 액션 이름으로 수정하세요

class CleaningNode(Node):
    def __init__(self):
        super().__init__('cleaning_node')

        # 로깅 레벨 설정
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # /cleaning 토픽 구독
        self.cleaning_subscriber = self.create_subscription(
            Bool, '/cleaning', self.cleaning_callback, 10)

        # /map 토픽 구독 제거
        # self.map_subscriber = self.create_subscription(
        #     OccupancyGrid, '/map', self.map_callback, 10)

        # "wall_follow" 액션 클라이언트 초기화
        self.wall_follow_client = ActionClient(self, WallFollow, 'wall_follow')

        # "navigate_to_pose" 액션 클라이언트 초기화
        self.navigate_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 변수 초기화
        self.map_data = None
        self.map_info = None
        self.cleaning_started = False
        self.initial_pose = None  # 초기 위치 저장

        # TF2 Buffer 및 Listener 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # RViz 시각화를 위한 마커 퍼블리셔 초기화
        self.marker_publisher = self.create_publisher(MarkerArray, 'cleaning_markers', 10)

        # 맵 파일 경로 설정
        self.map_file_path = '/home/theo/4_ws/map_test.yaml'  # 실제 맵 파일 경로로 수정하세요

        # 맵 데이터 로드
        self.load_map(self.map_file_path)

        # 초기 위치 저장
        self.get_initial_pose()

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
            resolution = map_yaml['resolution']
            origin = map_yaml['origin']
            negate = map_yaml.get('negate', 0)
            occupied_thresh = map_yaml.get('occupied_thresh', 0.65)
            free_thresh = map_yaml.get('free_thresh', 0.196)
        except KeyError as e:
            self.get_logger().error(f'맵 YAML 파일에서 키를 찾을 수 없습니다: {e}')
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
            map_info.origin.position.x = origin[0]
            map_info.origin.position.y = origin[1]
            map_info.origin.position.z = origin[2]
            map_info.origin.orientation.x = 0.0
            map_info.origin.orientation.y = 0.0
            map_info.origin.orientation.z = 0.0
            map_info.origin.orientation.w = 1.0

            self.map_data = map_data
            self.map_info = map_info

            self.get_logger().info(f'맵을 성공적으로 로드하였습니다: {width}x{height}, 해상도={resolution}m/pix')

        except Exception as e:
            self.get_logger().error(f'맵 이미지를 처리하는 중 오류 발생: {e}')
            return

    def get_initial_pose(self):
        """
        노드 시작 시 로봇의 초기 위치를 저장합니다.
        """
        try:
            # 'map' 프레임에서 'base_link' 프레임으로의 변환을 가져옵니다.
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.initial_pose = PoseStamped()
            self.initial_pose.header.frame_id = 'map'
            self.initial_pose.header.stamp = self.get_clock().now().to_msg()
            self.initial_pose.pose.position = trans.transform.translation
            self.initial_pose.pose.orientation = trans.transform.rotation
            self.get_logger().info(f'초기 위치 저장: ({self.initial_pose.pose.position.x:.2f}, {self.initial_pose.pose.position.y:.2f})')
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'초기 위치를 가져올 수 없습니다: {e}')

    def cleaning_callback(self, msg):
        """
        /cleaning 토픽의 콜백 함수.
        청소 시작 신호를 받으면 청소를 시작합니다.
        """
        if msg.data and not self.cleaning_started:
            self.get_logger().info('청소 시작 신호를 수신하였습니다. 청소를 시작합니다.')
            self.cleaning_started = True
            self.start_cleaning()
        elif msg.data and self.cleaning_started:
            self.get_logger().info('이미 청소가 진행 중입니다.')
        else:
            self.get_logger().info('잘못된 청소 신호를 수신하였습니다.')

    # map_callback 제거
    # def map_callback(self, msg):
    #     """
    #     /map 토픽의 콜백 함수.
    #     맵 데이터를 업데이트합니다.
    #     """
    #     self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
    #     self.map_info = msg.info

    def start_cleaning(self):
        """
        청소를 시작합니다.
        """
        # 맵 데이터가 있는지 확인
        if self.map_data is None or self.map_info is None:
            self.get_logger().error('맵 데이터가 없습니다. 청소를 시작할 수 없습니다.')
            self.cleaning_started = False
            return

        # "wall_follow" 액션 서버가 준비될 때까지 대기
        self.get_logger().info('"wall_follow" 액션 서버를 기다리는 중...')
        if not self.wall_follow_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('"wall_follow" 액션 서버가 준비되지 않았습니다.')
            self.cleaning_started = False
            return
        self.get_logger().info('"wall_follow" 액션 서버가 준비되었습니다.')

        # "wall_follow" 액션 목표 생성
        goal_msg = WallFollow.Goal()
        goal_msg.desired_distance = 0.5  # 벽과의 원하는 거리 (미터, 필요에 따라 조정)

        # 액션 요청 보내기
        self.get_logger().info('"wall_follow" 액션을 시작합니다.')
        send_goal_future = self.wall_follow_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
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
            return

        self.get_logger().info('"wall_follow" 액션 목표가 수락되었습니다.')

        # 목표 결과를 비동기로 가져옴
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.wall_follow_get_result_callback)

    def wall_follow_get_result_callback(self, future):
        """
        "wall_follow" 액션 서버로부터 결과를 처리하는 콜백 함수입니다.
        """
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('"wall_follow" 액션이 성공적으로 완료되었습니다.')
            # 초기 위치로 돌아가기
            self.return_to_initial_pose()
        else:
            self.get_logger().warn(f'"wall_follow" 액션이 실패하였습니다. 상태 코드: {status}')
            self.cleaning_started = False

    def feedback_callback(self, feedback_msg):
        """
        "wall_follow" 액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        """
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'청소 진행 중: 현재 벽과의 거리 = {feedback.current_distance:.2f}m')

    def return_to_initial_pose(self):
        """
        초기 위치로 돌아갑니다.
        """
        if self.initial_pose is None:
            self.get_logger().error('초기 위치 정보가 없습니다. 초기 위치로 돌아갈 수 없습니다.')
            self.cleaning_started = False
            return

        # "navigate_to_pose" 액션 서버가 준비될 때까지 대기
        self.get_logger().info('"navigate_to_pose" 액션 서버를 기다리는 중...')
        if not self.navigate_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('"navigate_to_pose" 액션 서버가 준비되지 않았습니다.')
            self.cleaning_started = False
            return
        self.get_logger().info('"navigate_to_pose" 액션 서버가 준비되었습니다.')

        # "navigate_to_pose" 액션 목표 생성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.initial_pose

        # 액션 요청 보내기
        self.get_logger().info('초기 위치로 돌아가는 "navigate_to_pose" 액션을 시작합니다.')
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
            return

        self.get_logger().info('"navigate_to_pose" 액션 목표가 수락되었습니다.')

        # 목표 결과를 비동기로 가져옴
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.navigate_get_result_callback)

    def navigate_get_result_callback(self, future):
        """
        "navigate_to_pose" 액션 서버로부터 결과를 처리하는 콜백 함수입니다.
        """
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('로봇이 초기 위치로 성공적으로 돌아왔습니다.')
        else:
            self.get_logger().warn(f'"navigate_to_pose" 액션이 실패하였습니다. 상태 코드: {status}')

        # 청소 플래그 리셋
        self.cleaning_started = False

    def navigate_feedback_callback(self, feedback_msg):
        """
        "navigate_to_pose" 액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        """
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'초기 위치로 이동 중: 현재 경로 진행 = {feedback.current_pose.pose.position}')

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
            marker.color.g = 0.0
            marker.color.b = 1.0
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
