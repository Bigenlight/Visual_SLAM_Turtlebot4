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
from slam_toolbox.srv import SaveMap  # 맵 저장 서비스 임포트
from std_msgs.msg import String

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # 로깅 레벨을 DEBUG로 설정하여 자세한 로그를 출력
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # /map 토픽 구독
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # NavigateToPose 액션 클라이언트 초기화
        self.navigator = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 맵 저장 서비스 클라이언트 초기화
        self.save_map_client = self.create_client(SaveMap, '/slam_toolbox/save_map')

        # 변수 초기화
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.exploring = False
        self.max_frontier_distance = 20.0  # 최대 탐사 거리 (미터)
        self.min_frontier_distance = 0.36  # 최소 목표 거리 (미터, 허용 오차)
        self.safety_distance = 0.02  # 안전 거리 (미터)
        self.max_retries = 10  # 최대 목표 재시도 횟수
        self.retry_count = 0
        self.no_frontier_count = 0  # Number of consecutive times no frontiers were found
        self.max_no_frontier_retries = 5  # Maximum retries before shutdown

        # 이동 모니터링 변수
        self.last_moving_position = None
        self.last_moving_time = None
        self.movement_check_interval = 1.0  # 매 1초마다 확인
        self.movement_threshold = 0.13  # 13 cm
        self.movement_timeout = 15.0  # 15초 동안 이동하지 않으면 정지

        # Publisher to cmd_vel to stop the robot
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # TF2 Buffer 및 Listener 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 액션 서버가 준비될 때까지 대기
        self.get_logger().info('NavigateToPose 액션 서버를 기다리는 중...')
        self.navigator.wait_for_server()
        self.get_logger().info('NavigateToPose 액션 서버가 준비되었습니다.')

        # 방문한 프론티어 목록 초기화
        self.visited_frontiers = []
        self.failed_frontiers = []  # 실패한 프론티어를 기록할 리스트
        self.frontier_distance_threshold = 0.1  # 10 cm 이내의 프론티어는 방문한 것으로 간주

        # 현재 목표 위치를 저장할 변수
        self.current_goal_position = None

        # 청소 알고리즘으로 신호를 보내기 위한 퍼블리셔 초기화
        self.cleaning_publisher = self.create_publisher(Bool, '/cleaning', 10)

        # Shutdown flag to prevent multiple shutdowns
        self.shutting_down = False

        # Exploration check timer (calls is_map_explored every 10 seconds)
        self.exploration_check_timer = self.create_timer(10.0, self.exploration_check_callback)

        # Add a timer to shut down after 600 seconds (8 minutes)
        self.shutdown_timer = self.create_timer(600.0, self.shutdown_timer_callback)
        self.get_logger().info('Shutdown timer started. The node will shut down after 600 seconds.')

    def shutdown_timer_callback(self):
        """
        Timer callback to save the map and shut down the node after 600 seconds.
        """
        self.get_logger().info('480 seconds have passed. Saving map and shutting down.')
        self.stop_robot()
        self.shutdown()
        
    def map_callback(self, msg):
        """
        /map 토픽의 콜백 함수.
        맵 데이터를 업데이트하고 탐사가 진행 중이 아니라면 탐사를 시작합니다.
        """
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

        if not self.exploring and not self.shutting_down:
            self.find_and_navigate_to_frontier()

    def exploration_check_callback(self):
        """
        주기적으로 맵이 완전히 탐사되었는지 확인합니다.
        만약 완전히 탐사되었다면, 맵을 저장하고 시스템을 종료합니다.
        """
        if self.shutting_down:
            return  # 이미 종료 중이면 아무것도 하지 않음

        if self.is_map_explored():
            self.get_logger().info('주기적 확인 중 맵이 완전히 탐사되었습니다. 시스템을 종료합니다.')
            self.stop_robot()
            self.shutdown()

    def find_and_navigate_to_frontier(self):
        """
        프론티어를 탐지하고, 클러스터링하며, 유효한 프론티어를 선택하여 네비게이션 목표를 보냅니다.
        """
        if self.exploring or self.shutting_down:
            self.get_logger().debug('Already exploring or shutting down. Skipping find_and_navigate_to_frontier.')
            return

        self.exploring = True  # Indicate that we are now exploring

        self.get_logger().debug('Starting find_and_navigate_to_frontier.')

        # 프론티어 탐지
        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info('프론티어가 감지되지 않았습니다.')

            # 맵의 미지 영역 비율과 접근 가능한 미지 영역 비율을 확인하여 맵핑 완료 여부 판단
            if self.is_map_explored():
                self.get_logger().info('맵이 충분히 탐사되었습니다. 탐사가 완료되었습니다.')
                self.stop_robot()
                self.shutdown()
            else:
                self.get_logger().info('맵이 완전히 탐사되지 않았지만, 도달 가능한 프론티어가 없습니다.')
                self.get_logger().info('Retrying exploration...')
                self.exploring = False  # Allow another attempt
            return  # 탐사 종료

        # 프론티어 클러스터링
        clustered_frontiers = self.cluster_frontiers(frontiers)

        if not clustered_frontiers:
            self.get_logger().info('클러스터링 후 유효한 프론티어가 없습니다.')

            if self.is_map_explored():
                self.get_logger().info('맵이 충분히 탐사되었습니다. 탐사가 완료되었습니다.')
                self.stop_robot()
                self.shutdown()
            else:
                self.get_logger().info('맵이 완전히 탐사되지 않았지만, 도달 가능한 프론티어가 없습니다.')
                self.stop_robot()
                self.shutdown()
            return  # 탐사 종료

        # 가장 가까운 유효한 프론티어 선택
        goal_position = self.select_frontier(clustered_frontiers)

        if goal_position is None:
            self.get_logger().info('지정된 거리 범위 내에서 도달 가능하고 안전한 프론티어가 없습니다.')

            # Retry including failed frontiers
            if self.failed_frontiers:
                self.get_logger().info('이전 실패한 프론티어를 다시 시도합니다.')
                goal_position = self.select_frontier(clustered_frontiers, include_failed_frontiers=True)

        if goal_position is None:
            self.no_frontier_count += 1
            if self.no_frontier_count >= self.max_no_frontier_retries:
                if self.is_map_explored():
                    self.get_logger().info('맵이 충분히 탐사되었습니다. 탐사가 완료되었습니다.')
                    self.stop_robot()
                    self.shutdown()
                else:
                    self.get_logger().info('맵이 완전히 탐사되지 않았지만, 도달 가능한 프론티어가 없습니다.')
                    self.stop_robot()
                    self.shutdown()
                return  # 탐사 종료
            else:
                self.get_logger().info(f'No valid frontiers found. Retrying... ({self.no_frontier_count}/{self.max_no_frontier_retries})')
                self.exploring = False  # Allow another attempt
                return
        else:
            # Reset no_frontier_count when a goal is found
            self.no_frontier_count = 0

        # 목표 지점이 있으면 탐사 진행
        self.current_goal_position = goal_position  # 현재 목표 위치 저장

        # 네비게이션 목표 메시지 생성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'odom'  # 'map'에서 'odom'으로 변경
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position = goal_position
        goal_msg.pose.pose.orientation.w = 1.0  # 앞으로 향하도록 설정

        self.get_logger().info(f'프론티어로 이동 중: ({goal_position.x:.2f}, {goal_position.y:.2f})')

        send_goal_future = self.navigator.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

        # 이동 모니터링 타이머는 목표가 수락된 후에 시작됨

    def detect_frontiers(self):
        """
        Detects frontier points in the map.
        A frontier is a free cell that is adjacent to at least one unknown cell and not adjacent to any obstacle cells.
        """
        if self.map_data is None:
            self.get_logger().warn('Map data is not available.')
            return []

        height, width = self.map_data.shape
        frontier_points = []

        # Find all free cells in the map
        free_cells = np.argwhere(self.map_data == 0)

        for cell in free_cells:
            y, x = cell
            neighbors = self.get_neighbors(x, y)
            has_unknown_neighbor = False
            has_obstacle_neighbor = False

            for nx, ny in neighbors:
                neighbor_value = self.map_data[ny, nx]
                if neighbor_value == -1:
                    has_unknown_neighbor = True
                elif neighbor_value >= 50:
                    has_obstacle_neighbor = True
                    break  # No need to check further if obstacle is found

            # Only consider the cell as a frontier if it has unknown neighbors and no obstacle neighbors
            if has_unknown_neighbor and not has_obstacle_neighbor:
                mx, my = self.grid_to_map(x, y)
                frontier_points.append([mx, my])

        self.get_logger().info(f'Detected {len(frontier_points)} frontier points.')
        return frontier_points


    def cluster_frontiers(self, frontiers):
        """
        DBSCAN을 사용하여 프론티어 포인트를 클러스터링하고 작은 클러스터는 필터링합니다.
        유효한 클러스터의 중심점을 반환합니다.
        """
        if not frontiers:
            self.get_logger().warn('No frontiers to cluster.')
            return []

        # DBSCAN 파라미터 조정 가능
        eps = 0.08
        min_samples = 2

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(frontiers)
        labels = clustering.labels_

        unique_labels = set(labels)
        clustered_frontiers = []

        min_cluster_size = 2  # 클러스터의 최소 포인트 수

        for label in unique_labels:
            if label == -1:
                # 노이즈 포인트는 제외
                continue
            indices = np.where(labels == label)[0]
            cluster = np.array(frontiers)[indices]
            cluster_size = len(cluster)

            # 너무 작은 클러스터는 무시
            if cluster_size < min_cluster_size:
                self.get_logger().debug(f'Ignoring small cluster with label {label} of size {cluster_size}')
                continue

            # 클러스터의 중심점 계산
            centroid = np.mean(cluster, axis=0)
            point = Point(x=centroid[0], y=centroid[1], z=0.0)
            clustered_frontiers.append(point)

        self.get_logger().info(f'Clustered into {len(clustered_frontiers)} valid frontiers after filtering.')

        return clustered_frontiers

    def select_frontier(self, frontiers, include_failed_frontiers=False):
        """
        방문하지 않았고 안전한 가장 가까운 프론티어를 선택합니다.
        """
        robot_position = self.get_robot_pose()
        if robot_position is None:
            self.get_logger().warning('로봇의 위치를 가져올 수 없어, 첫 번째 프론티어를 선택합니다.')
            return frontiers[0] if frontiers else None

        valid_frontiers = []
        distances = []
        for frontier in frontiers:
            # 이미 방문했거나 실패한 프론티어인지 확인
            if self.is_frontier_visited(frontier) or (not include_failed_frontiers and self.is_frontier_failed(frontier)):
                self.get_logger().debug(f'프론티어 ({frontier.x:.2f}, {frontier.y:.2f}) 이미 방문했거나 실패한 프론티어입니다.')
                continue

            dx = frontier.x - robot_position.x
            dy = frontier.y - robot_position.y
            distance = math.hypot(dx, dy)
            self.get_logger().debug(f'프론티어 ({frontier.x:.2f}, {frontier.y:.2f}), 거리: {distance:.2f}m')

            if self.min_frontier_distance <= distance <= self.max_frontier_distance:
                # 목표가 안전한지 확인
                is_safe = self.is_goal_safe(frontier.x, frontier.y, self.safety_distance)
                self.get_logger().debug(f'프론티어 ({frontier.x:.2f}, {frontier.y:.2f}) 거리 범위 내 및 안전 확인: {is_safe}')
                if is_safe:
                    valid_frontiers.append(frontier)
                    distances.append(distance)
                    self.get_logger().info(f'유효한 프론티어 ({frontier.x:.2f}, {frontier.y:.2f}), 거리: {distance:.2f}m')
            else:
                self.get_logger().debug(f'프론티어 ({frontier.x:.2f}, {frontier.y:.2f}) 거리 범위를 벗어났습니다.')

        if not valid_frontiers:
            self.get_logger().info('유효한 프론티어가 없습니다.')
            return None

        # 가장 가까운 프론티어 선택
        min_index = np.argmin(distances)
        selected_frontier = valid_frontiers[min_index]
        self.visited_frontiers.append(selected_frontier)  # 방문한 프론티어로 추가
        self.get_logger().info(f'선택된 프론티어 ({selected_frontier.x:.2f}, {selected_frontier.y:.2f})')
        return selected_frontier

    def is_goal_safe(self, goal_x, goal_y, safety_distance=0.1):
        """
        목표 위치가 안전한지 확인하여, 안전 거리 내에 장애물이 없는지 검사합니다.
        """
        if self.map_info is None or self.map_data is None:
            self.get_logger().warn('Map information or data is not available for safety check.')
            return False

        # 안전 거리에 해당하는 셀 수 계산
        num_cells = int(math.ceil(safety_distance / self.map_info.resolution))

        # 목표 위치를 그리드 좌표로 변환
        goal_grid_x = int((goal_x - self.map_info.origin.position.x) / self.map_info.resolution)
        goal_grid_y = int((goal_y - self.map_info.origin.position.y) / self.map_info.resolution)

        height, width = self.map_data.shape

        # #### 여기 삭제 시도하자
        # # 검사할 그리드 범위 정의
        # min_x = max(goal_grid_x - num_cells, 0)
        # max_x = min(goal_grid_x + num_cells, width - 1)
        # min_y = max(goal_grid_y - num_cells, 0)
        # max_y = min(goal_grid_y + num_cells, height - 1)

        # # 안전 거리 내의 각 셀을 검사
        # for y in range(min_y, max_y + 1):
        #     for x in range(min_x, max_x + 1):
        #         cell_value = self.map_data[y, x]
        #         if cell_value >= 50:  # 장애물 임계값 (필요에 따라 조정)
        #             self.get_logger().debug(f'장애물 발견: 그리드 ({x}, {y}), 값: {cell_value}')
        #             return False
        # #####
        self.get_logger().debug(f'목표 ({goal_x:.2f}, {goal_y:.2f})는 안전합니다.')
        return True

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
        그리드 좌표를 맵 좌표로 변환합니다.
        """
        mx = self.map_info.origin.position.x + (x + 0.5) * self.map_info.resolution
        my = self.map_info.origin.position.y + (y + 0.5) * self.map_info.resolution
        return mx, my

    def get_robot_pose(self):
        """
        로봇의 현재 위치를 'odom' 프레임에서 가져옵니다.
        """
        try:
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.get_logger().debug(f'로봇 위치: ({trans.transform.translation.x:.2f}, {trans.transform.translation.y:.2f})')
            return trans.transform.translation
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'로봇 위치를 가져올 수 없습니다: {e}')
            return None

    def is_frontier_visited(self, frontier):
        """
        프론티어가 이미 방문했는지, 또는 방문 시도 중인지를 확인합니다.
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
        프론티어가 이미 시도했지만 실패했는지 확인합니다.
        """
        for failed in self.failed_frontiers:
            dx = frontier.x - failed.x
            dy = frontier.y - failed.y
            distance = math.hypot(dx, dy)
            if distance < self.frontier_distance_threshold:
                return True
        return False

    def is_map_explored(self):
        """
        맵의 미지 영역 비율과 접근 가능한 미지 영역 비율을 계산하여 맵핑이 완료되었는지 판단합니다.
        """
        if self.map_data is None:
            return False

        total_cells = self.map_data.size
        unknown_cells = np.count_nonzero(self.map_data == -1)
        unknown_ratio = unknown_cells / total_cells

        self.get_logger().info(f'미지 영역 비율: {unknown_ratio:.2}')

        # 접근 가능한 미지 영역 비율 계산
        accessible_unknown_cells = self.count_accessible_unknown_cells()
        accessible_unknown_ratio = accessible_unknown_cells / total_cells

        self.get_logger().info(f'total cells: {total_cells:.2%}')

        self.get_logger().info(f'접근 가능한 미지 영역 비율: {accessible_unknown_ratio:.2%}')

        # 임계값 설정 (예: 접근 가능한 미지 영역이 수치 미만일 때 탐색 완료로 판단)
        if accessible_unknown_ratio < 0.0012:
            return True
        else:
            return False

    def count_accessible_unknown_cells(self):
        """
        Counts the number of accessible unknown cells in the map.
        Accessible unknown cells are unknown cells that are adjacent to free space.
        """
        if self.map_data is None:
            return 0

        height, width = self.map_data.shape
        accessible_unknown = 0

        for y in range(height):
            for x in range(width):
                if self.map_data[y, x] == -1:  # Unknown cell
                    neighbors = self.get_neighbors(x, y)
                    for nx, ny in neighbors:
                        if self.map_data[ny, nx] == 0:  # Free cell
                            accessible_unknown += 1
                            break  # No need to check other neighbors

        self.get_logger().info(f'접근 가능한 미지 영역 수: {accessible_unknown}')
        return accessible_unknown


    def goal_response_callback(self, future):
        """
        액션 서버로부터 목표 수락 응답을 처리하는 콜백 함수입니다.
        """
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'목표 서버에 도달하지 못했습니다: {e}')
            self.exploring = False
            self.stop_robot()
            return

        if not goal_handle.accepted:
            self.get_logger().info('목표가 거부되었습니다.')
            self.exploring = False
            self.stop_robot()
            return

        self.get_logger().info('목표가 수락되었습니다.')
        self.current_goal = goal_handle

        # 목표 결과를 비동기로 가져옴
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

        # 이동 모니터링 타이머 시작
        self.start_movement_monitoring()

    def get_result_callback(self, future):
        """
        액션 서버로부터 목표 결과를 처리하는 콜백 함수입니다.
        """
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'목표 결과를 가져오는 데 실패했습니다: {e}')
            self.handle_goal_failure()
            return

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('목표에 도달하였습니다!')
            self.retry_count = 0  # 성공 시 재시도 횟수 초기화
        else:
            self.get_logger().info(f'목표가 실패하였습니다. 상태: {status}')
            self.handle_goal_failure()

        # 이동 모니터링 타이머 정지
        self.stop_movement_monitoring()

        # 탐사 플래그 리셋하여 다음 탐사를 허용
        self.exploring = False

        # 다음 프론티어 탐색
        self.find_and_navigate_to_frontier()

    def handle_goal_failure(self):
        """
        목표 실패 시 재시도하거나, 최대 재시도 횟수를 초과하면 다음 프론티어로 이동합니다.
        """
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.get_logger().warn('최대 재시도 횟수에 도달하였습니다. 프론티어를 실패로 기록하고 다음 프론티어를 탐색합니다.')
            if self.current_goal_position is not None:
                self.failed_frontiers.append(self.current_goal_position)  # 실패한 프론티어로 기록
            self.retry_count = 0  # 재시도 횟수 초기화
            self.current_goal_position = None  # 현재 목표 위치 초기화
            self.exploring = False
            self.find_and_navigate_to_frontier()
        else:
            self.get_logger().info('같은 프론티어로 재시도합니다.')
            self.exploring = False  # 탐사를 재시도하도록 플래그 리셋
            self.find_and_navigate_to_frontier()

    def feedback_callback(self, feedback_msg):
        """
        액션 서버로부터 피드백을 처리하는 콜백 함수입니다.
        현재는 사용되지 않지만 필요에 따라 구현할 수 있습니다.
        """
        pass


    # Update the shutdown method
    def shutdown(self):
        """
        탐사를 중단하고 노드를 정상적으로 종료합니다.
        """
        if self.shutting_down:
            return  # 이미 종료 중이면 아무것도 하지 않음

        self.shutting_down = True
        self.get_logger().info('탐사 노드를 종료합니다.')

        try:
            # 맵 저장 서비스 호출
            self.save_map()
        except Exception as e:
            self.get_logger().error(f'맵 저장 중 예외 발생: {e}')

        # Cancel any current goals
        if self.current_goal is not None:
            self.get_logger().info('현재 목표를 취소합니다.')
            cancel_future = self.current_goal.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)
            self.current_goal = None
            
        # Stop the robot
        self.stop_robot()

        # 청소 알고리즘 시작 신호 발행
        cleaning_msg = Bool()
        cleaning_msg.data = True
        self.cleaning_publisher.publish(cleaning_msg)
        self.get_logger().info('청소 시작 신호를 발행하였습니다.')

        # 모든 타이머 취소
        self.cancel_all_timers()

        # 노드 파괴 및 종료
        self.destroy_node()
        rclpy.shutdown()

    def save_map(self):
        """
        현재 맵을 저장하기 위해 /slam_toolbox/save_map 서비스를 호출합니다.
        """
        self.get_logger().info('맵 저장 서비스를 호출합니다.')

        # 서비스가 준비될 때까지 대기
        if not self.save_map_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/slam_toolbox/save_map 서비스가 준비되지 않았습니다.')
            return

        # 서비스 요청 생성
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
                self.get_logger().info('맵이 성공적으로 저장되었습니다.')
            else:
                self.get_logger().error(f"맵 저장에 실패하였습니다: {future.result().message}")
        else:
            self.get_logger().error('맵 저장 서비스 호출에 실패하였습니다.')

    
    def cancel_all_timers(self):
        """
        모든 활성 타이머를 취소하여 깨끗하게 종료합니다.
        """
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.get_logger().debug('이동 모니터링 타이머를 취소하였습니다.')

        if hasattr(self, 'exploration_check_timer') and self.exploration_check_timer is not None:
            self.exploration_check_timer.cancel()
            self.get_logger().debug('탐사 확인 타이머를 취소하였습니다.')


    def stop_robot(self):
        """
        로봇을 정지시키기 위해 제로 속도를 퍼블리시합니다.
        """
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_msg)
        self.get_logger().info('로봇 정지 명령을 보냈습니다.')

    # 이동 모니터링 메소드
    def start_movement_monitoring(self):
        """
        현재 위치를 기록하고 이동을 모니터링하기 위한 타이머를 시작합니다.
        """
        # 기존의 이동 모니터링 타이머가 있다면 취소
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.get_logger().debug('기존 이동 모니터링 타이머가 취소되었습니다.')

        self.last_moving_position = self.get_robot_pose()
        self.last_moving_time = None  # 초기화
        self.movement_timer = self.create_timer(self.movement_check_interval, self.check_movement_callback)
        self.get_logger().debug('이동 모니터링을 시작하였습니다.')

    def stop_movement_monitoring(self):
        """
        이동 모니터링 타이머를 정지합니다.
        """
        if hasattr(self, 'movement_timer') and self.movement_timer is not None:
            self.movement_timer.cancel()
            self.movement_timer = None
            self.get_logger().debug('이동 모니터링을 정지하였습니다.')

    def check_movement_callback(self):
        self.get_logger().debug('check_movement_callback called')

        """
        주기적으로 로봇이 충분히 이동했는지 확인합니다.
        이동하지 않았을 경우 다음 프론티어를 탐색합니다.
        """
        current_time = self.get_clock().now()
        current_position = self.get_robot_pose()
        if current_position is None or self.last_moving_position is None:
            return  # 위치를 가져올 수 없는 경우

        # 마지막으로 이동한 이후로 이동한 거리 계산
        dx = current_position.x - self.last_moving_position.x
        dy = current_position.y - self.last_moving_position.y
        distance_moved = math.hypot(dx, dy)

        if distance_moved >= self.movement_threshold:
            # 충분히 이동했으면 위치와 시간을 업데이트
            self.last_moving_position = current_position
            self.last_moving_time = None  # 리셋
            self.get_logger().debug('로봇이 충분히 이동하였습니다.')
        else:
            if self.last_moving_time is None:
                # 로봇이 이동하지 않았을 경우 타이머 시작
                self.last_moving_time = current_time
                self.get_logger().debug('로봇이 충분히 이동하지 않았습니다. 이동 타임아웃 타이머를 시작합니다.')
                return

            # 마지막으로 이동한 이후로 경과한 시간 계산
            time_since_last_move = (current_time - self.last_moving_time).nanoseconds / 1e9  # 초 단위
            self.get_logger().debug(f'로봇이 {time_since_last_move:.2f}초 동안 충분히 이동하지 않았습니다.')

            if time_since_last_move >= self.movement_timeout:
                # 이동하지 않은 시간이 임계값을 초과하면 다음 프론티어 탐색
                self.get_logger().warn('로봇이 정체되었습니다. 다음 프론티어를 탐색합니다.')
                if self.current_goal_position is not None:
                    self.failed_frontiers.append(self.current_goal_position)  # 실패한 프론티어로 기록
                # Reset exploring flag
                self.exploring = False
                # Send the next goal
                self.find_and_navigate_to_frontier()
                # Reset movement monitoring variables
                self.last_moving_position = current_position
                self.last_moving_time = None
                # Exploration continues


def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()

    # Use a MultiThreadedExecutor to allow timers and callbacks to run in parallel
    executor = rclpy.executors.MultiThreadedExecutor()
    try:
        rclpy.spin(explorer, executor=executor)
    except KeyboardInterrupt:
        explorer.get_logger().info('키보드 인터럽트가 발생하여 종료합니다.')
        explorer.shutdown()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
