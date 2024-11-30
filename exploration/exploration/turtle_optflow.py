import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2


class RobotCamORB(Node):
    def __init__(self):
        super().__init__('robot_cam_orb')
        self.bridge = CvBridge()
        self.get_logger().info("Initializing RobotCamORB Node")

        # ROS 파라미터 선언 및 초기화
        self.declare_parameter('max_lost_frames', 10)  # 기본값: 10
        self.max_lost_frames = self.get_parameter('max_lost_frames').value
        self.get_logger().info(f"Max lost frames set to: {self.max_lost_frames}")

        # 경로 길이 제한 파라미터 추가
        self.declare_parameter('max_path_length', 10)  # 기본값: 10
        self.max_path_length = self.get_parameter('max_path_length').value
        self.get_logger().info(f"Max path length set to: {self.max_path_length}")

        # 최소 및 최대 특징점 수 파라미터 추가
        self.declare_parameter('min_features', 8)     # 기본값: 8
        self.declare_parameter('max_features', 25)    # 기본값: 25
        self.min_features = self.get_parameter('min_features').value
        self.max_features = self.get_parameter('max_features').value
        self.get_logger().info(f"Minimum number of features set to: {self.min_features}")
        self.get_logger().info(f"Maximum number of features set to: {self.max_features}")

        # Lucas-Kanade Optical Flow 초기화
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(21, 21),  # Increased window size for better flow
            maxLevel=3,         # Increased maxLevel
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
        )

        # 랜덤 색상 생성
        self.colors = np.random.randint(0, 255, (1000, 3))

        # 마스크 이미지 생성
        self.mask = None

        # ORB 특징점 검출기 초기화 및 파라미터 조정
        self.orb = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=1.2,        # Default: 1.2
            nlevels=8,              # Default: 8
            edgeThreshold=15,       # Reduced edgeThreshold for more features
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,  # ORB_HARRIS_SCORE or ORB_FAST_SCORE
            patchSize=31,
            fastThreshold=10        # Lowered to detect more keypoints
        )

        # 특징점 리스트 초기화
        self.features = []  # [{'path': [Point2f, ...], 'color': (B,G,R), 'lost': int}, ...]

        # 이미지 구독자 생성
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed to /oakd/rgb/preview/image_raw/compressed")

        # 크기 조정 가능 창 생성 및 초기 크기 설정
        cv2.namedWindow("Robot Camera - ORB Optical Flow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Camera - ORB Optical Flow", 640, 640)  # 초기 창 크기 설정

    def image_callback(self, msg):
        """
        카메라 이미지 콜백 함수.
        """
        try:
            # 압축된 ROS 이미지 메시지를 OpenCV 이미지로 변환
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 이미지 크기 조정 (업스케일링 포함)
            frame = self.resize_image(frame, max_size=640, min_size=300)

            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 이미지 대비 향상 (CLAHE 적용)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # 샤프닝 필터 적용
            kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)

            # 노이즈 감소
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # 이미지 품질 로그 출력
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            self.get_logger().info(f"Image mean: {mean_val:.2f}, std: {std_val:.2f}")

            # 첫 번째 프레임일 경우 초기화
            if self.prev_gray is None:
                self.prev_gray = gray
                keypoints = self.orb.detect(self.prev_gray, None)
                keypoints, descriptors = self.orb.compute(self.prev_gray, keypoints)
                points = [kp.pt for kp in keypoints]

                # 최소 특징점 수 확인
                if len(points) < self.min_features:
                    self.get_logger().warn(f"Detected keypoints ({len(points)}) less than min_features ({self.min_features})")

                self.features = [{
                    'path': [np.array(pt, dtype=np.float32)],
                    'color': tuple(self.colors[i % len(self.colors)].tolist()),
                    'lost': 0
                } for i, pt in enumerate(points)]

                self.mask = np.zeros_like(frame)
                self.get_logger().info(f"Initial keypoints detected: {len(points)}")

                # 초기 키포인트 시각화
                frame_with_keypoints = cv2.drawKeypoints(
                    frame, keypoints, None, color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                img_large = cv2.resize(frame_with_keypoints, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)  # No upscaling here
                cv2.imshow("Robot Camera - ORB Optical Flow", img_large)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("Exiting camera display...")
                    rclpy.shutdown()
                return

            # Optical Flow 계산을 위한 이전 포인트 리스트
            if self.features:
                p0 = np.array([feature['path'][-1] for feature in self.features], dtype=np.float32).reshape(-1, 1, 2)
            else:
                p0 = None

            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            else:
                p1, st, err = None, None, None

            self.mask = np.zeros_like(frame)

            if p1 is not None and st is not None:
                new_features = []
                for i, (new, status) in enumerate(zip(p1, st)):
                    if status[0] == 1:
                        new_pt = new[0]
                        feature = self.features[i]
                        feature['path'].append(new_pt)
                        feature['lost'] = 0
                        color = feature['color']

                        if len(feature['path']) > self.max_path_length:
                            feature['path'].pop(0)

                        for j in range(1, len(feature['path'])):
                            pt1 = tuple(map(int, feature['path'][j - 1]))
                            pt2 = tuple(map(int, feature['path'][j]))
                            cv2.line(self.mask, pt1, pt2, color, 2)

                        cv2.circle(frame, tuple(map(int, new_pt)), 5, color, -1)
                        new_features.append(feature)
                    else:
                        feature = self.features[i]
                        feature['lost'] += 1
                        if feature['lost'] < self.max_lost_frames:
                            new_features.append(feature)

                self.features = new_features
                self.get_logger().info(f"Active features before addition: {len(self.features)}")

            # 활성화된 특징점이 최소 요구 수 미만일 경우 새로운 특징점 추가
            if len(self.features) < self.min_features:
                # 새로운 특징점 검출
                keypoints = self.orb.detect(gray, None)
                keypoints, descriptors = self.orb.compute(gray, keypoints)
                new_points = [kp.pt for kp in keypoints]

                # 기존 특징점과 너무 가까운 특징점 제거
                existing_points = [feature['path'][-1] for feature in self.features]
                new_features_to_add = []
                for pt in new_points:
                    too_close = False
                    for existing_pt in existing_points:
                        distance = np.linalg.norm(np.array(pt) - np.array(existing_pt))
                        if distance < 10:  # 거리 임계값 (픽셀 단위)
                            too_close = True
                            break
                    if not too_close:
                        new_features_to_add.append(pt)
                    if len(new_features_to_add) + len(self.features) >= self.max_features:
                        break

                # 새로운 특징점 추가
                for pt in new_features_to_add:
                    color = tuple(self.colors[len(self.features) % len(self.colors)].tolist())
                    self.features.append({
                        'path': [np.array(pt, dtype=np.float32)],
                        'color': color,
                        'lost': 0
                    })
                    if len(self.features) >= self.max_features:
                        break

                self.get_logger().info(f"Added {len(new_features_to_add)} new features")

            self.get_logger().info(f"Total active features after addition: {len(self.features)}")

            # 활성화된 특징점이 최대 수를 초과하지 않도록 제한
            if len(self.features) > self.max_features:
                self.features = self.features[:self.max_features]
                self.get_logger().info(f"Trimmed features to max_features: {self.max_features}")

            # 업데이트된 특징점으로 마스크 및 시각화
            for feature in self.features:
                if len(feature['path']) > 1:
                    for j in range(1, len(feature['path'])):
                        pt1 = tuple(map(int, feature['path'][j - 1]))
                        pt2 = tuple(map(int, feature['path'][j]))
                        cv2.line(self.mask, pt1, pt2, feature['color'], 2)
                # 현재 위치에 원 그리기
                cv2.circle(frame, tuple(map(int, feature['path'][-1])), 5, feature['color'], -1)

            self.get_logger().info(f"Active features after enforcing limits: {len(self.features)}")

            self.prev_gray = gray.copy()

            img = cv2.add(frame, self.mask)

            # 이미지 두 배로 업스케일링
            img_large = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            # 현재 활성화된 특징점 시각화
            if self.features:
                current_points = [tuple(map(int, feature['path'][-1])) for feature in self.features]
                for pt in current_points:
                    cv2.circle(img_large, pt, 3, (0, 255, 0), -1)

            # 팝업 창에 Optical Flow 결과 표시
            cv2.imshow("Robot Camera - ORB Optical Flow", img_large)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Exiting camera display...")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    @staticmethod
    def resize_image(image, max_size=640, min_size=300):
        """
        이미지 크기 조정 함수
        - max_size: 최대 크기
        - min_size: 최소 크기 (업스케일링을 위해)
        """
        height, width = image.shape[:2]
        max_dimension = max(width, height)
        min_dimension = min(width, height)
        scale = 1.0

        if max_dimension > max_size:
            scale = max_size / max_dimension
        elif min_dimension < min_size:
            scale = min_size / min_dimension

        if scale != 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            print(f"Image resized to: {new_width}x{new_height}")
        else:
            print("Image size within desired range. No resizing applied.")

        return image

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    robot_cam = RobotCamORB()
    try:
        rclpy.spin(robot_cam)
    except KeyboardInterrupt:
        pass
    robot_cam.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
