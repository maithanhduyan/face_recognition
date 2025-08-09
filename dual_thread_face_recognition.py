"""
Dual-Thread Face Recognition System
Tách biệt hoàn toàn: Detection Thread + Recognition/Database Thread
- Detection Thread: Chỉ nhận diện khuôn mặt (nhanh, real-time)
- Database Thread: So sánh vector với database (không chặn camera)
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import List, Dict, Optional, Tuple
import json
import os

# Import libraries
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully!")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe not available: {e}")

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Face recognition library loaded successfully!")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f"⚠️ Face recognition library not available: {e}")

try:
    import face_recognition_rust

    RUST_AVAILABLE = True
    print("✅ Rust backend with VectorDatabase loaded successfully!")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"⚠️ Rust backend not available: {e}")


class DetectionResult:
    """Container for detection results"""

    def __init__(
        self,
        face_id: str,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        vector: Optional[List[float]] = None,
    ):
        self.face_id = face_id
        self.bbox = bbox  # (x, y, w, h)
        self.confidence = confidence
        self.vector = vector
        self.recognition_name: Optional[str] = None
        self.recognition_confidence: float = 0.0
        self.timestamp = time.time()


class CameraThread:
    """Luồng camera chuyên biệt - chỉ capture frames"""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

    def start(self):
        """Bắt đầu thread camera"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("🎥 Camera thread started")

    def stop(self):
        """Dừng thread camera"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.capture.release()
        print("🎥 Camera thread stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Lấy frame mới nhất (non-blocking)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self):
        """Vòng lặp capture chính"""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue

            self.frame_count += 1

            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            # Clear queue if full và thêm frame mới
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put(frame, timeout=0.001)
            except queue.Full:
                pass


class DetectionThread:
    """Luồng nhận diện khuôn mặt - chỉ detection, không so sánh database"""

    def __init__(self):
        self.running = False
        self.thread = None

        # Queues
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=5)

        # Setup MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.7
            )
            print("✅ MediaPipe face detection initialized")

        # Setup OpenCV fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.detection_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def start(self):
        """Bắt đầu detection thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("👁️ Detection thread started")

    def stop(self):
        """Dừng detection thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("👁️ Detection thread stopped")

    def add_frame(self, frame: np.ndarray):
        """Thêm frame để xử lý (non-blocking)"""
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.input_queue.put(frame.copy(), timeout=0.001)
        except queue.Full:
            pass

    def get_results(self) -> List[DetectionResult]:
        """Lấy kết quả detection (non-blocking)"""
        results = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                results.extend(result)
            except queue.Empty:
                break
        return results

    def _detection_loop(self):
        """Vòng lặp detection chính"""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)

                # Detect faces
                detections = self._detect_faces(frame)

                # Extract vectors if faces found
                detection_results = []
                for i, (bbox, confidence) in enumerate(detections):
                    face_id = f"face_{int(time.time()*1000)}_{i}"
                    vector = self._extract_face_vector(frame, bbox)

                    result = DetectionResult(face_id, bbox, confidence, vector)
                    detection_results.append(result)

                # Send to output queue
                if detection_results:
                    try:
                        self.output_queue.put(detection_results, timeout=0.001)
                    except queue.Full:
                        pass

                self.detection_count += 1

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.detection_count / (current_time - self.last_fps_time)
                    self.detection_count = 0
                    self.last_fps_time = current_time

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Detection error: {e}")

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces trong frame"""
        if MEDIAPIPE_AVAILABLE:
            return self._detect_with_mediapipe(frame)
        else:
            return self._detect_with_opencv(frame)

    def _detect_with_mediapipe(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """MediaPipe detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results = self.face_detection.process(rgb_frame)
        detections = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]

                detections.append(((x, y, width, height), confidence))

        return detections

    def _detect_with_opencv(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """OpenCV fallback detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        return [((x, y, w, h), 0.8) for x, y, w, h in faces]

    def _extract_face_vector(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[List[float]]:
        """Trích xuất vector khuôn mặt"""
        if not FACE_RECOGNITION_AVAILABLE:
            return None

        x, y, w, h = bbox
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to face_recognition format
        face_locations = [(y, x + w, y + h, x)]
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        return encodings[0].tolist() if encodings else None


class DatabaseThread:
    """Luồng database - xử lý recognition và so sánh vector"""

    def __init__(self, db_file: str = "face_vectors.json"):
        self.running = False
        self.thread = None
        self.db_file = db_file

        # Queues
        self.input_queue = queue.Queue(maxsize=10)
        self.result_cache = {}  # face_id -> recognition result

        # Initialize Rust Vector Database
        if RUST_AVAILABLE:
            self.vector_db = face_recognition_rust.VectorDatabase()
            self.vector_db.set_threshold(0.6)  # Set threshold after creation
            self.vector_db.load_from_file(db_file)
            print(
                f"✅ Rust VectorDatabase initialized with {self.vector_db.get_face_count()} faces"
            )
        else:
            self.vector_db = None
            print("⚠️ Rust VectorDatabase not available")

        self.search_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def start(self):
        """Bắt đầu database thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._database_loop, daemon=True)
        self.thread.start()
        print("🗄️ Database thread started")

    def stop(self):
        """Dừng database thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("🗄️ Database thread stopped")

    def add_detection_results(self, results: List[DetectionResult]):
        """Thêm detection results để so sánh database"""
        for result in results:
            if result.vector:  # Only process if vector available
                try:
                    self.input_queue.put(result, timeout=0.001)
                except queue.Full:
                    pass

    def get_recognition_result(self, face_id: str) -> Optional[Tuple[str, float]]:
        """Lấy kết quả recognition cho face_id"""
        return self.result_cache.get(face_id)

    def add_face_to_database(self, name: str, vector: List[float]) -> bool:
        """Thêm khuôn mặt mới vào database"""
        if not self.vector_db:
            return False

        face_id = f"{name}_{int(time.time())}"
        success = self.vector_db.add_face_vector(face_id, name, vector)

        if success:
            # Save to file
            self.vector_db.save_to_file(self.db_file)
            print(f"✅ Added face: {name}")

        return success

    def get_known_faces_count(self) -> int:
        """Lấy số lượng faces trong database"""
        if self.vector_db:
            return self.vector_db.get_face_count()
        return 0

    def _database_loop(self):
        """Vòng lặp database chính"""
        while self.running:
            try:
                detection_result = self.input_queue.get(timeout=0.1)

                if self.vector_db and detection_result.vector:
                    # Search in database
                    search_result = self.vector_db.search_face(detection_result.vector)

                    if search_result:
                        face_id, name, confidence = search_result  # Rust returns (id, name, confidence)
                        # Store result in cache
                        self.result_cache[detection_result.face_id] = (name, confidence)

                    self.search_count += 1

                    # Calculate FPS
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.search_count / (current_time - self.last_fps_time)
                        self.search_count = 0
                        self.last_fps_time = current_time

                    # Clean old cache entries (older than 5 seconds)
                    current_time = time.time()
                    old_keys = [
                        k
                        for k, v in self.result_cache.items()
                        if current_time - getattr(detection_result, "timestamp", 0) > 5.0
                    ]
                    for key in old_keys:
                        self.result_cache.pop(key, None)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Database error: {e}")


class DualThreadFaceRecognition:
    """Hệ thống nhận diện khuôn mặt dual-thread"""

    def __init__(self):
        print("🚀 Initializing Dual-Thread Face Recognition System...")

        # Initialize threads
        self.camera_thread = CameraThread()
        self.detection_thread = DetectionThread()
        self.database_thread = DatabaseThread()

        # Display state
        self.current_detections = []
        self.show_landmarks = True
        self.show_labels = True
        self.show_stats = True

        print("✅ Dual-Thread System initialized!")

    def start(self):
        """Khởi động tất cả threads"""
        print("🚀 Starting all threads...")

        self.camera_thread.start()
        self.detection_thread.start()
        self.database_thread.start()

        print("✅ All threads started!")

    def stop(self):
        """Dừng tất cả threads"""
        print("🛑 Stopping all threads...")

        self.database_thread.stop()
        self.detection_thread.stop()
        self.camera_thread.stop()

        print("✅ All threads stopped!")

    def run_detection_only_mode(self):
        """Chế độ chỉ nhận diện khuôn mặt (không so sánh database)"""
        print("👀 Starting Detection-Only Mode...")
        print("Controls: q=quit, l=toggle landmarks, t=toggle labels, s=stats")

        self.start()

        cv2.namedWindow("Face Detection Only", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                # Get latest frame
                frame = self.camera_thread.get_latest_frame()
                if frame is not None:
                    # Send to detection
                    self.detection_thread.add_frame(frame)

                    # Get detection results
                    detections = self.detection_thread.get_results()
                    if detections:
                        self.current_detections = detections

                    # Draw results
                    display_frame = self._draw_detection_only(frame, self.current_detections)

                    # Show stats
                    if self.show_stats:
                        stats_text = f"Camera: {self.camera_thread.fps:.1f} FPS | Detection: {self.detection_thread.fps:.1f} FPS | Faces: {len(self.current_detections)}"
                        cv2.putText(
                            display_frame,
                            stats_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow("Face Detection Only", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord("t"):
                    self.show_labels = not self.show_labels
                    print(f"Labels: {'ON' if self.show_labels else 'OFF'}")
                elif key == ord("s"):
                    self.show_stats = not self.show_stats
                    print(f"Stats: {'ON' if self.show_stats else 'OFF'}")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def run_full_recognition_mode(self):
        """Chế độ nhận diện + so sánh database"""
        print("🎯 Starting Full Recognition Mode...")
        print("Controls: q=quit, l=landmarks, t=labels, s=stats, a=add face")

        self.start()

        cv2.namedWindow("Face Recognition", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Face Recognition", self._mouse_callback)

        # Face addition state
        self.adding_face = False
        self.selected_detection = None

        try:
            while True:
                # Get latest frame
                frame = self.camera_thread.get_latest_frame()
                if frame is not None:
                    # Send to detection
                    self.detection_thread.add_frame(frame)

                    # Get detection results
                    detections = self.detection_thread.get_results()
                    if detections:
                        self.current_detections = detections
                        # Send to database thread for recognition
                        self.database_thread.add_detection_results(detections)

                    # Update recognition results
                    for detection in self.current_detections:
                        result = self.database_thread.get_recognition_result(detection.face_id)
                        if result:
                            detection.recognition_name, detection.recognition_confidence = result

                    # Draw results
                    display_frame = self._draw_full_recognition(frame, self.current_detections)

                    # Show stats
                    if self.show_stats:
                        stats_text = (
                            f"Cam: {self.camera_thread.fps:.1f} | "
                            f"Det: {self.detection_thread.fps:.1f} | "
                            f"DB: {self.database_thread.fps:.1f} | "
                            f"Faces: {len(self.current_detections)} | "
                            f"Known: {self.database_thread.get_known_faces_count()}"
                        )
                        cv2.putText(
                            display_frame,
                            stats_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow("Face Recognition", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord("t"):
                    self.show_labels = not self.show_labels
                    print(f"Labels: {'ON' if self.show_labels else 'OFF'}")
                elif key == ord("s"):
                    self.show_stats = not self.show_stats
                    print(f"Stats: {'ON' if self.show_stats else 'OFF'}")
                elif key == ord("a"):
                    self._toggle_add_face_mode()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def _draw_detection_only(
        self, frame: np.ndarray, detections: List[DetectionResult]
    ) -> np.ndarray:
        """Vẽ kết quả detection only"""
        display_frame = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            confidence = detection.confidence

            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            if self.show_labels:
                # Detection info
                label = f"Face {confidence:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        return display_frame

    def _draw_full_recognition(
        self, frame: np.ndarray, detections: List[DetectionResult]
    ) -> np.ndarray:
        """Vẽ kết quả full recognition"""
        display_frame = frame.copy()

        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            confidence = detection.confidence

            # Color based on recognition status
            if detection.recognition_name:
                color = (0, 255, 0)  # Green for known
                name_text = f"{detection.recognition_name} ({detection.recognition_confidence:.2f})"
            else:
                color = (0, 255, 255)  # Yellow for unknown
                name_text = "Unknown"

            # Highlight if selected for adding
            if self.adding_face and self.selected_detection == i:
                color = (255, 0, 255)  # Magenta for selected

            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            if self.show_labels:
                # Recognition name
                cv2.putText(
                    display_frame, name_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                # Detection confidence
                det_text = f"Det: {confidence:.2f}"
                cv2.putText(
                    display_frame, det_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        # Draw add face instructions
        if self.adding_face:
            cv2.putText(
                display_frame,
                "Click on a face to add to database",
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Press 'a' again to cancel",
                (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        return display_frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for face selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.adding_face:
            # Find which face was clicked
            for i, detection in enumerate(self.current_detections):
                fx, fy, fw, fh = detection.bbox
                if fx <= x <= fx + fw and fy <= y <= fy + fh:
                    self.selected_detection = i
                    self._add_face_dialog(detection)
                    break

    def _toggle_add_face_mode(self):
        """Toggle face addition mode"""
        self.adding_face = not self.adding_face
        self.selected_detection = None
        if self.adding_face:
            print("👤 Add face mode ON - click on a face")
        else:
            print("❌ Add face mode OFF")

    def _add_face_dialog(self, detection: DetectionResult):
        """Dialog to add face"""
        if not detection.vector:
            print("❌ No face vector available")
            return

        name = input("👤 Enter name for this face: ").strip()
        if name:
            success = self.database_thread.add_face_to_database(name, detection.vector)
            if success:
                print(f"✅ Added face: {name}")
            else:
                print(f"❌ Failed to add face: {name}")

        self.adding_face = False
        self.selected_detection = None


def main():
    """Main application"""
    print("🎯 Dual-Thread Face Recognition System")
    print("=" * 60)
    print(f"🤖 MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print(f"🦀 Rust VectorDB: {'✅' if RUST_AVAILABLE else '❌'}")
    print(f"👤 Face Recognition: {'✅' if FACE_RECOGNITION_AVAILABLE else '❌'}")
    print()

    system = DualThreadFaceRecognition()

    while True:
        print("\n🎯 Choose Mode:")
        print("1. 👀 Detection Only (Fast, no database)")
        print("2. 🎯 Full Recognition (Detection + Database)")
        print("3. ❌ Exit")

        choice = input("\n👉 Choose (1-3): ").strip()

        if choice == "1":
            system.run_detection_only_mode()
        elif choice == "2":
            system.run_full_recognition_mode()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
