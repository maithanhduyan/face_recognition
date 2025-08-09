"""
Advanced Facial Features Detection - Nhận dạng mắt, mũi, miệng với độ chính xác cao
Sử dụng MediaPipe + OpenCV + Rust backend
"""

import cv2
import numpy as np
import time
import threading
import queue
import os
import json
from typing import List, Dict, Tuple, Optional

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully!")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe not available: {e}")

try:
    import face_recognition_rust

    RUST_AVAILABLE = True
    print("✅ Rust backend loaded successfully!")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"⚠️ Rust backend not available: {e}")

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Face recognition library loaded successfully!")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f"⚠️ Face recognition library not available: {e}")


class FaceDatabase:
    """Simple face database manager"""

    def __init__(self, db_path: str = "face_database.json"):
        self.db_path = db_path
        self.known_faces = {}  # name: encoding
        self.load_database()

    def add_face(self, name: str, encoding: np.ndarray):
        """Add face to database"""
        self.known_faces[name] = encoding.tolist() if isinstance(encoding, np.ndarray) else encoding
        self.save_database()

    def get_faces(self):
        """Get all known faces"""
        return self.known_faces

    def save_database(self):
        """Save database to file"""
        data = {
            "faces": self.known_faces,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(self.known_faces),
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_database(self):
        """Load database from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.known_faces = data.get("faces", {})
                print(f"📚 Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
                self.known_faces = {}
        else:
            print("📝 No existing database found")


class InteractiveOverlay:
    """Interactive overlay for adding faces"""

    def __init__(self):
        self.show_add_interface = False
        self.input_text = ""
        self.selected_face = None
        self.input_active = False

    def draw_button(self, image, x, y, w, h, text, color=(0, 255, 0), text_color=(255, 255, 255)):
        """Draw a button on image"""
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        return (x, y, w, h)

    def draw_input_box(self, image, x, y, w, h, text, active=False):
        """Draw text input box"""
        color = (0, 255, 255) if active else (100, 100, 100)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(image, (x + 2, y + 2), (x + w - 2, y + h - 2), (50, 50, 50), -1)

        # Text
        display_text = text if text else "Enter name..."
        cv2.putText(
            image, display_text, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return (x, y, w, h)

    def is_point_in_rect(self, point, rect):
        """Check if point is inside rectangle"""
        px, py = point
        x, y, w, h = rect
        return x <= px <= x + w and y <= py <= y + h


class AsyncLogger:
    """Non-blocking logger"""

    def __init__(self):
        self.log_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()

    def _log_worker(self):
        while self.running:
            try:
                message = self.log_queue.get(timeout=0.1)
                if message is None:
                    break
                print(message)
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def log(self, message: str):
        if not self.log_queue.full():
            self.log_queue.put(message)

    def shutdown(self):
        self.running = False
        self.log_queue.put(None)
        self.thread.join(timeout=1.0)


logger = AsyncLogger()


class ThreadedCamera:
    """Threaded camera capture for improved performance"""

    def __init__(self, src=0, buffer_size=2):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None

    def start(self):
        """Start the camera capture thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the camera capture thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def read(self):
        """Get the latest frame"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def _capture_frames(self):
        """Internal method to capture frames in separate thread"""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break

            # Clear queue if full to prevent lag
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass

    def release(self):
        """Release camera resources"""
        self.stop()
        self.capture.release()


class FaceProcessor:
    """Threaded face processing for improved performance"""

    def __init__(self, detector, detection_only: bool = False):
        self.detector = detector
        self.detection_only = detection_only
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None

    def start(self):
        """Start the processing thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def process_frame(self, frame):
        """Add frame to processing queue"""
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.input_queue.put(frame, timeout=0.01)
        except queue.Full:
            pass

    def get_result(self):
        """Get processed result"""
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None

    def _process_frames(self):
        """Internal method to process frames in separate thread"""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)

                # Process the frame
                faces = self.detector.detect_facial_features(frame)
                
                # Only perform face recognition if not in detection-only mode
                if not self.detection_only:
                    faces = self.detector.recognize_known_faces(faces)

                # Clear output queue if full
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Store result
                try:
                    self.output_queue.put((frame, faces), timeout=0.01)
                except queue.Full:
                    pass

            except queue.Empty:
                continue


class PerformanceBenchmarker:
    """Real performance benchmarking for Rust vs Python"""

    def __init__(self):
        self.results = {}

    def benchmark_distance_calculation(self, iterations: int = 1000):
        """Benchmark face distance calculation: Rust vs Python"""
        print(f"\n🏁 Benchmarking Distance Calculation ({iterations} iterations)...")

        # Generate test encodings (128-dimensional, typical for face_recognition)
        encoding1 = np.random.rand(128).tolist()
        encoding2 = np.random.rand(128).tolist()

        # Python benchmark using face_recognition library
        if FACE_RECOGNITION_AVAILABLE:
            start_time = time.time()
            for _ in range(iterations):
                distance = face_recognition.face_distance(
                    [np.array(encoding1)], np.array(encoding2)
                )[0]
            python_time = time.time() - start_time
        else:
            python_time = float("inf")

        # Rust benchmark
        if RUST_AVAILABLE:
            start_time = time.time()
            for _ in range(iterations):
                distance = face_recognition_rust.calculate_distance(encoding1, encoding2)
            rust_time = time.time() - start_time
        else:
            rust_time = float("inf")

        # Results
        speedup = python_time / rust_time if rust_time > 0 else 0

        print(f"📊 Distance Calculation Results:")
        print(f"   🐍 Python: {python_time:.4f}s ({python_time/iterations*1000:.2f}ms per op)")
        print(f"   🦀 Rust:   {rust_time:.4f}s ({rust_time/iterations*1000:.2f}ms per op)")
        print(f"   ⚡ Speedup: {speedup:.1f}x")

        self.results["distance_calculation"] = {
            "python_time": python_time,
            "rust_time": rust_time,
            "speedup": speedup,
            "iterations": iterations,
        }

        return speedup

    def benchmark_batch_processing(self, num_faces: int = 50):
        """Benchmark batch face matching: Sequential vs Parallel using available functions"""
        print(f"\n🏁 Benchmarking Batch Processing ({num_faces} faces)...")

        # Generate test data
        target_encoding = np.random.rand(128).tolist()
        known_encodings = [np.random.rand(128).tolist() for _ in range(num_faces)]

        if not RUST_AVAILABLE:
            print("❌ Rust not available for batch processing benchmark")
            return 0

        # Run multiple iterations for more accurate timing
        iterations = 10

        # Sequential processing benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            for encoding in known_encodings:
                distance = face_recognition_rust.calculate_distance(target_encoding, encoding)
        sequential_time = time.perf_counter() - start_time

        # Batch processing benchmark (if available)
        try:
            start_time = time.perf_counter()
            for _ in range(iterations):
                distances = face_recognition_rust.calculate_distances_batch(
                    target_encoding, known_encodings
                )
            parallel_time = time.perf_counter() - start_time
            batch_available = True
        except AttributeError:
            parallel_time = sequential_time  # Fallback if batch function not available
            batch_available = False

        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

        print(f"📊 Batch Processing Results:")
        print(
            f"   🔄 Sequential: {sequential_time:.6f}s ({sequential_time/iterations:.6f}s per iteration)"
        )
        print(
            f"   ⚡ Parallel:   {parallel_time:.6f}s ({parallel_time/iterations:.6f}s per iteration)"
        )
        if not batch_available:
            print("   ⚠️  Batch function not available - using sequential fallback")
        print(f"   🚀 Speedup: {speedup:.1f}x")

        self.results["batch_processing"] = {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
            "num_faces": num_faces,
            "iterations": iterations,
        }

        return speedup

    def benchmark_face_recognition(self, num_faces: int = 20, iterations: int = 100):
        """Benchmark complete face recognition pipeline"""
        print(
            f"\n🏁 Benchmarking Face Recognition Pipeline ({num_faces} faces, {iterations} iterations)..."
        )

        if not FACE_RECOGNITION_AVAILABLE or not RUST_AVAILABLE:
            print("❌ Missing dependencies for face recognition benchmark")
            return 0

        # Generate test data
        target_encoding = np.random.rand(128)
        known_encodings = [np.random.rand(128) for _ in range(num_faces)]
        known_names = [f"Person_{i}" for i in range(num_faces)]

        # Python face_recognition benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            distances = face_recognition.face_distance(known_encodings, target_encoding)
            if len(distances) > 0:
                best_match = np.argmin(distances)
        python_time = time.perf_counter() - start_time

        # Rust SimpleFaceRecognizer benchmark
        try:
            rust_recognizer = face_recognition_rust.SimpleFaceRecognizer(0.6)
            for i, encoding in enumerate(known_encodings):
                rust_recognizer.add_known_face(encoding.tolist(), known_names[i])

            start_time = time.perf_counter()
            for _ in range(iterations):
                result = rust_recognizer.recognize_face(target_encoding.tolist())
            rust_time = time.perf_counter() - start_time

        except Exception as e:
            print(f"⚠️ Rust recognizer error: {e}")
            rust_time = python_time  # Fallback

        speedup = python_time / rust_time if rust_time > 0 else 0.0

        print(f"📊 Face Recognition Results:")
        print(
            f"   🐍 Python: {python_time:.6f}s ({python_time/iterations*1000:.3f}ms per recognition)"
        )
        print(f"   🦀 Rust:   {rust_time:.6f}s ({rust_time/iterations*1000:.3f}ms per recognition)")
        print(f"   ⚡ Speedup: {speedup:.1f}x")

        self.results["face_recognition"] = {
            "python_time": python_time,
            "rust_time": rust_time,
            "speedup": speedup,
            "num_faces": num_faces,
            "iterations": iterations,
        }

        return speedup

    def run_comprehensive_benchmark(self):
        """Run all benchmarks and provide summary"""
        print("🎯 Starting Comprehensive Performance Benchmark...")
        print("=" * 60)

        # Run benchmarks
        distance_speedup = self.benchmark_distance_calculation(1000)
        batch_speedup = self.benchmark_batch_processing(50)
        recognition_speedup = self.benchmark_face_recognition(20, 100)

        # Summary
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE SUMMARY:")
        print("=" * 60)
        print(f"🔢 Distance Calculation:  {distance_speedup:.1f}x faster")
        print(f"⚡ Batch Processing:      {batch_speedup:.1f}x faster")
        print(f"👤 Face Recognition:      {recognition_speedup:.1f}x faster")

        # Memory estimation
        print(f"💾 Memory per 100 faces:  ~{100 * 128 * 8 / 1024:.1f} KB (estimated)")

        print("=" * 60)

        return {
            "distance_speedup": distance_speedup,
            "batch_speedup": batch_speedup,
            "recognition_speedup": recognition_speedup,
        }


class AdvancedFacialFeatureDetector:
    """Advanced facial feature detection system with interactive face adding"""

    def __init__(self):
        logger.log("👁️ Initializing Advanced Facial Feature Detection System...")

        # Initialize components
        self.face_database = FaceDatabase()
        # Overlay will be initialized only when needed for interactive mode

        # Show pre-trained models info
        self.show_pretrained_models()

        # OpenCV face detection - Sử dụng Haar Cascade models có sẵn
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.log(f"📂 OpenCV Haar Cascade loaded: {cascade_path}")

        # MediaPipe setup - Sử dụng pre-trained TensorFlow Lite models
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            # Face mesh for detailed landmarks
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.7,
            )

            # Face detection for bounding boxes
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.7
            )

            logger.log("✅ MediaPipe initialized with high precision settings")

        # Rust backend
        if RUST_AVAILABLE:
            self.performance_timer = face_recognition_rust.PerformanceTimer()
            self.rust_recognizer = face_recognition_rust.SimpleFaceRecognizer(0.6)

            # Load existing faces into Rust recognizer
            for name, encoding in self.face_database.known_faces.items():
                self.rust_recognizer.add_known_face(list(encoding), name)

            logger.log("✅ Rust performance timer and fast recognizer ready")
        else:
            self.rust_recognizer = None

        # Define facial landmark groups (MediaPipe indices)
        self.landmark_groups = {
            "left_eye": [
                33,
                7,
                163,
                144,
                145,
                153,
                154,
                155,
                133,
                173,
                157,
                158,
                159,
                160,
                161,
                246,
            ],
            "right_eye": [
                362,
                382,
                381,
                380,
                374,
                373,
                390,
                249,
                263,
                466,
                388,
                387,
                386,
                385,
                384,
                398,
            ],
            "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "right_eyebrow": [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            "nose_bridge": [
                6,
                51,
                48,
                115,
                131,
                134,
                102,
                49,
                220,
                305,
                309,
                392,
                439,
                438,
                457,
                358,
                344,
            ],
            "nose_tip": [
                1,
                2,
                5,
                4,
                6,
                19,
                20,
                94,
                125,
                141,
                235,
                236,
                237,
                238,
                239,
                240,
                241,
                242,
            ],
            "mouth_outer": [
                61,
                146,
                91,
                181,
                84,
                17,
                314,
                405,
                320,
                307,
                375,
                269,
                270,
                267,
                271,
                272,
            ],
            "mouth_inner": [
                78,
                95,
                88,
                178,
                87,
                14,
                317,
                402,
                318,
                324,
                308,
                324,
                318,
                312,
                308,
                324,
            ],
            "jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
        }

        logger.log("✅ Advanced Facial Feature Detector ready!")

    def show_pretrained_models(self):
        """Show information about pre-trained models being used"""
        logger.log("🤖 PRE-TRAINED MODELS ANALYSIS:")

        # OpenCV Haar Cascade Models
        haar_cascades = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
            "haarcascade_eye.xml",
            "haarcascade_eye_tree_eyeglasses.xml",
            "haarcascade_lefteye_2splits.xml",
            "haarcascade_righteye_2splits.xml",
            "haarcascade_smile.xml",
        ]

        logger.log("📂 OpenCV Haar Cascade Models Available:")
        available_cascades = []
        for cascade in haar_cascades:
            cascade_path = cv2.data.haarcascades + cascade
            try:
                test_cascade = cv2.CascadeClassifier(cascade_path)
                if not test_cascade.empty():
                    available_cascades.append(cascade)
                    logger.log(f"  ✅ {cascade}")
                else:
                    logger.log(f"  ❌ {cascade} (empty)")
            except:
                logger.log(f"  ❌ {cascade} (not found)")

        # MediaPipe Models Info
        if MEDIAPIPE_AVAILABLE:
            logger.log("🧠 MediaPipe Pre-trained Models:")
            logger.log("  ✅ Face Detection Model:")
            logger.log("    - TensorFlow Lite optimized")
            logger.log("    - Trained on millions of faces")
            logger.log("    - MobileNet architecture for speed")
            logger.log("    - ~1-2MB model size")

            logger.log("  ✅ Face Mesh Model:")
            logger.log("    - 468 3D facial landmarks")
            logger.log("    - TensorFlow Lite optimized")
            logger.log("    - Real-time performance")
            logger.log("    - ~20MB model size")
            logger.log("    - Trained on diverse dataset")

        logger.log("⚡ WHY DETECTION IS FAST:")
        logger.log("  1. 🎯 Pre-trained models = No training time")
        logger.log("  2. 🚀 TensorFlow Lite = Mobile optimized")
        logger.log("  3. 💾 Models cached in memory")
        logger.log("  4. 🔧 Hardware acceleration (when available)")
        logger.log("  5. 📱 MobileNet = Designed for speed")

        return available_cascades

    def detect_facial_features(self, image: np.ndarray) -> List[Dict]:
        """Main detection method with high accuracy"""
        results = []

        if RUST_AVAILABLE:
            self.performance_timer.start()

        # Use MediaPipe for high precision detection
        if MEDIAPIPE_AVAILABLE:
            results = self._detect_with_mediapipe(image)
        else:
            # Fallback to OpenCV
            results = self._detect_with_opencv(image)

        if RUST_AVAILABLE:
            detection_time = self.performance_timer.stop()
            if detection_time > 0.05:  # Only log if slow
                logger.log(f"⚡ Detection time: {detection_time:.4f}s")

        return results

    def _detect_with_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """High-precision detection using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Face mesh results
        mesh_results = self.face_mesh.process(rgb_image)
        # Face detection results
        detection_results = self.face_detection.process(rgb_image)

        faces = []

        if mesh_results.multi_face_landmarks and detection_results.detections:
            for detection, face_landmarks in zip(
                detection_results.detections, mesh_results.multi_face_landmarks
            ):
                # Face bounding box from detection
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                confidence = detection.score[0]

                # Extract detailed landmarks
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append(
                        {
                            "x": int(landmark.x * w),
                            "y": int(landmark.y * h),
                            "z": landmark.z,  # Depth information
                        }
                    )

                # Calculate feature regions
                features = self._extract_feature_regions(landmarks, w, h)

                face_data = {
                    "face_box": (x, y, width, height),
                    "confidence": confidence,
                    "source": "mediapipe",
                    "features": features,
                    "landmarks": landmarks,
                    "landmark_count": len(landmarks),
                }

                faces.append(face_data)

        return faces

    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict]:
        """Fallback detection using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_opencv = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        faces = []
        for x, y, w, h in faces_opencv:
            # Estimate feature positions based on face geometry
            features = self._estimate_features_opencv(x, y, w, h)

            face_data = {
                "face_box": (x, y, w, h),
                "confidence": 0.8,
                "source": "opencv",
                "features": features,
                "landmarks": [],
                "landmark_count": 0,
            }
            faces.append(face_data)

        return faces

    def _extract_feature_regions(self, landmarks: List[Dict], w: int, h: int) -> Dict:
        """Extract precise feature regions from MediaPipe landmarks"""
        features = {}

        for feature_name, indices in self.landmark_groups.items():
            points = []
            for idx in indices:
                if idx < len(landmarks):
                    point = landmarks[idx]
                    points.append((point["x"], point["y"]))

            if points:
                # Calculate bounding box
                xs, ys = zip(*points)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Add padding
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                features[feature_name] = {
                    "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
                    "points": points,
                    "center": (int(sum(xs) / len(xs)), int(sum(ys) / len(ys))),
                    "confidence": 0.95,
                }

        return features

    def _estimate_features_opencv(self, x: int, y: int, w: int, h: int) -> Dict:
        """Estimate feature positions for OpenCV detection"""
        features = {
            "left_eye": {
                "bbox": (x + int(w * 0.2), y + int(h * 0.25), int(w * 0.15), int(h * 0.15)),
                "center": (x + int(w * 0.275), y + int(h * 0.325)),
                "confidence": 0.7,
            },
            "right_eye": {
                "bbox": (x + int(w * 0.65), y + int(h * 0.25), int(w * 0.15), int(h * 0.15)),
                "center": (x + int(w * 0.725), y + int(h * 0.325)),
                "confidence": 0.7,
            },
            "nose_tip": {
                "bbox": (x + int(w * 0.4), y + int(h * 0.4), int(w * 0.2), int(h * 0.3)),
                "center": (x + int(w * 0.5), y + int(h * 0.55)),
                "confidence": 0.6,
            },
            "mouth_outer": {
                "bbox": (x + int(w * 0.3), y + int(h * 0.65), int(w * 0.4), int(h * 0.2)),
                "center": (x + int(w * 0.5), y + int(h * 0.75)),
                "confidence": 0.6,
            },
        }
        return features

    def draw_features(
        self,
        image: np.ndarray,
        faces: List[Dict],
        show_landmarks: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw detected features on image with high detail"""
        result_image = image.copy()

        for face in faces:
            x, y, w, h = face["face_box"]
            confidence = face["confidence"]
            source = face["source"]

            # Draw face box with recognition color
            recognized_name = face.get("recognized_name", "Unknown")
            recognition_confidence = face.get("recognition_confidence", 0.0)

            # Color based on recognition status
            if recognized_name != "Unknown" and recognition_confidence > 0.5:
                face_color = (0, 255, 0)  # Green for recognized
            else:
                face_color = (
                    (0, 255, 255) if source == "mediapipe" else (255, 100, 0)
                )  # Yellow for unknown

            cv2.rectangle(result_image, (x, y), (x + w, y + h), face_color, 2)

            if show_labels:
                # Show recognition name prominently
                if recognized_name != "Unknown":
                    name_label = f"{recognized_name} ({recognition_confidence:.2f})"
                    cv2.putText(
                        result_image,
                        name_label,
                        (x, y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                # Technical info (source and detection confidence)
                tech_label = f"{source.upper()} {confidence:.2f}"
                cv2.putText(
                    result_image,
                    tech_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    face_color,
                    1,
                )

            # Draw features
            features = face.get("features", {})

            # Color scheme for different features
            feature_colors = {
                "left_eye": (255, 0, 255),  # Magenta
                "right_eye": (255, 0, 255),  # Magenta
                "left_eyebrow": (0, 255, 255),  # Cyan
                "right_eyebrow": (0, 255, 255),  # Cyan
                "nose_bridge": (0, 200, 255),  # Orange
                "nose_tip": (0, 165, 255),  # Orange-red
                "mouth_outer": (0, 0, 255),  # Red
                "mouth_inner": (0, 0, 200),  # Dark red
                "jaw": (100, 100, 100),  # Gray
            }

            for feature_name, feature_data in features.items():
                if feature_name not in feature_colors:
                    continue

                color = feature_colors[feature_name]

                # Draw bounding box
                if "bbox" in feature_data:
                    fx, fy, fw, fh = feature_data["bbox"]
                    cv2.rectangle(result_image, (fx, fy), (fx + fw, fy + fh), color, 1)

                # Draw center point
                if "center" in feature_data:
                    center = feature_data["center"]
                    cv2.circle(result_image, center, 3, color, -1)

                # Draw landmarks if available
                if show_landmarks and "points" in feature_data:
                    points = feature_data["points"]
                    for point in points:
                        cv2.circle(result_image, point, 1, color, -1)

                # Feature labels
                if show_labels and "center" in feature_data:
                    center = feature_data["center"]
                    feature_conf = feature_data.get("confidence", 0)
                    label = f"{feature_name} {feature_conf:.2f}"
                    cv2.putText(
                        result_image,
                        label,
                        (center[0] - 30, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1,
                    )

        return result_image

    def extract_face_encoding(
        self, image: np.ndarray, face_box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract face encoding for recognition"""
        if not FACE_RECOGNITION_AVAILABLE:
            logger.log("⚠️ Face recognition not available")
            return None

        x, y, w, h = face_box
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to face_recognition format
        face_locations = [(y, x + w, y + h, x)]
        encodings = face_recognition.face_encodings(rgb_image, face_locations)

        return encodings[0] if encodings else None

    def recognize_known_faces(self, faces: List[Dict]) -> List[Dict]:
        """Recognize faces against known database using fast Rust backend"""
        if not FACE_RECOGNITION_AVAILABLE or not self.face_database.known_faces:
            # Set default values for unknown faces
            for face_data in faces:
                face_data["recognized_name"] = "Unknown"
                face_data["recognition_confidence"] = 0.0
            return faces

        # Prepare known faces data
        known_encodings = []
        known_names = []

        for name, encoding in self.face_database.known_faces.items():
            known_names.append(name)
            known_encodings.append(list(encoding))  # Convert to list for Rust

        # Process each detected face
        for face_data in faces:
            x, y, w, h = face_data["face_box"]

            # Extract face region from current frame
            if hasattr(self, "current_frame"):
                # Get face encoding for current face
                current_encoding = self.extract_face_encoding(self.current_frame, (x, y, w, h))

                if current_encoding is not None and known_encodings:
                    if RUST_AVAILABLE and self.rust_recognizer is not None:
                        # Use fast Rust SimpleFaceRecognizer (preferred method)
                        try:
                            recognized_name = self.rust_recognizer.recognize_face(
                                list(current_encoding)
                            )

                            if recognized_name is not None:
                                face_data["recognized_name"] = recognized_name
                                # Calculate confidence using Rust distance function
                                known_encoding = None
                                for name, encoding in self.face_database.known_faces.items():
                                    if name == recognized_name:
                                        known_encoding = list(encoding)
                                        break

                                if known_encoding:
                                    distance = face_recognition_rust.calculate_distance(
                                        list(current_encoding), known_encoding
                                    )
                                    confidence = max(0.0, 1.0 - (distance / 0.6))
                                    face_data["recognition_confidence"] = confidence
                                else:
                                    face_data["recognition_confidence"] = 0.8
                            else:
                                face_data["recognized_name"] = "Unknown"
                                face_data["recognition_confidence"] = 0.0

                        except Exception as e:
                            logger.log(f"⚠️ Rust recognition error: {e}, falling back to Python")
                            # Fallback to Python implementation
                            self._recognize_with_python(
                                face_data, current_encoding, known_encodings, known_names
                            )
                    else:
                        # Fallback to Python implementation
                        self._recognize_with_python(
                            face_data, current_encoding, known_encodings, known_names
                        )
                else:
                    face_data["recognized_name"] = "Unknown"
                    face_data["recognition_confidence"] = 0.0
            else:
                face_data["recognized_name"] = "Unknown"
                face_data["recognition_confidence"] = 0.0

        return faces

    def _recognize_with_python(self, face_data, current_encoding, known_encodings, known_names):
        """Fallback Python recognition method"""
        # Convert back to numpy arrays for face_recognition library
        known_encodings_np = [np.array(enc) for enc in known_encodings]
        distances = face_recognition.face_distance(known_encodings_np, current_encoding)

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            distance = distances[best_match_index]

            threshold = 0.6
            if distance < threshold:
                face_data["recognized_name"] = known_names[best_match_index]
                face_data["recognition_confidence"] = 1 - distance
            else:
                face_data["recognized_name"] = "Unknown"
                face_data["recognition_confidence"] = 0.0
        else:
            face_data["recognized_name"] = "Unknown"
            face_data["recognition_confidence"] = 0.0

    def draw_interactive_overlay(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Draw interactive overlay for adding faces"""
        overlay_image = image.copy()

        # Add Face button (top-right corner)
        img_h, img_w = image.shape[:2]
        add_btn = self.overlay.draw_button(
            overlay_image, img_w - 120, 10, 100, 40, "Add Face", (0, 200, 0), (255, 255, 255)
        )

        # Store button location for click detection
        self.add_button_rect = add_btn

        # Show face selection interface
        if self.overlay.show_add_interface and faces:
            # Draw selection boxes on faces
            for i, face in enumerate(faces):
                x, y, w, h = face["face_box"]

                # Highlight selected face
                if i == self.overlay.selected_face:
                    cv2.rectangle(
                        overlay_image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 3
                    )
                    cv2.putText(
                        overlay_image,
                        f"Selected Face {i+1}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.rectangle(overlay_image, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    cv2.putText(
                        overlay_image,
                        f"Click to select {i+1}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 0),
                        2,
                    )

        # Show input interface
        if self.overlay.show_add_interface:
            # Semi-transparent background
            overlay = overlay_image.copy()
            cv2.rectangle(overlay, (50, img_h - 150), (img_w - 50, img_h - 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0, overlay_image)

            # Input box
            input_rect = self.overlay.draw_input_box(
                overlay_image,
                70,
                img_h - 130,
                300,
                40,
                self.overlay.input_text,
                self.overlay.input_active,
            )
            self.input_box_rect = input_rect

            # Buttons
            save_btn = self.overlay.draw_button(
                overlay_image, 390, img_h - 130, 80, 40, "Save", (0, 200, 0), (255, 255, 255)
            )
            self.save_button_rect = save_btn

            cancel_btn = self.overlay.draw_button(
                overlay_image, 490, img_h - 130, 80, 40, "Cancel", (0, 0, 200), (255, 255, 255)
            )
            self.cancel_button_rect = cancel_btn

            # Instructions
            cv2.putText(
                overlay_image,
                "1. Click on a face to select",
                (70, img_h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                overlay_image,
                "2. Enter name and click Save",
                (70, img_h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Show known faces count
        known_count = len(self.face_database.known_faces)
        cv2.putText(
            overlay_image,
            f"Known faces: {known_count}",
            (10, img_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return overlay_image

    def process_video_stream(self, camera_index: int = 0, detection_only: bool = False):
        """Process live video stream with interactive face adding using threading"""
        # Initialize threaded components
        camera = ThreadedCamera(camera_index)
        face_processor = FaceProcessor(self, detection_only=detection_only)

        # Start threads
        camera.start()
        face_processor.start()

        if not camera.capture.isOpened():
            logger.log("❌ Cannot open camera")
            camera.release()
            face_processor.stop()
            return

        if detection_only:
            logger.log("👀 Face Detection Only - Live Stream (Threaded)")
        else:
            logger.log("🎥 Interactive Facial Feature Detection - Live Stream (Threaded)")
        logger.log("Controls:")
        logger.log("  q - quit")
        logger.log("  l - toggle landmarks")
        logger.log("  t - toggle labels")
        logger.log("  s - save screenshot")
        logger.log("  r - reset performance stats")
        if not detection_only:
            logger.log("  a - add face to database")
            logger.log("  Click 'Add Face' button to add faces interactively!")

        # Set up mouse callback
        window_title = "Face Detection Only" if detection_only else "Interactive Facial Feature Detection"
        cv2.namedWindow(window_title)
        if not detection_only:
            cv2.setMouseCallback(window_title, self.mouse_callback)
            # Initialize overlay for interactive mode only
            if not hasattr(self, 'overlay'):
                self.overlay = InteractiveOverlay()

        show_landmarks = True
        show_labels = True
        frame_count = 0
        fps_start = time.time()
        fps_counter = 0
        current_faces = []
        current_frame = None
        last_process_time = 0

        while True:
            # Get latest frame from camera thread
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)  # Small delay if no frame available
                continue

            frame_count += 1
            fps_counter += 1

            # Send frame for processing (non-blocking)
            current_time = time.time()
            if current_time - last_process_time > 0.033:  # ~30 FPS processing
                face_processor.process_frame(frame.copy())
                last_process_time = current_time

            # Get processed results (non-blocking)
            result = face_processor.get_result()
            if result is not None:
                processed_frame, faces = result
                current_faces = faces
                self.current_frame = processed_frame  # Store for face extraction
                self.current_faces = current_faces  # Store for mouse callback

            # Always display current frame with latest results
            if detection_only:
                result_frame = self.draw_features(frame, current_faces, show_landmarks, show_labels)
            else:
                result_frame = self.draw_features(frame, current_faces, show_landmarks, show_labels)
                result_frame = self.draw_interactive_overlay(result_frame, current_faces)

            # Calculate and display FPS (improved accuracy)
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
            else:
                fps = 0

            # Status overlay
            if detection_only:
                status_text = f"FPS: {fps:.1f} | Faces: {len(current_faces)} | Detection Only Mode"
            else:
                status_text = f"FPS: {fps:.1f} | Faces: {len(current_faces)} | Interactive Mode"
            cv2.putText(
                result_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow(window_title, result_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                show_landmarks = not show_landmarks
                logger.log(f"🔍 Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord("t"):
                show_labels = not show_labels
                logger.log(f"🏷️ Labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord("s"):
                filename = f"facial_features_{int(time.time())}.jpg"
                cv2.imwrite(filename, result_frame)
                logger.log(f"💾 Saved: {filename}")
            elif key == ord("r"):
                if RUST_AVAILABLE:
                    self.performance_timer = face_recognition_rust.PerformanceTimer()
                    logger.log("🔄 Performance stats reset")

            # Handle text input for name entry (only in interactive mode)
            if not detection_only and hasattr(self, 'overlay') and self.overlay.input_active:
                if 32 <= key <= 126:  # Printable characters
                    self.overlay.input_text += chr(key)
                elif key == 8:  # Backspace
                    self.overlay.input_text = self.overlay.input_text[:-1]
                elif key == 13:  # Enter
                    self.save_selected_face()

        # Cleanup threaded components
        camera.release()
        face_processor.stop()
        cv2.destroyAllWindows()
        logger.log("🎥 Threaded video stream ended")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interactive interface"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if Add Face button clicked
            if hasattr(self, "add_button_rect") and self.overlay.is_point_in_rect(
                (x, y), self.add_button_rect
            ):
                self.overlay.show_add_interface = not self.overlay.show_add_interface
                if self.overlay.show_add_interface:
                    logger.log("📝 Add face interface opened")
                else:
                    logger.log("❌ Add face interface closed")
                    self.reset_add_interface()
                return

            # Check interface buttons when add interface is shown
            if self.overlay.show_add_interface:
                # Input box clicked
                if hasattr(self, "input_box_rect") and self.overlay.is_point_in_rect(
                    (x, y), self.input_box_rect
                ):
                    self.overlay.input_active = True
                    logger.log("✏️ Name input activated - start typing!")
                    return

                # Save button clicked
                if hasattr(self, "save_button_rect") and self.overlay.is_point_in_rect(
                    (x, y), self.save_button_rect
                ):
                    self.save_selected_face()
                    return

                # Cancel button clicked
                if hasattr(self, "cancel_button_rect") and self.overlay.is_point_in_rect(
                    (x, y), self.cancel_button_rect
                ):
                    self.reset_add_interface()
                    logger.log("❌ Add face cancelled")
                    return

                # Face selection
                if hasattr(self, "current_faces"):
                    for i, face in enumerate(self.current_faces):
                        fx, fy, fw, fh = face["face_box"]
                        if fx <= x <= fx + fw and fy <= y <= fy + fh:
                            self.overlay.selected_face = i
                            logger.log(f"👤 Face {i+1} selected")
                            return

            # Deactivate input if clicked elsewhere
            self.overlay.input_active = False

    def save_selected_face(self):
        """Save the selected face with entered name"""
        if (
            self.overlay.selected_face is not None
            and hasattr(self, "current_faces")
            and self.overlay.input_text.strip()
        ):

            face_data = self.current_faces[self.overlay.selected_face]
            name = self.overlay.input_text.strip()

            # Extract face encoding
            encoding = self.extract_face_encoding(self.current_frame, face_data["face_box"])

            if encoding is not None:
                # Add to Python database
                self.face_database.add_face(name, encoding)

                # Also add to Rust recognizer for fast processing
                if RUST_AVAILABLE and self.rust_recognizer is not None:
                    self.rust_recognizer.add_known_face(list(encoding), name)

                logger.log(f"✅ Added face: {name} (Python + Rust backends)")

                # Reset interface
                self.reset_add_interface()
            else:
                logger.log("❌ Failed to extract face encoding")
        else:
            if self.overlay.selected_face is None:
                logger.log("⚠️ Please select a face first")
            elif not self.overlay.input_text.strip():
                logger.log("⚠️ Please enter a name")

    def reset_add_interface(self):
        """Reset add face interface"""
        self.overlay.show_add_interface = False
        self.overlay.input_text = ""
        self.overlay.selected_face = None
        self.overlay.input_active = False

    def show_model_details(self):
        """Show detailed information about pre-trained models"""
        print("\n🤖 PRE-TRAINED MODELS DEEP DIVE:")

        # OpenCV Models
        print("\n📂 OpenCV Haar Cascade Models:")
        print("  🎯 haarcascade_frontalface_default.xml:")
        print("    - Training: 24x24 positive samples")
        print("    - Features: Haar-like rectangular features")
        print("    - Speed: ~50-100 FPS")
        print("    - Accuracy: ~85-90% frontal faces")
        print("    - File size: ~930 KB")

        # MediaPipe Models
        if MEDIAPIPE_AVAILABLE:
            print("\n🧠 MediaPipe TensorFlow Lite Models:")
            print("  🎯 Face Detection Model (BlazeFace):")
            print("    - Architecture: MobileNet + SSD")
            print("    - Input size: 128x128 pixels")
            print("    - Training data: Google's proprietary dataset")
            print("    - Inference time: 1-2ms on mobile CPU")
            print("    - Accuracy: 95%+ on diverse faces")
            print("    - Model size: ~1.5 MB")

            print("  🎯 Face Mesh Model (FaceMesh):")
            print("    - Architecture: Custom CNN")
            print("    - Landmarks: 468 3D points")
            print("    - Training: Multi-ethnic face dataset")
            print("    - Inference time: 5-10ms")
            print("    - Accuracy: 98%+ landmark precision")
            print("    - Model size: ~20 MB")

        print("\n⚡ WHY SO FAST?")
        print("  1. 🚀 Pre-compiled models (no training needed)")
        print("  2. 🔧 TensorFlow Lite optimization")
        print("  3. 💾 Model caching in GPU/CPU memory")
        print("  4. 📱 MobileNet: Designed for mobile devices")
        print("  5. 🎯 Quantization: 8-bit instead of 32-bit")
        print("  6. ⚙️ Hardware acceleration (XNNPACK, etc.)")

    def benchmark_models(self):
        """Benchmark different detection methods"""
        print("\n🧪 RUNNING PERFORMANCE BENCHMARK...")

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        iterations = 50

        # Benchmark OpenCV
        print("\n📊 OpenCV Haar Cascade Benchmark:")
        start_time = time.time()
        for i in range(iterations):
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray)
        opencv_time = time.time() - start_time
        opencv_fps = iterations / opencv_time

        print(f"  ⏱️ Total time: {opencv_time:.4f}s")
        print(f"  🚀 FPS: {opencv_fps:.1f}")
        print(f"  📈 Avg per frame: {opencv_time/iterations*1000:.2f}ms")

        # Benchmark MediaPipe
        if MEDIAPIPE_AVAILABLE:
            print("\n📊 MediaPipe Benchmark:")
            rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            for i in range(iterations):
                detection_results = self.face_detection.process(rgb_image)
                mesh_results = self.face_mesh.process(rgb_image)
            mediapipe_time = time.time() - start_time
            mediapipe_fps = iterations / mediapipe_time

            print(f"  ⏱️ Total time: {mediapipe_time:.4f}s")
            print(f"  🚀 FPS: {mediapipe_fps:.1f}")
            print(f"  📈 Avg per frame: {mediapipe_time/iterations*1000:.2f}ms")

            # Speed comparison
            if opencv_time > 0:
                speedup = opencv_time / mediapipe_time
                faster = "MediaPipe" if speedup > 1 else "OpenCV"
                ratio = max(speedup, 1 / speedup)
                print(f"\n🏆 Winner: {faster} is {ratio:.1f}x faster")

        # Memory usage estimation
        print("\n💾 MEMORY USAGE ESTIMATES:")
        print("  📂 OpenCV Haar Cascade: ~1-2 MB RAM")
        if MEDIAPIPE_AVAILABLE:
            print("  🧠 MediaPipe Models: ~25-30 MB RAM")
            print("  📊 Total System: ~50-100 MB RAM")

        print("\n🎯 CONCLUSION:")
        print("  ✅ Pre-trained models = Instant startup")
        print("  ✅ Optimized inference = Real-time performance")
        print("  ✅ Memory efficient = Runs on mobile devices")
        print("  ✅ No training required = Ready to use")


def main():
    """Main application"""
    print("👁️ Advanced Facial Features Detection System")
    print("=" * 60)
    print(f"🤖 MediaPipe: {'✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not Available'}")
    print(f"🦀 Rust Backend: {'✅ Available' if RUST_AVAILABLE else '❌ Not Available'}")
    print()

    detector = AdvancedFacialFeatureDetector()

    while True:
        print("\n👁️ Advanced Features Menu:")
        print("1. 🎥 Live video stream detection (with face recognition)")
        print("2. � Face detection only (no database comparison)")
        print("3. �📸 Process single image")
        print("4. 🧪 Test camera")
        print("5. ⚙️ Show system info")
        print("6. 🧪 Model Performance Benchmark")
        print("7. ❌ Exit")

        choice = input("\n👉 Choose (1-7): ").strip()

        if choice == "1":
            detector.process_video_stream()

        elif choice == "2":
            detector.process_video_stream(detection_only=True)

        elif choice == "3":
            image_path = input("📂 Image path: ").strip()
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]

            try:
                image = cv2.imread(image_path)
                if image is None:
                    print("❌ Cannot load image")
                    continue

                faces = detector.detect_facial_features(image)
                result_image = detector.draw_features(image, faces)

                print(f"✅ Detected {len(faces)} faces")
                for i, face in enumerate(faces):
                    print(f"  Face {i+1}: {len(face.get('features', {}))} features detected")

                cv2.imshow("Facial Features Analysis", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                save_path = f"analyzed_{int(time.time())}.jpg"
                cv2.imwrite(save_path, result_image)
                print(f"💾 Analysis saved to: {save_path}")

            except Exception as e:
                print(f"❌ Error processing image: {e}")

        elif choice == "4":
            print("🧪 Testing camera...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✅ Camera working!")
                    cv2.imshow("Camera Test", frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("❌ Cannot capture frames")
                cap.release()
            else:
                print("❌ Cannot open camera")

        elif choice == "5":
            print("\n⚙️ System Information:")
            print(f"MediaPipe: {'✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not Available'}")
            print(f"Rust Backend: {'✅ Available' if RUST_AVAILABLE else '❌ Not Available'}")
            print(f"OpenCV Version: {cv2.__version__}")
            if MEDIAPIPE_AVAILABLE:
                print(f"MediaPipe Version: {mp.__version__}")
            print(f"Feature Groups: {len(detector.landmark_groups)} groups")
            for group_name, indices in detector.landmark_groups.items():
                print(f"  {group_name}: {len(indices)} landmarks")

            # Show detailed model information
            print("\n🤖 DETAILED MODEL ANALYSIS:")
            detector.show_model_details()

        elif choice == "6":
            print("🧪 Comprehensive Performance Benchmark")
            benchmarker = PerformanceBenchmarker()
            benchmarker.run_comprehensive_benchmark()

        elif choice == "7":
            print("👋 Goodbye!")
            logger.shutdown()
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
