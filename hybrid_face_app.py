"""
Hybrid Face Recognition System - Python + Rust
Sử dụng Rust backend khi có thể, fallback sang Python
"""

import cv2
import numpy as np
import os
import json
import time
import threading
import queue
from typing import List, Dict, Tuple

# Test Rust backend
try:
    import face_recognition_rust

    RUST_AVAILABLE = True
    print("✅ Rust backend loaded successfully!")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Rust backend not available. Using Python-only mode.")

# Test face_recognition library
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Face recognition library loaded successfully!")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  Face recognition library not available.")


class AsyncLogger:
    """Non-blocking logger that runs in separate thread"""

    def __init__(self):
        self.log_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()

    def _log_worker(self):
        """Worker thread for processing log messages"""
        while self.running:
            try:
                message = self.log_queue.get(timeout=0.1)
                if message is None:  # Shutdown signal
                    break
                print(message)
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def log(self, message: str):
        """Add message to queue for async logging"""
        if not self.log_queue.full():
            self.log_queue.put(message)

    def shutdown(self):
        """Shutdown the logger"""
        self.running = False
        self.log_queue.put(None)
        self.thread.join(timeout=1.0)


# Global async logger
async_logger = AsyncLogger()


class HybridFaceRecognitionSystem:
    """Hybrid face recognition system using Rust + Python"""

    def __init__(self, database_path: str = "hybrid_face_database.json", use_rust: bool = True):
        self.database_path = database_path
        self.use_rust = use_rust and RUST_AVAILABLE

        # Always initialize Python arrays for fallback
        self.known_face_encodings = []
        self.known_face_names = []

        # Initialize components
        if self.use_rust:
            self.rust_recognizer = face_recognition_rust.SimpleFaceRecognizer(0.6)
            self.rust_timer = face_recognition_rust.PerformanceTimer()
            print("🦀 Using Rust backend for recognition")
        else:
            print("🐍 Using Python-only mode")

        # OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.load_database()

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def add_known_person(self, name: str, image_path: str) -> bool:
        """Add a new person to the database"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detect_faces(image)
        if not faces:
            print(f"❌ No faces detected in image: {image_path}")
            return False

        if len(faces) > 1:
            print(f"⚠️  Multiple faces detected, using the largest one")

        # Use largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        if not FACE_RECOGNITION_AVAILABLE:
            # Simple pixel-based encoding (fallback)
            face_roi = rgb_image[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (64, 64))  # Standardize size
            face_encoding = face_roi.flatten().astype(np.float64) / 255.0  # Normalize

            print(f"⚠️  Using simple pixel-based encoding (length: {len(face_encoding)})")
        else:
            # Use face_recognition for proper encoding
            face_locations = [(y, x + w, y + h, x)]  # Convert to face_recognition format
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if not face_encodings:
                print(f"❌ Failed to encode face in image: {image_path}")
                return False

            face_encoding = face_encodings[0]
            async_logger.log(f"✅ Generated proper face encoding (length: {len(face_encoding)})")

        # Add to database
        if self.use_rust:
            try:
                self.rust_recognizer.add_known_face(face_encoding.tolist(), name)
                async_logger.log(f"✅ Added {name} to Rust recognizer")
            except Exception as e:
                async_logger.log(f"❌ Rust error: {e}")
                self.use_rust = False

        # Always add to Python arrays for fallback/consistency
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        async_logger.log(f"✅ Added {name} to Python recognizer")

        self.save_database()
        return True

    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Recognize faces in image - optimized for speed"""
        results = []

        # Performance timing (only in debug mode)
        start_time = time.time()
        if self.use_rust:
            self.rust_timer.start()

        # Detect faces (resize image for faster detection)
        small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # Half size for detection
        small_faces = self.detect_faces(small_image)

        # Scale back coordinates
        faces = [(int(x * 2), int(y * 2), int(w * 2), int(h * 2)) for (x, y, w, h) in small_faces]

        if not faces:
            return results

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process each face
        for face_location in faces:
            x, y, w, h = face_location

            if not FACE_RECOGNITION_AVAILABLE:
                # Simple pixel-based matching
                face_roi = rgb_image[y : y + h, x : x + w]
                face_roi = cv2.resize(face_roi, (64, 64))
                face_encoding = face_roi.flatten().astype(np.float64) / 255.0
                name = "Unknown"  # Simple mode doesn't support recognition
                confidence = 0.0
            else:
                # Proper face recognition
                face_locations_fr = [(y, x + w, y + h, x)]
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations_fr)

                if not face_encodings:
                    name = "Unknown"
                    confidence = 0.0
                else:
                    face_encoding = face_encodings[0]

                    if self.use_rust and self.rust_recognizer.get_known_faces_count() > 0:
                        # Use Rust recognizer
                        try:
                            rust_result = self.rust_recognizer.recognize_face(
                                face_encoding.tolist()
                            )
                            name = rust_result if rust_result else "Unknown"
                            confidence = 0.95 if rust_result else 0.0
                        except Exception as e:
                            print(f"Rust recognition error: {e}")
                            name = "Unknown"
                            confidence = 0.0
                    else:
                        # Python recognition
                        if self.known_face_encodings:
                            matches = face_recognition.compare_faces(
                                self.known_face_encodings, face_encoding
                            )
                            name = "Unknown"
                            confidence = 0.0

                            if True in matches:
                                first_match_index = matches.index(True)
                                name = self.known_face_names[first_match_index]
                                confidence = 0.95
                        else:
                            name = "Unknown"
                            confidence = 0.0

            results.append(
                {
                    "name": name,
                    "location": face_location,
                    "confidence": confidence,
                    "backend": "Rust" if self.use_rust else "Python",
                }
            )

        # Performance measurement (simplified) - async logging
        total_time = time.time() - start_time
        if self.use_rust and total_time > 0.1:  # Only log if slow
            try:
                rust_time = self.rust_timer.stop()
                async_logger.log(f"🐌 Slow frame: Rust {rust_time:.4f}s, Total {total_time:.4f}s")
            except:
                pass

        return results

    def process_video_stream(self, camera_index: int = 0, display_height: int = 600):
        """Process real-time video stream with async logging"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        # Set camera resolution and FPS for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better FPS
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

        backend_text = "Rust" if self.use_rust else "Python"
        print(f"🎥 Starting video stream with {backend_text} backend...")
        print(f"📺 Camera: 640x480 @ 30fps target")
        print(f"📺 Display height: {display_height}px (press '+' to increase, '-' to decrease)")
        print("🚀 Performance optimized for high FPS with async logging")
        print("Controls:")
        print("  q - quit")
        print("  s - save frame")
        print("  space - pause/resume")
        print("  r - toggle recognition")
        print("  t - show performance stats")
        print("  + - increase display size")
        print("  - - decrease display size")
        print("  f - toggle fast mode (skip more frames)")

        paused = False
        recognize_mode = True
        fast_mode = False  # Skip more frames in fast mode
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        fps = 0

        # Cache for recognition results to avoid recomputing
        results_cache = []
        last_recognition_frame = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                fps_counter += 1

            display_frame = frame.copy()

            # Resize frame for display (maintain aspect ratio)
            original_height, original_width = display_frame.shape[:2]
            if original_height > display_height:
                scale_factor = display_height / original_height
                new_width = int(original_width * scale_factor)
                display_frame = cv2.resize(display_frame, (new_width, display_height))

            # Scale factor for coordinates (for drawing rectangles)
            scale_x = display_frame.shape[1] / original_width
            scale_y = display_frame.shape[0] / original_height

            # Process frames based on mode with caching
            skip_frames = 10 if fast_mode else 5
            if recognize_mode and frame_count % skip_frames == 0:
                results_cache = self.recognize_faces(frame)
                last_recognition_frame = frame_count

                # Draw results
                for result in results_cache:
                    x, y, w, h = result["location"]
                    name = result["name"]
                    confidence = result["confidence"]
                    backend = result.get("backend", "Unknown")

                    # Scale coordinates for display
                    display_x = int(x * scale_x)
                    display_y = int(y * scale_y)
                    display_w = int(w * scale_x)
                    display_h = int(h * scale_y)

                    # Choose color based on recognition
                    if name != "Unknown":
                        color = (0, 255, 0) if backend == "Rust" else (0, 180, 255)
                    else:
                        color = (0, 0, 255)

                    # Draw rectangle
                    cv2.rectangle(
                        display_frame,
                        (display_x, display_y),
                        (display_x + display_w, display_y + display_h),
                        color,
                        2,
                    )

                    # Draw label
                    if confidence > 0:
                        label = f"{name} ({confidence:.2f}) [{backend}]"
                    else:
                        label = f"{name} [{backend}]"

                    cv2.putText(
                        display_frame,
                        label,
                        (display_x, display_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
            elif (
                recognize_mode
                and results_cache
                and (frame_count - last_recognition_frame) < skip_frames * 3
            ):
                # Use cached results for faster display
                for result in results_cache:
                    x, y, w, h = result["location"]
                    name = result["name"]
                    confidence = result["confidence"]
                    backend = result.get("backend", "Unknown")

                    # Scale coordinates for display
                    display_x = int(x * scale_x)
                    display_y = int(y * scale_y)
                    display_w = int(w * scale_x)
                    display_h = int(h * scale_y)

                    # Choose color based on recognition (dimmer for cached)
                    if name != "Unknown":
                        color = (0, 200, 0) if backend == "Rust" else (0, 150, 200)
                    else:
                        color = (0, 0, 200)

                    # Draw rectangle
                    cv2.rectangle(
                        display_frame,
                        (display_x, display_y),
                        (display_x + display_w, display_y + display_h),
                        color,
                        1,
                    )

                    # Draw label (cached indicator)
                    if confidence > 0:
                        label = f"{name} (cached) [{backend}]"
                    else:
                        label = f"{name} (cached) [{backend}]"

                    cv2.putText(
                        display_frame,
                        label,
                        (display_x, display_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

            # Calculate FPS more frequently for better accuracy
            if fps_counter >= 15:  # Calculate every 15 frames instead of 30
                fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
            else:
                fps = fps if "fps" in locals() else 0

            # Status overlay with FPS prominently displayed
            status_parts = [
                f"🚀 FPS: {fps:.1f}" if fps > 0 else "FPS: --",
                f"Frame: {frame_count}",
                f"Backend: {backend_text}",
                f"Recognition: {'ON' if recognize_mode else 'OFF'}",
            ]

            if fast_mode:
                status_parts.append("⚡ FAST")
            if paused:
                status_parts.append("⏸️ PAUSED")

            status_text = " | ".join(status_parts)
            cv2.putText(
                display_frame,
                status_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Hybrid Face Recognition System", display_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"captured_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                async_logger.log(f"💾 Saved: {filename}")
            elif key == ord(" "):
                paused = not paused
                async_logger.log(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord("r"):
                recognize_mode = not recognize_mode
                async_logger.log(f"🔍 Recognition: {'ON' if recognize_mode else 'OFF'}")
            elif key == ord("t"):
                if self.use_rust:
                    try:
                        avg_time = self.rust_timer.get_average()
                        measurements = len(self.rust_timer.get_measurements())
                        async_logger.log(
                            f"📊 Rust Performance: {avg_time:.4f}s avg ({measurements} samples)"
                        )
                    except:
                        async_logger.log("📊 No performance data available")
            elif key == ord("+") or key == ord("="):
                display_height = min(display_height + 50, 1200)
                async_logger.log(f"📺 Display height increased to: {display_height}px")
            elif key == ord("-") or key == ord("_"):
                display_height = max(display_height - 50, 300)
                async_logger.log(f"📺 Display height decreased to: {display_height}px")
            elif key == ord("f"):
                fast_mode = not fast_mode
                mode_text = "FAST (skip 10 frames)" if fast_mode else "NORMAL (skip 5 frames)"
                async_logger.log(f"⚡ Mode switched to: {mode_text}")

        cap.release()
        cv2.destroyAllWindows()

        # Cleanup async logger when video stream ends
        async_logger.log("🎥 Video stream ended")

    def list_known_people(self) -> List[str]:
        """List all known people"""
        if self.use_rust:
            return self.rust_recognizer.get_all_names()
        else:
            return self.known_face_names.copy()

    def save_database(self):
        """Save database to file"""
        # Use Python arrays as source of truth
        encodings_list = [
            enc.tolist() if isinstance(enc, np.ndarray) else enc
            for enc in self.known_face_encodings
        ]

        data = {
            "backend": "Rust" if self.use_rust else "Python",
            "names": self.known_face_names,
            "encodings": encodings_list,
            "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.0-hybrid",
        }

        with open(self.database_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        async_logger.log(f"💾 Saved {len(self.known_face_names)} people to database")

    def load_database(self):
        """Load database from file"""
        if not os.path.exists(self.database_path):
            print("📝 No existing database found")
            return

        try:
            with open(self.database_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            backend = data.get("backend", "Unknown")
            names = data.get("names", [])
            encodings = data.get("encodings", [])

            print(f"📚 Found database with {len(names)} people")
            print(f"💾 Last backend used: {backend}")
            print(f"📅 Last saved: {data.get('saved_at', 'unknown')}")

            # Load into appropriate backend
            if self.use_rust and encodings:
                try:
                    for name, encoding in zip(names, encodings):
                        self.rust_recognizer.add_known_face(encoding, name)
                    print(f"✅ Loaded {len(names)} faces into Rust backend")
                except Exception as e:
                    print(f"⚠️  Error loading into Rust backend: {e}")
                    self.use_rust = False

            # Always load into Python arrays for fallback
            if encodings and FACE_RECOGNITION_AVAILABLE:
                self.known_face_names = names[:]
                self.known_face_encodings = [np.array(enc) for enc in encodings]
                print(f"✅ Loaded {len(names)} faces into Python backend")

        except Exception as e:
            print(f"⚠️  Error loading database: {e}")


def main():
    """Main application with hybrid backend"""
    print("🎯 Hybrid Face Recognition System")
    print("=" * 50)
    print(f"🦀 Rust Backend: {'Available' if RUST_AVAILABLE else 'Not Available'}")
    print(f"🔍 Face Recognition: {'Available' if FACE_RECOGNITION_AVAILABLE else 'Not Available'}")
    print()

    system = HybridFaceRecognitionSystem()

    while True:
        print("\n📋 Hybrid Menu:")
        print("1. 🎥 Start video stream")
        print("2. 👤 Add person from image")
        print("3. 👥 List known people")
        print("4. 🧪 Test camera")
        print("5. ⚡ Performance test")
        print("6. ❌ Exit")

        choice = input("\n👉 Choose (1-6): ").strip()

        if choice == "1":
            system.process_video_stream()

        elif choice == "2":
            image_path = input("📂 Image path: ").strip()
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]

            name = input("👤 Person name: ").strip()

            if image_path and name:
                system.add_known_person(name, image_path)
            else:
                print("❌ Please provide both path and name")

        elif choice == "3":
            people = system.list_known_people()
            if people:
                backend = "Rust" if system.use_rust else "Python"
                print(f"\n👥 Known people ({len(people)}) - {backend} backend:")
                for i, name in enumerate(people, 1):
                    print(f"  {i}. {name}")
            else:
                print("\n👥 No people in database")

        elif choice == "4":
            print("🧪 Testing camera...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✅ Camera working!")
                    cv2.imshow("Camera Test - Press any key", frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("❌ Cannot capture frames")
                cap.release()
            else:
                print("❌ Cannot open camera")

        elif choice == "5":
            if RUST_AVAILABLE:
                print("⚡ Running performance test...")

                # Generate test data
                test_encoding = [np.random.random() for _ in range(128)]

                # Test Rust performance
                rust_timer = face_recognition_rust.PerformanceTimer()

                iterations = 1000
                rust_timer.start()
                for _ in range(iterations):
                    face_recognition_rust.calculate_distance(test_encoding, test_encoding)
                rust_time = rust_timer.stop()

                print(f"🦀 Rust: {iterations} distance calculations in {rust_time:.4f}s")
                print(f"   Average: {rust_time/iterations*1000:.4f}ms per calculation")

                # Test Python performance (for comparison)
                import math

                start_time = time.time()
                for _ in range(iterations):
                    # Simple Python distance calculation
                    sum_sq = sum((a - b) ** 2 for a, b in zip(test_encoding, test_encoding))
                    math.sqrt(sum_sq)
                python_time = time.time() - start_time

                print(f"🐍 Python: {iterations} distance calculations in {python_time:.4f}s")
                print(f"   Average: {python_time/iterations*1000:.4f}ms per calculation")

                if rust_time > 0:
                    speedup = python_time / rust_time
                    print(f"🚀 Rust is {speedup:.1f}x faster than Python!")

            else:
                print("❌ Rust backend not available for performance testing")

        elif choice == "6":
            print("👋 Goodbye!")
            async_logger.shutdown()  # Properly shutdown async logger
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
