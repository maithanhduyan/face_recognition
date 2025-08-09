"""
OpenCV-Only Face Detection System
Không sử dụng face_recognition library, chỉ dùng OpenCV
"""

import cv2
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple


class OpenCVFaceSystem:
    """Face detection system chỉ sử dụng OpenCV"""

    def __init__(self, database_path: str = "opencv_face_database.json"):
        self.database_path = database_path
        self.known_faces = []  # Store face images instead of encodings

        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():
            print("❌ Could not load face cascade classifier")
        else:
            print("✅ Face cascade classifier loaded successfully")

        self.load_database()
        print(f"🗄️  Database loaded with {len(self.known_faces)} known people")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def add_person(self, name: str, image_path: str) -> bool:
        """Add person to database"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return False

        faces = self.detect_faces(image)
        if not faces:
            print(f"❌ No faces detected in image: {image_path}")
            return False

        if len(faces) > 1:
            print(f"⚠️  Multiple faces detected, using the largest one")

        # Get the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Extract face region
        face_image = image[y : y + h, x : x + w]

        # Resize to standard size for consistency
        face_image = cv2.resize(face_image, (100, 100))

        # Store person data
        person_data = {
            "name": name,
            "face_data": face_image.tolist(),  # Convert to list for JSON storage
            "original_image": image_path,
            "face_location": largest_face,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.known_faces.append(person_data)
        self.save_database()

        print(f"✅ Added {name} to database")
        return True

    def recognize_basic(self, image: np.ndarray) -> List[Dict]:
        """Basic face matching using template matching"""
        faces = self.detect_faces(image)
        results = []

        for x, y, w, h in faces:
            face_roi = image[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (100, 100))

            best_match = None
            best_score = float("inf")

            # Simple template matching with all known faces
            for person in self.known_faces:
                known_face = np.array(person["face_data"], dtype=np.uint8)

                # Calculate difference (simple method)
                diff = cv2.absdiff(face_roi, known_face)
                score = np.sum(diff)

                if score < best_score:
                    best_score = score
                    best_match = person["name"]

            # Threshold for recognition (tunable)
            threshold = 1000000  # Adjust based on testing
            name = best_match if best_score < threshold else "Unknown"
            confidence = max(0, 1 - (best_score / threshold))

            results.append(
                {
                    "name": name,
                    "location": (x, y, w, h),
                    "confidence": confidence,
                    "score": best_score,
                }
            )

        return results

    def process_video(self, camera_index: int = 0):
        """Process video stream"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        print("🎥 Starting video stream...")
        print("Controls:")
        print("  q - quit")
        print("  s - save frame")
        print("  space - pause/resume")
        print("  r - recognize faces")

        paused = False
        recognize_mode = False
        frame_count = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

            display_frame = frame.copy()

            # Detect faces
            if frame_count % 3 == 0 or paused:  # Every 3rd frame for performance
                faces = self.detect_faces(frame)

                # Draw face rectangles
                for x, y, w, h in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        "Face",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Recognize if enabled
                if recognize_mode and self.known_faces:
                    results = self.recognize_basic(frame)
                    for result in results:
                        x, y, w, h = result["location"]
                        name = result["name"]
                        conf = result["confidence"]

                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                        label = f"{name} ({conf:.2f})"
                        cv2.putText(
                            display_frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            # Status overlay
            status = []
            if paused:
                status.append("PAUSED")
            if recognize_mode:
                status.append("RECOGNITION ON")

            status_text = (
                f"Frame: {frame_count} | Faces: {len(faces) if 'faces' in locals() else 0}"
            )
            if status:
                status_text += f" | {' | '.join(status)}"

            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("OpenCV Face Detection", display_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"captured_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
            elif key == ord(" "):
                paused = not paused
                print(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord("r"):
                recognize_mode = not recognize_mode
                print(f"🔍 Recognition: {'ON' if recognize_mode else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()

    def list_people(self) -> List[str]:
        """List known people"""
        return [person["name"] for person in self.known_faces]

    def remove_person(self, name: str) -> bool:
        """Remove person from database"""
        for i, person in enumerate(self.known_faces):
            if person["name"] == name:
                self.known_faces.pop(i)
                self.save_database()
                return True
        return False

    def save_database(self):
        """Save to JSON"""
        with open(self.database_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "faces": self.known_faces,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0-opencv",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def load_database(self):
        """Load from JSON"""
        if not os.path.exists(self.database_path):
            self.known_faces = []
            return

        try:
            with open(self.database_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.known_faces = data.get("faces", [])
            print(f"📅 Last saved: {data.get('saved_at', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Error loading database: {e}")
            self.known_faces = []


def main():
    """Main application"""
    print("🎯 OpenCV Face Detection System")
    print("=" * 40)
    print("🔧 Using OpenCV only (no face_recognition library)")
    print()

    system = OpenCVFaceSystem()

    while True:
        print("\n📋 Menu:")
        print("1. 🎥 Start video stream")
        print("2. 👤 Add person from image")
        print("3. 👥 List known people")
        print("4. 🗑️  Remove person")
        print("5. 🧪 Test camera")
        print("6. ❌ Exit")

        choice = input("\n👉 Choose (1-6): ").strip()

        if choice == "1":
            system.process_video()

        elif choice == "2":
            image_path = input("📂 Image path: ").strip()
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]

            name = input("👤 Person name: ").strip()

            if image_path and name:
                system.add_person(name, image_path)
            else:
                print("❌ Please provide both path and name")

        elif choice == "3":
            people = system.list_people()
            if people:
                print(f"\n👥 Known people ({len(people)}):")
                for i, name in enumerate(people, 1):
                    print(f"  {i}. {name}")
            else:
                print("\n👥 No people in database")

        elif choice == "4":
            name = input("👤 Name to remove: ").strip()
            if name and system.remove_person(name):
                print(f"✅ Removed {name}")
            else:
                print(f"❌ {name} not found")

        elif choice == "5":
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

        elif choice == "6":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
