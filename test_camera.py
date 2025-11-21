import cv2
import sys

def test_camera():
    print("Testing camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera with index 0")
            # Try other indices
            for i in range(1, 5):
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"SUCCESS: Camera opened with index {i}")
                    cap.release()
                    return True
                cap.release()
            print("ERROR: No camera found")
            return False

        # Test reading a frame
        ret, frame = cap.read()
        if ret:
            print("SUCCESS: Camera is working and can capture frames")
            print(f"Frame shape: {frame.shape}")
        else:
            print("WARNING: Camera opened but cannot read frames")

        cap.release()
        return True

    except Exception as e:
        print(f"ERROR: Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
