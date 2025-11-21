def test_confidence_logic():
    print("Testing confidence threshold logic:")
    
    test_cases = [
        (64.34, 50.00, "Should mark attendance"),
        (64.34, 50.01, "Should mark attendance"), 
        (64.34, 49.99, "Should NOT mark attendance"),
        (50.00, 50.00, "Should mark attendance (exactly at threshold)"),
        (49.99, 50.00, "Should NOT mark attendance"),
    ]
    
    for face_conf, gesture_conf, expected in test_cases:
        result = (face_conf >= 50) and (gesture_conf >= 50)
        status = "✅ PASS" if result == ("Should mark attendance" in expected) else "❌ FAIL"
        print(f"{status} | Face: {face_conf:.2f}%, Gesture: {gesture_conf:.2f}% | {expected} | Result: {result}")

if __name__ == "__main__":
    test_confidence_logic()
