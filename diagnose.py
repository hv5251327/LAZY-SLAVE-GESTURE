import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
    print(f"MediaPipe file: {mp.__file__}")
    print(f"Has 'solutions': {hasattr(mp, 'solutions')}")

    if hasattr(mp, 'solutions'):
        print(f"Has 'solutions.hands': {hasattr(mp.solutions, 'hands')}")
        print("[OK] Classic solutions API is available.")
    else:
        print("[INFO] 'solutions' attribute is not in MediaPipe " + str(mp.__version__))
        print("       MediaPipe 1.x / >= 0.10.15 uses mediapipe.tasks (Tasks API).")
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            print("[OK] mediapipe.tasks is available and functional!")
        except Exception as e:
            print(f"[ERROR] Cannot import mediapipe.tasks: {e}")

        print("Checking for conflicting local files:")
        found_any = False
        for f in os.listdir('.'):
            if 'mediapipe' in f.lower() and f.endswith(('.py', '.pyc')):
                print(f"   [WARNING] FOUND: {f}  <- DELETE OR RENAME THIS!")
                found_any = True
        if not found_any:
            print("   [OK] No conflicting mediapipe.py files found in current directory.")
except ImportError as e:
    print(f"[ERROR] Cannot import mediapipe: {e}")
