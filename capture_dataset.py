import cv2
import os

# --- USER INPUT ---
person_name = input("Enter the person's name: ").strip()
save_dir = f"dataset/{person_name}"
os.makedirs(save_dir, exist_ok=True)
num_images = int(input("How many images to capture? (e.g., 50): "))

cap = cv2.VideoCapture(0)
img_count = 0

print("Press 's' to save an image, 'q' to quit.")

while img_count < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow("Capture - Press 's' to save, 'q' to quit", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        img_path = os.path.join(save_dir, f"{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        img_count += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {img_count} images for {person_name}.")