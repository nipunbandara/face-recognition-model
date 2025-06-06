import os

def capture_dataset():
    os.system("python capture_dataset.py")

def train_model():
    os.system("python train_model.py")

def recognize_live():
    os.system("python recognize_live.py")

def convert_to_tfjs():
    """
    Converts the Keras .h5 model to TensorFlow.js format using the tensorflowjs_converter CLI.
    """
    h5_path = input("face_rec_model.h").strip()
    tfjs_dir = input("face_rec_model_tfjs").strip()
    cmd = f"tensorflowjs_converter --input_format keras {h5_path} {tfjs_dir}"
    print(f"Running: {cmd}")
    os.system(cmd)
    print(f"Model converted and saved to {tfjs_dir}")

def main():
    print("Face Recognition Project")
    print("1. Capture dataset")
    print("2. Train model")
    print("3. Recognize faces live")
    print("4. Convert model to TensorFlow.js format")
    print("5. Exit")
    while True:
        choice = input("Enter your choice (1/2/3/4/5): ").strip()
        if choice == '1':
            capture_dataset()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_live()
        elif choice == '4':
            convert_to_tfjs()
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()