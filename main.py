import os

def capture_dataset():
    os.system("python capture_dataset.py")

def train_model():
    os.system("python train_model.py")

def recognize_live():
    os.system("python recognize_live.py")

def main():
    print("Face Recognition Project")
    print("1. Capture dataset")
    print("2. Train model")
    print("3. Recognize faces live")
    print("4. Exit")
    while True:
        choice = input("Enter your choice (1/2/3/4): ").strip()
        if choice == '1':
            capture_dataset()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_live()
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()