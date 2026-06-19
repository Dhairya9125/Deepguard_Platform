from datasets import load_dataset
import sys

def test_dataset(name):
    print(f"\nTesting {name} ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for i, item in enumerate(ds):
            print(f"Keys: {item.keys()}")
            print(f"Item: {item}")
            break
        return True
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return False

if __name__ == "__main__":
    test_dataset("UniDataPro/deepfake-videos-dataset")
    test_dataset("angads24/deepfake-video")
