from datasets import load_dataset

def test_dataset(name, modality="image"):
    print(f"\nTesting {name} ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for i, item in enumerate(ds):
            print(f"Keys: {item.keys()}")
            print(f"Item: {item}")
            if i >= 0: # just get first
                break
    except Exception as e:
        print(f"Error loading {name}: {e}")

if __name__ == "__main__":
    test_dataset("garystafford/deepfake-audio-detection", "audio")
    test_dataset("insanescw/20K_real_and_deepfake_images", "image")
    test_dataset("belkhir-nacim/deepfake-videos", "video")
