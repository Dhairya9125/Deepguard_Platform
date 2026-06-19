from huggingface_hub import HfApi

api = HfApi()

def search_datasets(query):
    datasets = api.list_datasets(search=query, sort="downloads", limit=5)
    print(f"--- Top datasets for '{query}' ---")
    for d in datasets:
        print(f"{d.id} (Downloads: {d.downloads})")

if __name__ == "__main__":
    search_datasets("deepfake image")
    search_datasets("deepfake audio")
    search_datasets("deepfake video")
    search_datasets("ASVspoof")
    search_datasets("FaceForensics")
