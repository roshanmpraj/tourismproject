from huggingface_hub import HfApi
import os

# Initialize Hugging Face API with your token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload your deployment folder to your Hugging Face Space
api.upload_folder(
    folder_path="deployment",  # Local folder containing your app files
    repo_id="Roshanmpraj/tourismproject",  # Your Hugging Face Space repo ID
    repo_type="space",  # Type can be 'model', 'dataset', or 'space'
    path_in_repo="",  # Optional: subfolder inside repo (empty = root)
)

print("✅ Files uploaded successfully to Hugging Face Space!")
