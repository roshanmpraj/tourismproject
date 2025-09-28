from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize API
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "Roshanmpraj/tourismproject"
repo_type = "space"

# Attempt to find deployment folder
possible_paths = [
    "tourism_project/deployment",
    "deployment",
    os.path.join(os.getcwd(), "deployment")
]

folder_path = next((p for p in possible_paths if os.path.isdir(p)), None)
if folder_path is None:
    raise ValueError("Deployment folder not found. Make sure it exists in your repo.")

print(f"✅ Using deployment folder: {folder_path}")

# Step 1: Check if the Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new Space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload folder
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)

print("✅ Files uploaded successfully to Hugging Face Space!")
