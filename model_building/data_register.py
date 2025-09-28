from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Roshanmpraj/Tourism"    # please create your space and repository

repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
<<<<<<< HEAD
    folder_path="data",
=======
<<<<<<< HEAD
    folder_path="data",
=======
    folder_path="data",
>>>>>>> 56d2eae19a1dd642ded8778766eddace54891bb0
>>>>>>> 0d7e26c74894916eadade37083f9733b68e8f8a1
    repo_id=repo_id,
    repo_type=repo_type,
)
