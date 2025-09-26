from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
<<<<<<< HEAD
    folder_path="tourism_project/deployment",     # the local folder containing your files
=======
    folder_path="deployment",     # the local folder containing your files
>>>>>>> 56d2eae19a1dd642ded8778766eddace54891bb0
    # replace with your repoid
    repo_id="Roshanmpraj/tourismproject",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
