# Publish a private GitHub repository

This guide uses GitHub's website, so Git does not need to be installed on Windows. It prepares a clean export before selecting files for upload.

## 1. Apply the source update and check the project

Merge the preparation package into the original `bot-comment-detector` folder. The update package contains source and setup files; your existing `saved_models/`, `datasets/`, and `data/` supply the project assets.

In PowerShell, open the original project and check saved-model predictions:

```powershell
Set-Location "$env:USERPROFILE\Desktop\bot-comment-detector"
.\.venv-nlp\Scripts\python.exe verify_setup.py --smoke
```

These commands use the existing `.venv-nlp` environment. If you created the new environment described in [README.md](README.md), substitute `.venv` in the commands. Resolve missing-package, missing-resource, or model-loading errors before sharing a runnable copy. The smoke check exercises inference; it does not establish prediction accuracy or retrain the models.

## 2. Make the clean export

```powershell
.\.venv-nlp\Scripts\python.exe prepare_share.py
```

The script creates a new folder beside the original, named like `bot-comment-detector-github-YYYYMMDD-HHMMSS`. It prints the exact location, file count, total size, and a command to open it.

The export requires all eleven supplied source/setup files and includes regular files from `saved_models/`, `datasets/`, and `data/`. It checks that all six model artifacts exist and are nonempty for **English, Arabic, and multilingual** pipelines. It excludes the virtual environment, Git metadata, caches, unknown files at the project root, and linked files/directories. The original project is unchanged.

Every selected file must fit GitHub's **25 MiB browser upload limit**. If a file is too large, the script stops before creating an export. Git or Git LFS can handle an appropriate alternative workflow; see [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

For a different destination, specify a new folder outside the project:

```powershell
.\.venv-nlp\Scripts\python.exe prepare_share.py --output "$env:USERPROFILE\Desktop\bot-comment-detector-upload"
```

An existing destination is refused rather than overwritten.

## 3. Create the private repository

Open [Create a new repository](https://github.com/new) while signed in to your GitHub account:

1. Set the repository name to **`bot-comment-detector`**.
2. Choose **Private**.
3. Leave the initial README, `.gitignore`, and license options unselected; the prepared files already supply the documentation and ignore rules.
4. Create the repository.

## 4. Upload the export's contents

On the empty repository page, use **uploading an existing file**. For a repository that already has files, use **Add file → Upload files**.

Open the clean export folder and select its **contents**: root files plus the `saved_models`, `datasets`, and `data` folders that are present. Upload those contents so `README.md` and `main.py` appear directly at the repository root. Include `.gitignore` and `.gitattributes` too.

Use the export folder rather than selecting the entire original project. **A `.gitignore` does not automatically remove unwanted files from a manual browser selection.** The export has already excluded `.venv-nlp` and other local files.

Commit the upload with a message such as `Add NLP comment classification project`. If the upload is slow, use several batches while keeping the same folder structure. GitHub permits up to 100 files per browser upload; see [upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

Check that the README renders and all three language folders contain their six artifacts.

## 5. Access and share

You can access the private repository from another computer by signing in to the same GitHub account. Your friend needs a GitHub account for repository access; after the upload is ready, you can invite them through **Settings → Collaborators**. On a personal repository, collaborators can both read and push changes. See [GitHub's collaborator permissions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository).

The scripts only prepare and check local files. Creating the repository, uploading, and inviting a collaborator are separate website actions.
