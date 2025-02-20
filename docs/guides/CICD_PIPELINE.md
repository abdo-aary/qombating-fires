# CI/CD Pipeline Overview

Our project uses a **self-hosted GitHub Actions CI pipeline** to automatically run tests and validate changes before merging. The pipeline is designed to optimize execution time by leveraging a **prebuilt Docker image** whenever possible, reducing setup overhead.

---

## 1. How the Pipeline Works

Every time you open or update a PR, the CI pipeline is triggered. This pipeline performs the following steps:

1. **Check Out the Code**  
   - Pulls the latest changes from your branch.

2. **Use Prebuilt Docker Image (if available)**  
   - Instead of setting up everything from scratch, the pipeline first checks if a prebuilt Docker image (`qombat_image`) exists.  
   - If found, it starts a container from this image and only **updates project files and dependencies as needed**.
   - If no image is found, a fresh environment is set up.

3. **Set Up the Environment**  
   - Configures Python (e.g., using a specified version).  
   - Installs all dependencies from `requirements.txt` (only updating missing or outdated packages to speed up execution).

4. **Run Tests**  
   - Executes all tests in the `tests/` directory using:
     ```bash
     python -m unittest discover tests
     ```
   - Runs tests only on project-specific code, ensuring external dependencies don’t interfere with results.

5. **Report Status**  
   - The test results appear as status checks on your PR.
   - ✅ **Tests pass** → The PR is eligible for merging.  
   - ❌ **Tests fail** → The PR cannot be merged until issues are resolved.

---

## 2. CI Workflow Configuration

- **Workflow Location:**  
  The CI configuration is defined in `.github/workflows/ci_self_hosted.yml`.  
  This file instructs GitHub Actions on how to run the CI tasks.

- **Test Discovery:**  
  The pipeline automatically detects all test files following the `test_*.py` naming convention within the `tests/` directory.

- **Optimized CI Execution with Docker:**  
  - If running **locally with `act`**, the pipeline uses **volume mounting** to sync local files dynamically into the container.  
  - If running **on GitHub Actions**, it copies the updated project files into the container instead.  
  - This prevents redundant full builds and speeds up testing.

---

## 3. Self-Hosted Runner

### 🖥️ What is a Self-Hosted Runner?
Our CI jobs run on a **self-hosted machine with GPU support** rather than GitHub-hosted runners.  
This means:
- Faster execution times.
- Ability to run GPU-based tests.
- Greater flexibility in environment management.

### 🚀 How It Works
1. The runner is always **listening for jobs**.
2. When a PR is submitted, it picks up the job and executes the tests.
3. The results are sent back to GitHub as status checks.

---

## 4. Debugging CI Failures

If your PR fails CI, check the following:
- **Check the logs**: Navigate to the GitHub Actions page in your repository and inspect the detailed logs.
- **Run tests locally**: Before pushing changes, always verify tests locally using:
  ```bash
  python -m unittest discover tests
  ```
- **Ensure dependencies are up to date**: If failures are due to missing packages, manually update your environment and rebuild the prebuilt Docker image if necessary:
  ```bash
  docker build -t qombat_image .
  ```

