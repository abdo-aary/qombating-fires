# Contributing

Welcome, team!

This document provides a concise guide on how to contribute to our project and explains how our CI pipeline works.

---

## 1. Branching & Merging

- **Feature Branches:**  
  Create a new branch for each feature or bug fix.  
  Example: `feature/my-new-feature` or `bugfix/fix-issue`.

- **Pull Requests (PRs):**  
  Submit a PR to merge your feature branch into the protected `main` branch.  
  **Note:** Direct pushes to `main` are blocked by branch protection rules.  
  Your PR must pass all required status checks before it can be merged.

---

## 2. Code Standards

- **Style:**  
  Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines for Python code.

- **Commit Messages:**  
  Use clear, descriptive commit messages that explain your changes.

---

## 3. Testing

- **Local Testing:**  
  - Place your tests in the `tests/` directory.  
  - Follow the naming convention `test_*.py` so that they are automatically discovered.  
  - Run tests locally with:
    ```bash
    python -m unittest discover tests
    ```

- **CI Pipeline & Status Checks:**  
  Every time you open or update a PR, our GitHub Actions CI pipeline is triggered. This pipeline:
  - **Checks Out the Code:** Pulls the latest changes from your branch.
  - **Sets Up the Environment:** Configures Python (e.g., using a specified version) and installs all dependencies from `requirements.txt`.
  - **Runs Tests:** Executes all tests in the `tests/` directory using the command:
    ```bash
    python -m unittest discover tests
    ```
  - **Reports Status:** The results of these tests appear as status checks on your PR.  
    **Important:** The branch protection rules require that these status checks pass before merging.  
    If tests fail, the PR cannot be merged until the issues are resolved.

---

## 4. CI Pipeline Details

- **Workflow Location:**  
  The CI configuration is defined in `.github/workflows/ci_self_hosted.yml` (or similar).  
  This file instructs GitHub Actions on how to run the CI tasks.

- **Self-Hosted Runner:**  
  Our CI jobs run on a self-hosted runner with GPU support.  
  This runner listens for jobs continuously, so when a PR is submitted, it picks up the job and runs the tests.

- **Test Discovery:**  
  The status checks are performed by running `python -m unittest discover tests`, which recursively discovers all files matching `test*.py` in the `tests/` directory.  
  Ensure your test files follow this naming convention to be automatically included.

---

## 5. Review Process

- **CI Compliance:**  
  Your PR must pass all automated tests and status checks.
- **Code Review:**  
  The code will be reviewed for submitted changes. Please address any feedback.
- **Merge:**  
  Once the PR passes all checks and has been approved, it will be merged into `main`.

---

Thank you for contributing to our project! By following these guidelines, we can maintain a high-quality codebase and ensure smooth collaboration.
