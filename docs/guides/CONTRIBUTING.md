# Contributing

Welcome, team!

This document provides a concise guide on how to contribute to our project.

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

- **CI Pipeline:**  
  All PRs trigger an automated testing pipeline to validate changes before merging.  
  🔗 **For a detailed explanation of how the CICD pipeline works, refer to [`CI_PIPELINE.md`](./CICD_PIPELINE.md).**

---

## 4. Review Process

- **CI Compliance:**  
  Your PR must pass all automated tests and status checks.
- **Code Review:**  
  The code will be reviewed for submitted changes. Please address any feedback.
- **Merge:**  
  Once the PR passes all checks and has been approved, it will be merged into `main`.

---

Thank you for contributing to our project! By following these guidelines, we can maintain a high-quality codebase and ensure smooth collaboration.
