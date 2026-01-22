# CIFAR-10 Classification with Testing (Unit / Integration / Regression) + CI/CD Pipeline

This repository is an end-to-end PyTorch project for CIFAR-10-style image classification, designed to demonstrate **professional testing practices for ML code**:
- **Unit tests** for individual components (data, model, utils, inference, training step)
- **Integration tests** that validate modules work together (pipeline smoke tests)
- **Regression (snapshot) tests** that detect unintended behavior changes
- **CI (GitHub Actions)** that runs tests automatically on every push / pull request
- **CD (Github Actions)** that 

> Goal: show how to test ML code *reliably* without flaky “accuracy threshold” tests.

---

## Project Structure

```src/
- data.py # dataloaders + transforms
- model.py # build_model()
- train.py # train_one_epoch(), evaluate()
- infer.py # predict(), predict_proba()
- utils.py # seed + checkpoint helpers
- main_train.py # example training entrypoint

- tests/
- unit/ # fast, isolated tests
- integration/ # pipeline smoke tests
- regression/ # snapshot/golden tests
- assets/ # frozen inputs/outputs for regression tests

- .github/workflows/
- tests.yml # CI pipeline (pytest)
- ci.ymal # CI pipeline (automatic testing)
```

---

## Setup

### Option A: install in editable mode (recommended)
From the project root:

```bash
pip install -e .
pip install pytest torch torchvision
```

### Runing Training
```
python -m src.main_train
```

### Running Tests

```
pytest -q
```

- Run specific test groups:
  ```
  - pytest -q -m unit
  - pytest -q -m integration
  - pytest -q -m regression
  ```
- Run a single test file:
```
pytest -q tests/unit/test_model.py
```
## CI (Continuous Integration)
**Trigger**: Runs on every push or pull request to the main branch.
 - Steps:

   - Checkout: Retrieves the latest code.

   - Setup: Installs dependencies required for building and testing.

   - Build: Compiles or prepares the application.

   - Test: Runs automated tests to verify that the code works as expected.

## CD (Continuous Deployment):
**Trigger**: Runs automatically after successful CI on the main branch.
**Steps**: 
   - Build Docker Image: Packages your project into a Docker image.
   - Push Image to Registry: Uploads the image to Docker Hub.


 - every push
 - every pull request
 - 
Workflow file: `.github/workflows/tests.yml`
               `.github/workflows/ci.yml`




