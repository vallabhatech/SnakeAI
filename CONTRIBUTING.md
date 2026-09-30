# Contributing to Snake AI

Thanks for contributing to Snake AI.

## Before you start

Please read the README and the architecture documentation so changes fit the existing design.

## Development setup

1. Fork the repository.
2. Clone your fork.
3. Create a virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Make your changes.
6. Run the project locally with `python agent.py`.
7. Verify that your change does not introduce obvious regressions.
8. Open a pull request with a clear description.

## Pull requests

Keep pull requests focused on one logical change.

A useful pull request description should explain:

- What changed.
- Why it changed.
- How it was tested.
- Any known limitations or follow-up work.

## Reinforcement-learning changes

For changes to the learning system, document relevant changes to:

- State representation.
- Action space.
- Reward function.
- Network architecture.
- Optimizer or learning rate.
- Discount factor.
- Replay memory.
- Exploration strategy.

If a change affects training behavior, include enough information for another contributor to understand why results may differ.

## Code style

Prefer small, readable functions and descriptive names. Keep the existing project structure unless there is a clear reason to change it.

Avoid committing generated caches, virtual environments, editor state, or large training artifacts unless they are intentionally part of the project.

## Reporting issues

When opening an issue, include:

- Operating system.
- Python version.
- Relevant dependency versions.
- Exact error message.
- Steps to reproduce.
- Expected behavior.
- Actual behavior.

## License

By contributing to this repository, you agree that your contributions will be licensed under the MIT License.
