# Contributing to Paldo ATLM

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to **Paldo ATLM** (Automated Trading Learning Machine). These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by the [Paldo ATLM Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### 🐛 Reporting Bugs

This section guides you through submitting a bug report for Paldo ATLM. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

* **Check the [Issues](https://github.com/yourusername/Paldo-ATLM/issues) tab** to see if the problem has already been reported.
* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps to reproduce the problem** in as many details as possible.
* **Include your environment details:**
    * OS version (e.g., Windows 10/11)
    * Python version (e.g., 3.9.5)
    * MetaTrader 5 version
    * Broker being used (Demo/Real)

### 💡 Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Paldo ATLM, including completely new features and minor improvements to existing functionality.

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Explain why this enhancement would be useful** to most Paldo ATLM users.

### 💻 Pull Requests (Code Contributions)

1.  **Fork the repo** and create your branch from `main`.
2.  **Ensure your code adheres to the existing style.** We use standard Python PEP 8 conventions.
3.  **Sanitize your code.** Ensure **NO** personal API keys, Telegram tokens, or account numbers are included in your commits.
4.  **Test your changes.** If you've added code that should be tested, add tests. Ensure your changes don't break the existing trading logic.
5.  **Issue that Pull Request!**

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature").
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
* Limit the first line to 72 characters or less.
* Reference issues and pull requests liberally after the first line.

### Python Styleguide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
* Use clear variable names (e.g., `current_price` instead of `cp`).
* Comment complex logic, especially within the `ZenithBrain` and `OmniMindStrategy` classes.

## Financial Disclaimer

By contributing to this repository, you acknowledge that this software is for educational purposes only. Any code you contribute must not maliciously handle user funds or execute trades designed to cause intentional loss.

## Questions?

Feel free to open an issue with the tag `question` if you have any doubts or need clarification on the codebase.
