# Kernel only GP

Contains a very minimal python package to do GP inference given only kernel matrices (not underlying data points).
Potentially useful if working with data that cannot be stored as a tensor.

## Development

### Installation

For now, just add the project's directory to the `PYTHONPATH`.

### Formatting

Be sure to install pre-commit hooks:

```bash
pre-commit install
```

### Testing

Minimal tests can be run with:

```bash
python -m pytest
```

## Future to-do items

- Convert implementation to Jax for autodiff?
- Include classes for a fixed GP (e.g. for inference)
- Make into a proper package
