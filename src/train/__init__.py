"""Training package: Hydra entrypoint and dataset/loader construction helpers.

Contains:
    `run_experiment`: Hydra-decorated `main` that builds the data loaders and
        model from config, trains/evaluates them via a Lightning `Trainer`,
        and logs to Weights & Biases.
    `utils`: Thin convenience wrappers (`make_multiscale_dataset`,
        `make_multiscale_loader`) around `dataset.fmow_multiscale_dataset`
        used by `run_experiment` to build datasets/loaders from config values.
"""
