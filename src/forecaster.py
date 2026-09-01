"""
forecaster.py
=============
STATUS: not yet built. Built out in notebooks/03_forecaster.ipynb (Stage 3).

Will contain:
    - ProbabilisticLSTM (torch.nn.Module) -- outputs (mean, log_variance)
    - gaussian_nll_loss(mean, log_var, target)
    - train_forecaster(...) / evaluate_calibration(...)

See notebooks/00_overview.ipynb, Section 5 for the Gaussian NLL formula
this module implements, and why mean+variance (not a point forecast) is
required by the planner in agent_planner.py.
"""

raise NotImplementedError(
    "forecaster.py is a Stage 3 placeholder -- see notebooks/03_forecaster.ipynb"
)
