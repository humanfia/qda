"""qda-kl: score a quantized checkpoint by its KL divergence from the original model, in minutes.

Build a reference (the original's answers and top-k distributions) once, then score any checkpoint against it.
"""

__version__ = "0.1.0"
