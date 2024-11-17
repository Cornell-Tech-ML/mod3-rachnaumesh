"""Entry point for the minitorch package.

It imports various submodules and provides a convenient way to access their contents.

Submodules:
- testing: Contains classes and functions for testing mathematical operations.
- fast_ops: Contains fast mathematical operations implemented in Python.
- cuda_ops: Contains mathematical operations implemented using CUDA.
- tensor_data: Contains classes for representing tensor data.
- tensor_functions: Contains functions for manipulating tensors.
- tensor_ops: Contains mathematical operations on tensors.
- scalar: Contains classes for representing scalar values.
- scalar_functions: Contains functions for manipulating scalar values.
- module: Contains classes for defining neural network modules.
- autodiff: Contains classes and functions for automatic differentiation.
- datasets: Contains classes for handling datasets.
- optim: Contains classes and functions for optimization algorithms.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403
