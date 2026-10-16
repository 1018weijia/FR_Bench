"""
Backward-compatible alias for legacy double-env imports.
All logic lives in deploy_policy.py and supports both local + cloud RPC.
"""

from .deploy_policy import encode_obs
from .deploy_policy import eval
from .deploy_policy import get_model
from .deploy_policy import reset_model
