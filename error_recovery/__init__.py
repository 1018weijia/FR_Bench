"""
Universal error-recovery data collection system

This module provides a general framework that can automatically inject errors
into any robot task and collect recovery data.

Main components:
- ErrorInjector: error injector base class and concrete implementations
- TaskWrapper: task wrapper that automatically intercepts and modifies task execution
- DataCollector: unified data collection interface
- ErrorRecoveryFramework: main framework class

Usage:
    from error_recovery import ErrorRecoveryFramework
    
    # Create the framework
    framework = ErrorRecoveryFramework()
    
    # Collect error-recovery data for any task
    framework.collect_data(
        task_class=YourTaskClass,
        episodes=1000,
        error_types=['premature_close', 'grasp_slip', 'grasp_position_offset']
    )
"""

from .core.error_injectors import *
from .core.task_wrapper import TaskWrapper
from .core.data_collector import DataCollector
from .core.framework import ErrorRecoveryFramework

__version__ = "1.0.0"
__all__ = [
    'ErrorRecoveryFramework',
    'TaskWrapper', 
    'DataCollector',
    'ErrorInjector',
    'PrematureCloseInjector',
    'PositionOffsetInjector',
    'TimingDelayInjector',
    'RotationErrorInjector',
    'GraspForceInjector'
]
