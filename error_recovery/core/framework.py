"""
Error recovery framework - main framework class

Main entry point of the error-recovery data collection system. It provides an API
to collect error-recovery data for any robot task.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Type
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .error_injectors import ERROR_INJECTORS, ErrorInjector
from .framework_support import (
    create_error_injectors_for_framework,
    execute_data_collection,
    load_and_setup_config,
    resolve_task,
)


class ErrorRecoveryFramework:
    """Error-recovery data collection framework
    
    Usage:
        # 1. Create the framework
        framework = ErrorRecoveryFramework()
        
        # 2. Collect data
        framework.collect_data(
            task_class=YourTaskClass,
            episodes=1000,
            error_types=['premature_close', 'grasp_position_offset']
        )
    """
    
    def __init__(self):
        """Initialize the framework"""
        self.supported_error_types = list(ERROR_INJECTORS.keys())
        logger.info(f"[ErrorRecoveryFramework] Supported error types: {self.supported_error_types}")
    
    def collect_data(
        self,
        task_class: Optional[Type] = None,
        task_name: Optional[str] = None,
        episodes: int = 100,
        error_types: List[str] = ['premature_close'],
        error_probabilities: Optional[Dict[str, float]] = None,
        error_params: Optional[Dict[str, Dict[str, Any]]] = None,
        save_path: Optional[str] = None,
        render: bool = False,
        data_type: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Collect error-recovery data
        
        Args:
            task_class: Task class (used directly when provided)
            task_name: Task name (imported by name when task_class is not provided)
            episodes: Number of episodes to collect
            error_types: List of error types
            error_probabilities: Probability of each error type (optional)
            error_params: Extra parameters for each error type (optional, such as close_ratio for premature_close)
            save_path: Data save path (optional; default is ./error_recovery/data/{task_name})
            render: Whether to enable rendering
            data_type: Observation data types to save (overrides the default data_type)
            **kwargs: Other configuration parameters
        
        Returns:
            Collection statistics
        """
        # 1. Resolve the task class and name
        task_class, task_name = resolve_task(task_class, task_name)
        
        # 2. Set the save path
        if save_path is None:
            save_path = f"./error_recovery/data/{task_name}"
        
        # 3. Create error injectors
        error_injectors = create_error_injectors_for_framework(self, error_types, error_probabilities, error_params, logger)
        
        # 4. Load the task config
        if data_type is not None:
            kwargs['data_type'] = data_type
        config = load_and_setup_config(task_name, save_path, render, **kwargs)
        
        # 5. Run data collection
        return execute_data_collection(
            task_class, task_name, error_injectors, config, episodes, save_path, logger
        )
    
    def list_supported_error_types(self) -> List[str]:
        """List supported error types"""
        return self.supported_error_types.copy()
    
    def create_custom_injector(
        self, 
        name: str, 
        should_inject_fn: callable,
        inject_error_fn: callable,
        detect_failure_fn: callable,
        plan_recovery_fn: callable,
        probability: float = 0.5
    ) -> ErrorInjector:
        """Create a custom error injector
        
        Args:
            name: Error type name
            should_inject_fn: Function that decides whether to inject an error
            inject_error_fn: Function that injects the error
            detect_failure_fn: Function that detects failure
            plan_recovery_fn: Function that plans recovery
            probability: Injection probability
        
        Returns:
            Custom error injector
        """
        return type(
            "CustomErrorInjector",
            (ErrorInjector,),
            {
                "__init__": lambda self: ErrorInjector.__init__(self, name, probability),
                "should_inject": lambda self, context: should_inject_fn(context),
                "inject_error": lambda self, context: inject_error_fn(context),
                "detect_failure": lambda self, context: detect_failure_fn(context),
                "plan_recovery": lambda self, context: plan_recovery_fn(context),
            },
        )()


# Convenience function
def collect_error_recovery_data(
    task_name: str,
    episodes: int = 100,
    error_types: List[str] = ['premature_close'],
    save_path: Optional[str] = None,
    data_type: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function: collect error-recovery data
    
    Args:
        task_name: Task name
        episodes: Number of episodes
        error_types: List of error types
        save_path: Save path
        **kwargs: Other parameters
    
    Returns:
        Collection statistics
    """
    framework = ErrorRecoveryFramework()
    return framework.collect_data(
        task_name=task_name,
        episodes=episodes,
        error_types=error_types,
        save_path=save_path,
        data_type=data_type,
        **kwargs
    )
