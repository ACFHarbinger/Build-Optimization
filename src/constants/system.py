"""
System-wide constants and operation mappings.

This module provides infrastructure constants for:
- Multi-core processing synchronization
- File system operation enumerations
- Dynamic operator evaluation for GUI/config parsing

OPERATION_MAP Usage Context
---------------------------
The OPERATION_MAP enables runtime evaluation of operator strings in:
1. **GUI widgets**: Users can apply math operations (+=, *=, etc.) to config values
2. **Config file processing**: Dynamic value transformations during Hydra composition
3. **File system automation**: Conditional logic for batch file operations

Example usage:
    >>> op_func = OPERATION_MAP["+="]
    >>> result = op_func(10, 5)  # Returns 15
    >>>
    >>> # In GUI: User selects "+=" and enters value 5
    >>> current_value = 100
    >>> new_value = OPERATION_MAP[operation](current_value, user_input)

Supported operators:
- **Assignment**: =, (empty string for passthrough)
- **Arithmetic**: +, -, *, /, //, %, **, @, divmod
- **Comparison**: ==, !=, <, <=, >, >=
- **Bitwise**: <<, >>, |, &, ^
- **Identity/Membership**: is, isnot, in, notin
- **Compound assignment**: +=, -=, *=, /=, //=, %=, **=, <<=, >>=, |=, &=, ^=, @=

Thread Safety Note
------------------
LOCK_TIMEOUT is a module-level global that is modified by update_lock_wait_time().
This is safe for parallel data generation workers since each subprocess gets a
separate copy of the module globals.
"""

from typing import Any, Callable, Dict, List, Optional

# Multi-core processing settings
# Base timeout (milliseconds) for file/resource locking in single-core mode
CORE_LOCK_WAIT_TIME: int = 100

# Dynamic timeout (milliseconds) scaled by CPU count. Modified by update_lock_wait_time().
# In multiprocessing contexts, this prevents deadlocks when many workers contend for files.
LOCK_TIMEOUT: int = CORE_LOCK_WAIT_TIME


def update_lock_wait_time(num_cpu_cores: Optional[int] = None) -> int:
    """
    Updates the global LOCK_TIMEOUT based on the number of CPU cores.

    Args:
        num_cpu_cores: Number of CPU cores to scale the timeout by.

    Returns:
        The new (or default) value of LOCK_TIMEOUT.
    """
    global LOCK_TIMEOUT
    global CORE_LOCK_WAIT_TIME
    LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME if num_cpu_cores is None else CORE_LOCK_WAIT_TIME * num_cpu_cores
    return LOCK_TIMEOUT


# File system settings
# GUI confirmation dialog timeout (seconds). After this duration, destructive
# operations (delete, update) are auto-cancelled if user does not respond.
CONFIRM_TIMEOUT: int = 30

# Supported file system operations in GUI. Maps to gui/src/tabs/file_system/ handlers.
# - create: New file/directory generation
# - read: File content display
# - update: In-place modification
# - delete: Permanent removal (triggers confirmation dialog)
# - cryptography: Encrypt/decrypt data files
FS_COMMANDS: List[str] = ["create", "read", "update", "delete", "cryptography"]

# Operator string → callable function mapping. Enables runtime evaluation of
# operator symbols from user input or config files without using eval().
# Used primarily in GUI widgets for dynamic value transformations.
OPERATION_MAP: Dict[str, Callable[[Any, Any], Any]] = {
    # Assignment & passthrough
    "=": lambda x, y: y,  # Replace: return y regardless of x
    "": lambda x, y: x,  # Identity: return x unchanged (used for validation-only operations)
    # Arithmetic operations
    "+": lambda x, y: x + y,  # Addition
    "+=": lambda x, y: x + y,  # Add-assign (GUI: increment current value)
    "-": lambda x, y: x - y,  # Subtraction
    "-=": lambda x, y: x - y,  # Subtract-assign (GUI: decrement current value)
    "*": lambda x, y: x * y,  # Multiplication
    "*=": lambda x, y: x * y,  # Multiply-assign (GUI: scale current value)
    "/": lambda x, y: x / y,  # True division (returns float)
    "/=": lambda x, y: x / y,  # Divide-assign
    "**": lambda x, y: x**y,  # Exponentiation
    "**=": lambda x, y: x**y,  # Power-assign
    "//": lambda x, y: x // y,  # Floor division (returns int)
    "//=": lambda x, y: x // y,  # Floor-divide-assign
    "%": lambda x, y: x % y,  # Modulo
    "%=": lambda x, y: x % y,  # Modulo-assign
    "@": lambda x, y: x @ y,  # Matrix multiplication (NumPy/PyTorch)
    "@=": lambda x, y: x @ y,  # Matrix-multiply-assign
    "divmod": lambda x, y: divmod(x, y),  # Returns (quotient, remainder) tuple
    # Comparison operations (return bool)
    "==": lambda x, y: x == y,  # Equality
    "!=": lambda x, y: x != y,  # Inequality
    "<": lambda x, y: x < y,  # Less than
    "<=": lambda x, y: x <= y,  # Less than or equal
    ">": lambda x, y: x > y,  # Greater than
    ">=": lambda x, y: x >= y,  # Greater than or equal
    # Bitwise operations (for integer flags)
    "<<": lambda x, y: x << y,  # Left shift
    "<<=": lambda x, y: x << y,  # Left-shift-assign
    ">>": lambda x, y: x >> y,  # Right shift
    ">>=": lambda x, y: x >> y,  # Right-shift-assign
    "|": lambda x, y: x | y,  # Bitwise OR
    "|=": lambda x, y: x | y,  # OR-assign
    "&": lambda x, y: x & y,  # Bitwise AND
    "&=": lambda x, y: x & y,  # AND-assign
    "^": lambda x, y: x ^ y,  # Bitwise XOR
    "^=": lambda x, y: x ^ y,  # XOR-assign
    # Identity and membership tests (return bool)
    "is": lambda x, y: x is y,  # Object identity (same memory address)
    "isnot": lambda x, y: x is not y,  # Object non-identity
    "in": lambda x, y: x in y,  # Membership (x is element of y)
    "notin": lambda x, y: x not in y,  # Non-membership
}
