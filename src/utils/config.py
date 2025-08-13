import os
import re
from typing import Any

_env_var_pattern = re.compile(r"\$\{([^}]+)\}")


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match):
            var = match.group(1)
            return os.getenv(var, "")
        return _env_var_pattern.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value