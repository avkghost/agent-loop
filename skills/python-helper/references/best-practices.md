# Python Best Practices

## Code Organization
- Keep functions small and focused on a single task
- Use meaningful variable and function names
- Organize imports alphabetically and grouped by type

## Error Handling
```python
try:
    # Code that might raise an error
    result = operation()
except SpecificError as e:
    # Handle specific error
    logger.error(f"Operation failed: {e}")
except Exception as e:
    # Fallback for unexpected errors
    logger.exception("Unexpected error occurred")
```

## Type Hints
```python
from typing import List, Dict, Optional

def process_data(items: List[str], config: Optional[Dict] = None) -> int:
    """Process items and return count."""
    count = 0
    for item in items:
        if validate(item):
            count += 1
    return count
```

## Docstrings
Always include docstrings for modules, functions, and classes:

```python
def calculate_total(values: List[float]) -> float:
    """
    Calculate the sum of all values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Sum of all values
    """
    return sum(values)
```

## Testing
- Write tests alongside your code
- Aim for at least 80% code coverage
- Use descriptive test names that explain what is being tested
