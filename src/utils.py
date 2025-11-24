"""
Utility functions for KAVA evaluation.
Includes robust numerical answer extraction and exact match scoring.
Aligned with GSM8k official evaluation: https://github.com/openai/grade-school-math
"""

import re
from typing import Optional, Union, List, Tuple


def extract_answer_number(text: str) -> Optional[str]:
    """
    Extract final answer number from text.
    Aligned with GSM8k official evaluation: removes commas, units, trailing symbols.
    
    Priority order:
    1. "#### NUMBER" format (GSM8k standard)
    2. "answer is NUMBER" / "Answer: NUMBER"
    3. "boxed{NUMBER}" (LaTeX format)
    4. "= NUMBER" at end
    5. Last number in text
    
    Args:
        text: Generated answer text
    
    Returns:
        Extracted number as string, or None if not found
    
    Examples:
        >>> extract_answer_number("The answer is 42")
        '42'
        >>> extract_answer_number("#### 1,234.56")
        '1234.56'
        >>> extract_answer_number("$25.99")
        '25.99'
        >>> extract_answer_number("\\\\boxed{100}")
        '100'
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: "#### NUMBER" format (GSM8k standard)
    match = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        return normalize_number(match.group(1))
    
    # Strategy 2: "answer is/Answer:" patterns
    patterns = [
        r'[Aa]nswer\s+is[:\s]+([-+]?\d[\d,]*\.?\d*)',  # "answer is 42"
        r'[Aa]nswer[:\s]+([-+]?\d[\d,]*\.?\d*)',        # "Answer: 42"
        r'[Ff]inal\s+answer[:\s]+([-+]?\d[\d,]*\.?\d*)', # "final answer: 42"
        r'[Tt]he\s+answer[:\s]+([-+]?\d[\d,]*\.?\d*)',  # "the answer: 42"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_number(match.group(1))
    
    # Strategy 3: LaTeX boxed format
    match = re.search(r'\\boxed\{([-+]?\d[\d,]*\.?\d*)\}', text)
    if match:
        return normalize_number(match.group(1))
    
    # Strategy 4: "= NUMBER" at end
    match = re.search(r'=\s*([-+]?\d[\d,]*\.?\d*)\s*$', text)
    if match:
        return normalize_number(match.group(1))
    
    # Strategy 5: Extract last number in text
    numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*', text)
    # Filter out empty strings and strings without digits
    numbers = [n for n in numbers if n and re.search(r'\d', n)]
    if numbers:
        return normalize_number(numbers[-1])
    
    return None


def normalize_number(number_str: str) -> str:
    """
    Normalize number string by removing commas and extra zeros.
    
    Args:
        number_str: Raw number string (may contain commas, extra zeros)
    
    Returns:
        Normalized number string
    
    Examples:
        >>> normalize_number("1,234.56")
        '1234.56'
        >>> normalize_number("42.00")
        '42'
        >>> normalize_number("-0.50")
        '-0.5'
    """
    if not number_str:
        return ""
    
    # Remove commas
    number_str = number_str.replace(',', '')
    
    # Remove spaces
    number_str = number_str.strip()
    
    # Try to convert to float and back to remove trailing zeros
    try:
        num = float(number_str)
        # If it's an integer, return as int
        if num.is_integer():
            return str(int(num))
        else:
            # Remove trailing zeros in decimal part
            return str(num).rstrip('0').rstrip('.')
    except ValueError:
        # If conversion fails, return original
        return number_str


def exact_match_numeric(pred: str, gold: str, tolerance: float = 1e-3) -> bool:
    """
    Check if predicted and gold answers match numerically.
    Tolerates small floating point errors.
    
    Args:
        pred: Predicted answer text
        gold: Gold answer text
        tolerance: Tolerance for floating point comparison
    
    Returns:
        True if answers match, False otherwise
    
    Examples:
        >>> exact_match_numeric("42", "42.0")
        True
        >>> exact_match_numeric("The answer is 100", "#### 100")
        True
        >>> exact_match_numeric("99.999", "100.001")
        False
    """
    pred_num = extract_answer_number(pred)
    gold_num = extract_answer_number(gold)
    
    if pred_num is None or gold_num is None:
        return False
    
    try:
        pred_float = float(pred_num)
        gold_float = float(gold_num)
        return abs(pred_float - gold_float) < tolerance
    except (ValueError, TypeError):
        # Fall back to string comparison if conversion fails
        return pred_num == gold_num


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    Removes extra whitespace, lowercases, removes punctuation.
    
    Args:
        text: Answer text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Lowercase
    text = text.lower()
    
    # Remove common punctuation (but keep decimal points)
    text = re.sub(r'[,\$%]', '', text)
    
    return text.strip()


def calculate_accuracy(predictions: list, golds: list) -> float:
    """
    Calculate exact match accuracy.
    
    Args:
        predictions: List of predicted answers
        golds: List of gold answers
    
    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(golds):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(golds)} golds")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(exact_match_numeric(pred, gold) for pred, gold in zip(predictions, golds))
    return correct / len(predictions)


def format_metrics(metrics: dict, decimals: int = 4) -> dict:
    """
    Format metrics dictionary with consistent decimal places.
    
    Args:
        metrics: Dictionary of metric names to values
        decimals: Number of decimal places
    
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = round(value, decimals)
        else:
            formatted[key] = value
    return formatted


# Compatibility aliases for backward compatibility
extract_numerical_answer = extract_answer_number


# ==================================================
# Unit Tests
# ==================================================

def test_extract_answer_number():
    """Test answer extraction with various formats"""
    test_cases = [
        # (input, expected_output, description)
        ("#### 42", "42", "GSM8k standard format"),
        ("#### 1,234.56", "1234.56", "GSM8k with commas"),
        ("#### -123", "-123", "Negative number"),
        ("The answer is 42", "42", "Natural language: 'answer is'"),
        ("Answer: $1,500", "1500", "With dollar sign"),
        ("Final answer: 3.14", "3.14", "Final answer pattern"),
        ("\\boxed{123}", "123", "LaTeX boxed format"),
        ("x = 99", "99", "Equals sign at end"),
        ("Some text with 10 and 20 and 30", "30", "Multiple numbers - last one"),
        ("No numbers here", None, "No numbers"),
        ("", None, "Empty string"),
        ("42.000", "42", "Trailing zeros removed"),
        ("The total cost is $3,500.00", "3500", "Complex format"),
    ]
    
    print("="*70)
    print("Testing extract_answer_number()")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for text, expected, description in test_cases:
        result = extract_answer_number(text)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        # Truncate text for display
        text_display = text[:40] + "..." if len(text) > 40 else text
        print(f"  {status} {description}")
        print(f"      Input: '{text_display}'")
        print(f"      Expected: {expected}, Got: {result}")
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_exact_match_numeric():
    """Test numerical matching"""
    test_cases = [
        # (pred, gold, expected_result, description)
        ("The answer is 42", "#### 42", True, "Standard match"),
        ("Total: 1,234", "#### 1234", True, "Comma normalization"),
        ("Result: 42.001", "#### 42", False, "Outside tolerance"),
        ("Result: 42.0005", "#### 42", True, "Within tolerance"),
        ("Answer: 100", "#### 99", False, "Wrong answer"),
        ("No number", "#### 42", False, "Missing prediction"),
        ("The answer is 50", "No gold", False, "Missing gold"),
        ("$3,500.00", "#### 3500", True, "Currency format"),
        ("\\boxed{256}", "#### 256", True, "LaTeX format"),
    ]
    
    print("\n" + "="*70)
    print("Testing exact_match_numeric()")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for pred, gold, expected, description in test_cases:
        result = exact_match_numeric(pred, gold)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        pred_display = pred[:30] + "..." if len(pred) > 30 else pred
        print(f"  {status} {description}")
        print(f"      Pred: '{pred_display}' | Gold: '{gold}' | Match: {result}")
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_normalize_number():
    """Test number normalization"""
    test_cases = [
        ("1,234.56", "1234.56", "Remove commas"),
        ("42.00", "42", "Remove trailing zeros"),
        ("-0.50", "-0.5", "Negative decimal"),
        ("100", "100", "Integer unchanged"),
        ("3.14159", "3.14159", "Decimal preserved"),
    ]
    
    print("\n" + "="*70)
    print("Testing normalize_number()")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for input_str, expected, description in test_cases:
        result = normalize_number(input_str)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"  {status} {description}: '{input_str}' → '{result}' (expected: '{expected}')")
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    print("\n" + "="*70)
    print("KAVA Utils - Unit Tests")
    print("="*70 + "\n")
    
    test1_pass = test_extract_answer_number()
    test2_pass = test_exact_match_numeric()
    test3_pass = test_normalize_number()
    
    print("\n" + "="*70)
    if test1_pass and test2_pass and test3_pass:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
