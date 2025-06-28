"""
Chain-of-Thought (CoT) instructions for different contradiction detection tasks
These instructions are to be added to the prompt when `cot_instruction` is set to `True`.
"""

COT_DETECT_PROMPT = """\
Think step by step before answering.
"""

COT_TYPE_PROMPT = """
5. Think step by step before answering.
"""

COT_GUIDED_SEGMENT_PROMPT = """\
5. Think step by step before answering.
"""

COT_BLIND_SEGMENT_PROMPT = """\
4. Think step by step before answering.
"""

EMPTY_COT_PROMPT = ""