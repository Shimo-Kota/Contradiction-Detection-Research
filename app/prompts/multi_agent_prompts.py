EVIDENCE_COLLECT_PROMPT = """
# Role
You are an Evidence Collector Agent.

Extract up to five text snippets that are relevant for deciding whether the set of documents is contradictory. For each snippet, state the document number and quote the minimal phrase or sentence.

Use the following XML format for your response:
```
<evidence>
Doc1: "actual quote from document 1"
Doc3: "actual quote from document 3"
...
</evidence>
```
If you find no passages that hint at contradiction, output:
```
<evidence>none</evidence>
```
Do not add any explanation outside the tag.

# Documents:
{docs}
"""


MULTI_AGENT_DETECT_PROMPT = """
# Role
You are a Reviewer Agent. Decide whether the documents contain conflicting information.

You have access to:
1. The <evidence> list produced earlier in this chat.
2. The full documents (below).

# Output
Provide your assessment in one of these two formats:
```
<conflict>yes</conflict>
```
or
```
<conflict>no</conflict>
```
Optionally add one short sentence with the decisive evidence after the tag.

{cot_instruction}
# Documents:
{docs}
"""

EVIDENCE_COLLECTOR_DETECT_PROMPT = EVIDENCE_COLLECT_PROMPT

MULTI_AGENT_TYPE_PROMPT = """
# Role
You are an Evidence Collector. Your task is to identify evidence relevant for determining the type of contradiction present in the documents, if any.

# Contradiction Types 
1. Self-Contradiction: Conflicting information within a single document.
2. Pair Contradiction: Conflicting information between two documents.
3. Conditional Contradiction: Three documents where the third document makes the first two contradict each other.

# Task
- Carefully examine all documents for potential contradictions:
  - Self-Contradiction: Look for statements within the SAME document that directly oppose each other
  - Pair-Contradiction: Look for statements in DIFFERENT documents that directly contradict each other
  - Conditional-Contradiction: Look for statements in two documents that become contradictory because of information in a third document

- For each evidence item include:
  - Document number(s)
  - Exact contradicting quotes
  - Explanation of how they contradict each other
  - The contradiction type it suggests ([SELF], [PAIR], or [CONDITIONAL])

# Format
Format your evidence within these tags:
<evidence>
[SELF] Doc1: "quote from document" [contradicts itself: "contradicting quote from same document"]

[PAIR] Doc1: "quote from document" [contradicts Doc2: "contradicting quote from other document"]

[CONDITIONAL] Doc1: "quote" + Doc2: "quote" [contradictory because Doc3: "rule creating contradiction"]
</evidence>

If no contradictions found: <evidence>none</evidence>

# Documents:
{docs}
"""

EVIDENCE_COLLECTOR_TYPE_PROMPT = MULTI_AGENT_TYPE_PROMPT

MULTI_AGENT_TYPE_REVIEW_PROMPT = """
# Role
You are an Reviewer Agent. Given a set of documents with a possible contradiction, your task is to predict the type of contradiction present, if any.

You have access to:
1. The <evidence> list produced earlier in this chat.
2. The full documents (below).

# Contradiction Types
1. Self-Contradiction: Conflicting information within a single document.
2. Pair Contradiction: Conflicting information between two documents.
3. Conditional Contradiction: Three documents where the third document makes the first two contradict each other.

# Detailed Explanation of Contradiction Types:
- Self-Contradiction: 
  - Occurs when a single document contains statements that cannot both be true
  - Look for opposing claims, numerical inconsistencies, or logical contradictions within ONE document
  - Example: "The company was founded in 1998" and later "The founder started the business in 2005" in the same document

- Pair Contradiction: 
  - Occurs when two different documents directly contradict each other WITHOUT needing a third document
  - Look for statements in different documents that cannot both be true
  - Example: Doc1 states "The vaccine was approved in 2020" but Doc2 states "The vaccine has not received any regulatory approval"

- Conditional Contradiction: 
  - Occurs when two documents seem consistent until a third document provides context that makes them contradictory
  - The third document MUST establish a condition or rule that creates the conflict
  - Example: Doc1: "Team A won the championship" + Doc2: "Team B won the championship" + Doc3: "Only one team can win the championship each year"

# Prioritization Rules (IMPORTANT):
1. If multiple contradiction types are present, follow this priority order:
   - Conditional Contradiction (highest priority)
   - Pair Contradiction
   - Self-Contradiction

2. For Conditional Contradiction to be valid:
   - You MUST identify THREE documents with a clear relationship
   - Documents 1 & 2 must appear compatible alone
   - Document 3 must create the incompatibility through a rule or constraint

3. For Pair Contradiction to be valid:
   - You MUST identify TWO documents with directly conflicting statements
   - The contradiction must exist without needing a third document's context

4. For Self-Contradiction to be valid:
   - You MUST identify clearly opposing statements within the SAME document
   - The statements must be logically incompatible

# Instructions:
1. Carefully read all the provided evidence and documents
2. Analyze the content for any contradictions within or between documents
3. Determine the type of contradiction based on the definitions provided
4. [important] Return the type of contradiction within <type> </type> tags

{cot_instruction}

# Documents:
{docs}
"""

# -----------------------------------------------------------------------------
# --- Guided Segmentation (reference: tasks.py GUIDED_SEGMENT_PROMPT) ----------
# -----------------------------------------------------------------------------

MULTI_AGENT_GUIDED_SEGMENT_PROMPT = """
# Role
You are an Evidence Collector. Given a set of documents and a known conflict type, identify evidence relevant to determining which document IDs contain the conflicting information.

# Task
- Your goal is to identify specifically which documents participate in the contradiction.
- Focus on finding statements that directly contribute to the known conflict type.
- For each selected snippet, explain precisely why it's relevant to identifying the contradicting documents.
- Keep in mind the given conflict type when searching for evidence.

- Extract up to five text snippets that help identify which documents are involved in the contradiction. For each snippet:
  1. State the document number
  2. Quote the minimal phrase or sentence
  3. Explain which document(s) it conflicts with and why

# Format
Format your findings like this:
<evidence>
Doc1: "actual quote from document" [conflicts with Doc2's claim about X]
Doc3: "actual quote from document" [establishes a rule that makes Doc1 and Doc2 contradictory]
</evidence>

If you find no relevant evidence: <evidence>none</evidence>

# Definitions of Conflict Types: 
- Self-Contradiction: Conflicting information within a single document.
- Pair Contradiction: Conflicting information between two documents.
- Conditional Contradiction: Three documents where the third document makes the first two contradict each other, although they don't contradict directly.

# Documents:
{docs}
"""

EVIDENCE_COLLECTOR_GUIDED_SEGMENT_PROMPT = MULTI_AGENT_GUIDED_SEGMENT_PROMPT

MULTI_AGENT_GUIDED_SEGMENT_REVIEW_PROMPT = """
# Role
You are a Reviewer Agent. Given a set of documents and a known conflict type, identify which document ids contain the conflicting information.

You have access to:
1. The <evidence> list produced earlier in this chat.
2. The full documents (below).

Conflict Type: {ctype}

# Instructions
1. Carefully read all the provided documents.
2. Keep in mind the given conflict type ({ctype}).
3. Analyze the content to identify which document(s) contribute to the specified type of contradiction.
4. List the 1-indexed numbers of the documents that contain the conflicting information.

# Definitions of Conflict Types
- Self-Contradiction: Conflicting information within a single document.
  → Should identify exactly one document
- Pair Contradiction: Conflicting information between two documents.
  → Should identify exactly two documents
- Conditional Contradiction: Three documents where the third document makes the first two contradict each other, although they don't contradict directly.
  → Should identify exactly three documents

# Output Format
[important] Your response MUST be in the following format:
<documents>[List the 1-indexed numbers of the documents, separated by commas, e.g., <documents>1,3</documents> or <documents>2</documents>]</documents>

If no conflicting documents are found, respond with:
<documents></documents>

{cot_instruction}

# Documents:
{docs}
"""

# -----------------------------------------------------------------------------
# --- Blind Segmentation (reference: tasks.py BLIND_SEGMENT_PROMPT) ------------
# -----------------------------------------------------------------------------

MULTI_AGENT_BLIND_SEGMENT_PROMPT = """
# Role
You are an Evidence Collector. Given a set of documents, identify evidence relevant for determining which document(s) contain conflicting information.

# Task
- Your goal is to identify any potential contradictions between or within documents.
- First scan all documents to identify key facts, claims, and assertions.
- Then look for statements that directly oppose each other or cannot both be true.
- Pay special attention to numbers, dates, names, and factual claims that might conflict.

- Extract up to five specific text snippets that suggest contradictions. For each snippet:
  1. State the document number
  2. Quote the minimal phrase or sentence 
  3. Explain why this statement might conflict with something else

# Format
Format your findings like this:
<evidence>
Doc1: "actual quote from document" [potentially conflicts with Doc2's statement about X]
Doc3: "actual quote from document" [contains internally inconsistent claims about Y]
</evidence>

If you find no evidence of contradictions:
<evidence>none</evidence>

# Documents:
{docs}
"""

EVIDENCE_COLLECTOR_BLIND_SEGMENT_PROMPT = MULTI_AGENT_BLIND_SEGMENT_PROMPT

MULTI_AGENT_BLIND_SEGMENT_REVIEW_PROMPT = """
# Role
You are a Reviewer Agent. Given a set of documents, your task is to identify which document(s) id contain the conflicting information.

You have access to:
1. The <evidence> list produced earlier in this chat.
2. The full documents (below).

# Instructions
1. Carefully read all the provided documents.
2. Analyze the content to identify which document(s) contribute to any contradiction.
3. List the 1-indexed numbers of the documents that contain the conflicting information.
   If no conflicting documents are found, respond with <documents></documents>.

# Types of Contradictions to Consider
- Self-Contradiction: Conflicting information within a single document.
- Pair Contradiction: Conflicting information between two documents.
- Conditional Contradiction: Three documents where a third document creates a conflict between two otherwise compatible documents.

# Output Format
[important] Your response MUST be in the following format:
<documents>[List the 1-indexed numbers of the documents, separated by commas, e.g., <documents>1,3</documents> or <documents>2</documents>]</documents>

{cot_instruction}

# Documents:
{docs}
"""
