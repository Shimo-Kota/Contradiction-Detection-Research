DETECT_PROMPT = """\
You are given a set of documents. Do the documents contain conflicting information?
Your answer MUST include 'Yes' or 'No'.
{cot_instruction}
Here are the documents:
{docs}
"""

TYPE_PROMPT = """\
Given a set of documents with a contradiction, your task is to predict the type of contradiction present, if any. 
The possible types are:
1. Self-Contradiction: Conflicting information within a single document. 
2. Pair Contradiction: Conflicting information between two documents. 
3. Conditional Contradiction: Three documents where the third document makes the first two contradict each other.

Instructions: 
1. Carefully read all the provided documents. 
2. Analyze the content for any contradictions within or between documents. 
3. Determine the type of contradiction based on the definitions provided.
4. [important] Return the type of contradiction within <type> </type> tags. 
{cot_instruction}
Documents:
{docs}
"""

GUIDED_SEGMENT_PROMPT = """\
Given a set of documents and a known conflict type, identify which document ids contain the conflicting information.
Conflict Type: {ctype}

Instructions: 
1. Carefully read all the provided documents.
2. Keep in mind the given conflict type.
3. Analyze the content to identify which document(s) contribute to the specified type of contradiction. 
4. List the 1-indexed numbers of the documents that contain the conflicting information.If no conflicting documents are found, respond with <documents></documents>.
{cot_instruction}
[important] Your response should be in the following format: 
<documents>[List the 1-indexed numbers of the documents, separated by commas, e.g., <documents>1,3</documents> or <documents>2</documents>]</documents>

Definitions of Conflict Types: 
- SelfContradiction: Conflicting information within a single document. 
- Pair Contradiction: Conflicting information between two documents. 
- Conditional Contradiction: Three documents where the third document makes the first two contradict each other, although they don't contradict directly.

Here are the documents: 
{docs}
"""

BLIND_SEGMENT_PROMPT = """\
Given a set of documents, your task is to identify which document(s) id contain the conflicting information. 
Instructions: 
1. Carefully read all the provided documents.
2. Analyze the content to identify which document(s) contribute to the specified type of contradiction. 
3. List the 1-indexed numbers of the documents that contain the conflicting information.
   If no conflicting documents are found, respond with <documents></documents>.
{cot_instruction}
[important] Your response should be in the following format: 
<documents>[List the 1-indexed numbers of the documents, separated by commas, e.g., <documents>1,3</documents> or <documents>2</documents>]</documents> 
Here are the documents:
{docs}
"""

COT_INSTRUCTION = "\nThink step by step before answering."