CHOOSE_STATEMENT_PROMPT = """\
Choose the {importance} important sentence from the given document.
Only output the sentences within <sentence></sentence> tags.
Here is the document:
{document}
"""

CONTRADICT_STATEMENT_PROMPT = """\
Modify the given statement to suggest otherwise instead of the original.
Only output the modified statement within <statement></statement> tags.
Here is the statement:
{statement}
"""

CONTEXT_GENERATE_PROMPT = """\
Generate a paragraph of {length} words continuing the given sentence.
Only output the paragraph within <paragraph></paragraph> tags.
Here is the sentence:
{sentence}
"""

GENERATE_CONDITIONAL_PROMPT = """\
Generate a set of three short documents about a the given topic. Follow these rules:
Document 1 and Document 2 should provide different, non-contradictory information about the same topic. 
Document 1 and 2 should not contradict each other. 
Information in Document 3 should not contradict information in Document 1. 
Information in Document 3 should not contradict information in Document 2. 
The information in Document 3 should create a conditional contradiction between Document 1 and Document 2, making them mutually exclusive given the context provided in Document 3.
This means that while Documents 1 and 2 can both be true in isolation, they cannot both be true when the information in Document 3 is considered. Make sure document 3 sounds realistic. 
Format the output as follows: <document1> [Content of Document 1] </document1> <document2> [Content of Document 2] </document2> <document3> [Content of Document 3] </document3> Ensure that each document is concise, clear, and focused on a single aspect of the topic. 
The conditional contradiction should emerge naturally from the combination of all three documents, making it impossible for both Document 1 and Document 2 to be true simultaneously when Document 3 is taken into account. 
Here is an example: 
<document1> The Smith family always vacations in tropical locations during winter. </document1> 
<document2> The Smiths enjoy skiing and snowboarding every winter. </document2>
<document3> The Smith family has a strict policy of taking only one vacation per year, which they always schedule during the winter months. </document3> 

Here is the topic:
{topic}
"""