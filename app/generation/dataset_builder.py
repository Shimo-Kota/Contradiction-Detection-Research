"""
Synthetic Dataset Generation Framework.

This module implements Algorithm 1 from the paper "Contradiction Detection in RAG Systems"
(Gokul et al., 2025), which describes a framework for creating synthetic contradiction
datasets using LLMs. The implementation supports the three types of contradictions:

1. Self: Contradictions within a single document
2. Pair: Contradictions between two documents
3. Conditional: Contradictions involving three documents with conditional exclusivity

The generation process follows the exact steps outlined in Section 2.3 of the paper,
with parameters α (alpha) controlling statement salience and λ (lambda) controlling
context length. The LLM prompts used for generation are defined in Appendix A.1.
"""
from __future__ import annotations
import asyncio, json, random, pathlib, datetime as dt
from ..core.llm_clients import chat_completion
from ..prompts import data_generation as P

class SyntheticDatasetBuilder:
    """
    Implementation of Algorithm 1 from the paper for synthetic dataset generation.
    
    This class implements the contradiction generation algorithms described in 
    Section 2.3, using LLM prompts to perform the following operations:
    - ChooseStatement: Select important/unimportant statements from documents
    - ContradictStatement: Generate contradictions of selected statements
    - ContextGenerate: Generate surrounding context of specified length
    - GenerateConditionalContradiction: Create three-document conditional contradictions
    
    Attributes:
        alpha: Salience probability (0.5 by default) - controls whether to select
              most important or least important statements
        λ: Context length parameter (100 by default) - target word count for contexts
        model: LLM model to use for generation
        provider: Model provider ( google, anthropic)
    """

    def __init__(self, alpha: float = 0.5, λ: int = 100, model: str = None, provider: str = None):
        """
        Initialize the dataset builder with generation parameters.
        
        Args:
            alpha: Salience probability (default 0.5)
            λ: Context length parameter (default 100)
            model: LLM model to use for generation
            provider: Model provider (google, anthropic)
        """
        self.alpha = alpha
        self.λ = λ
        self.model = model
        self.provider = provider

    async def _choose_stmt(self, doc: str) -> tuple[str, str]:
        """
        Select a statement from a document based on importance.
        
        This implements the ChooseStatement function from Algorithm 1, which selects
        either the most important or least important statement from a document based
        on the alpha parameter.
        
        Args:
            doc: Document text to extract a statement from
            
        Returns:
            Tuple of (selected statement, importance level "most" or "least")
        """
        # Randomly choose most/least important based on alpha parameter
        importance = "most" if random.random() < self.alpha else "least"
        prompt = P.CHOOSE_STATEMENT_PROMPT.format(importance=importance, document=doc)
        resp = await chat_completion([{"role": "user", "content": prompt}], model=self.model, provider=self.provider)
        statement = _safe_extract_between(resp, "<sentence>", "</sentence>")
        return statement, importance

    async def _contradict(self, stmt: str) -> str:
        """
        Generate a contradiction of the given statement.
        
        This implements the ContradictStatement function from Algorithm 1, which
        creates a statement with opposite meaning to the original.
        
        Args:
            stmt: Statement to contradict
            
        Returns:
            Contradicting statement
        """
        prompt = P.CONTRADICT_STATEMENT_PROMPT.format(statement=stmt)
        resp = await chat_completion([{"role": "user", "content": prompt}], model=self.model, provider=self.provider)
        return _safe_extract_between(resp, "<statement>", "</statement>")

    async def _gen_context(self, stmt: str) -> tuple[str, int]:
        """
        Generate context around a statement.
        
        This implements the GenContext function from Algorithm 1, which creates
        surrounding text for a contradictory statement, targeting the specified
        context length (λ).
        
        Args:
            stmt: Statement to generate context around
            
        Returns:
            Tuple of (generated context paragraph, word count)
        """
        prompt = P.CONTEXT_GENERATE_PROMPT.format(sentence=stmt, length=self.λ)
        resp = await chat_completion([{"role": "user", "content": prompt}], model=self.model, provider=self.provider)
        paragraph = _safe_extract_between(resp, "<paragraph>", "</paragraph>")
        word_count = len(paragraph.split())
        return paragraph, word_count

    async def gen_self_contrad(self, doc: str) -> tuple[str, str, int]:
        """
        Generate a self-contradiction example.
        
        This implements the GenSelfContrad function from Algorithm 1, which creates
        a contradiction within a single document by adding a contradictory statement
        with surrounding context.
        
        Args:
            doc: Original document text
            
        Returns:
            Tuple of (document with added self-contradiction, statement importance, evidence length)
        """
        # Select a statement from the document
        s, importance = await self._choose_stmt(doc)
        # Generate its contradiction
        s_dash = await self._contradict(s)
        # Generate context around the contradictory statement
        c_dash, length = await self._gen_context(s_dash)
        # Return original document with contradiction appended along with metadata
        return doc + "\n" + c_dash, importance, length

    async def gen_pair_contrad(self, docs: list[str], cfg: str = "near") -> tuple[list[str], str, int]:
        """
        Generate a pair contradiction example.
        
        This implements the GenPairContrad function from Algorithm 1, which creates
        a contradiction between two documents by adding a contradictory statement
        from one document to another.
        
        Args:
            docs: List of original documents (minimum 2)
            cfg: Configuration for insertion - "near" places contradiction adjacent,
                 "far" places it randomly
                 
        Returns:
            Tuple of (list of documents with added contradiction, statement importance, evidence length)
        """
        docs = docs.copy()  # Avoid destructive changes
        # Randomly select a document to contradict
        i = random.randrange(len(docs))
        # Select a statement from this document
        s, importance = await self._choose_stmt(docs[i])
        # Generate its contradiction
        s_dash = await self._contradict(s)
        # Generate context around the contradictory statement
        c_dash, length = await self._gen_context(s_dash)
        # Insert the contradiction based on configuration
        insert_idx = min(i + 1, len(docs)) if cfg == "near" else random.randrange(len(docs) + 1)
        docs.insert(insert_idx, c_dash)
        return docs, importance, length

    async def gen_cond_contrad(self, doc: str, cfg: str = "contig") -> list[str]:
        """
        Generate a conditional contradiction example.
        
        This implements the GenCondContrad function from Algorithm 1, which creates
        a set of three documents with a conditional contradiction. The first document
        topic is used as a seed for the LLM to generate three related documents where
        the third establishes a mutually exclusive condition.
        
        Args:
            doc: Original document to use as topic seed
            cfg: Configuration for document ordering - "contig" keeps them together,
                 "separate" shuffles them
                 
        Returns:
            Tuple of (list of documents, list of document IDs involved in contradiction)
        """
        # Extract the first sentence to use as a topic seed
        first_sentence = doc.split(".")[0] + "."
        prompt = P.GENERATE_CONDITIONAL_PROMPT.format(topic=first_sentence)
        # Request LLM to generate the three documents with conditional contradiction
        resp = await chat_completion([{"role": "user", "content": prompt}], model=self.model, provider=self.provider)
        # Extract the three documents from the response
        d1 = _safe_extract_between(resp, "<document1>", "</document1>")
        d2 = _safe_extract_between(resp, "<document2>", "</document2>")
        d3 = _safe_extract_between(resp, "<document3>", "</document3>")
        docs = [d1, d2, d3]
        # Shuffle documents if configured to separate them
        if cfg == "separate":
            random.shuffle(docs)
        # Create mapping of documents to their original IDs
        # This ensures correct contradiction labeling even after shuffling
        id_map = {d: i+1 for i, d in enumerate(docs)}
        return docs, [id_map[d] for d in docs]
        
    async def _wrap_record(self, docs, ctype, ids, llm_indices=None, statement_importance=None, 
                          conflicting_evidence_length=None, pair_proximity=None):
        """
        Create a standard JSON record for dataset output.
        
        This helper function formats the generated documents into the standardized
        dataset format described in Section 2.6 of the paper, with additional metadata
        for experimental analysis.
        
        Args:
            docs: List of document texts
            ctype: Contradiction type ("self", "pair", "conditional", or "none")
            ids: 1-indexed list of document IDs involved in contradiction
            llm_indices: 1-indexed list of LLM-generated document indices
            statement_importance: Whether the contradicted statement was "most" or "least" important
            conflicting_evidence_length: Word count of the contradicting evidence
            pair_proximity: For pair contradictions, whether documents are "near" or "far"
            
        Returns:
            Formatted dataset record as a dictionary
        """
        # Convert plain text documents to record objects with metadata
        doc_objs = []
        for i, d in enumerate(docs):
            is_llm = (llm_indices is not None and (i+1) in llm_indices)
            is_contrad = (i+1) in ids
            doc_objs.append({
                "text": d,
                "source": "llm" if is_llm else "original",
                "contradiction": is_contrad
            })
        
        # Create the standard record format with metadata
        record = {
            "docs": doc_objs,
            "conflict": ctype != "none",
            "conflict_type": ctype,
            "doc_ids": ids,
        }
        
        # Add metadata for experimental analysis when present
        if statement_importance is not None:
            record["statement_importance"] = statement_importance
            
        if conflicting_evidence_length is not None:
            record["conflicting_evidence_length"] = conflicting_evidence_length
            
        if pair_proximity is not None:
            record["pair_proximity"] = pair_proximity
            
        return record

    async def build_dataset(self, source_path: pathlib.Path, out_dir: pathlib.Path, n: int = 100, ratio_none: float = 0.37):
        """
        Build a synthetic contradiction dataset.
        
        This method generates a dataset with the specified number of examples,
        following the distribution described in Section 2.5 of the paper:
        - none: ~37.5%
        - self: ~26.3%
        - pair: ~19.1%
        - conditional: ~17.1%
        
        Args:
            source_path: Path to the source corpus file (HotpotQA format)
            out_dir: Output directory for the generated dataset
            n: Number of examples to generate (default 100)
            ratio_none: Proportion of non-contradiction examples (default 0.37)
            
        Returns:
            Path to the generated dataset file
        
        Raises:
            ValueError: If the source corpus has fewer than 3 documents
        """
        def context_to_str(context):
            """Convert context from various formats to string."""
            if isinstance(context, list):
                return ' '.join(str(x) for x in context)
            return str(context)

        # Load source corpus based on file format
        if source_path.suffix == ".json":
            with source_path.open() as f:
                data = json.load(f)
            corpus = [context_to_str(item["context"]) for item in data if "context" in item]
        else:
            with source_path.open() as f:
                corpus = [context_to_str(json.loads(l)["context"]) for l in f if l.strip()]

        # Validate corpus size
        if len(corpus) < 3:
            raise ValueError("Corpus has fewer than 3 documents. At least 3 are required for dataset generation.")

        # Generate dataset records
        recs = []
        # Define contradiction types and their distribution weights
        choice_pool = ["none", "self", "pair", "cond"]
        weights = [ratio_none, (1-ratio_none)/3, (1-ratio_none)/3, (1-ratio_none)/3]
        
        for _ in range(n):
            # Select contradiction type based on weights
            typ = random.choices(choice_pool, weights)[0]
            
            if typ == "none":
                # For negative examples, select 1-3 random documents without modification
                k = random.randint(1, min(3, len(corpus)))
                docs = random.sample(corpus, k)
                # All documents are original with no contradiction
                recs.append(await self._wrap_record(docs, "none", []))
                
            elif typ == "self":
                # For self-contradictions, select one document and add contradictory text
                doc = random.choice(corpus)
                new_doc, importance, length = await self.gen_self_contrad(doc)
                doc_list = [new_doc]  # Single document containing the contradiction
                # Convert length to category (50, 100, 200)
                if length <= 60:
                    length_cat = 50
                elif length <= 150:
                    length_cat = 100
                else:
                    length_cat = 200
                recs.append(await self._wrap_record(
                    doc_list, "self", [1], llm_indices=[1],
                    statement_importance=importance,
                    conflicting_evidence_length=length_cat
                ))
                
            elif typ == "pair":
                # For pair contradictions, select two documents and add contradiction
                docs = random.sample(corpus, 2)
                # Randomly select near or far placement for experiment
                cfg = random.choice(["near", "far"])
                new_docs, importance, length = await self.gen_pair_contrad(docs, cfg=cfg)
                llm_indices = [i+1 for i, d in enumerate(new_docs) if d not in corpus]
                ids = llm_indices.copy()
                # Convert length to category (50, 100, 200)
                if length <= 60:
                    length_cat = 50
                elif length <= 150:
                    length_cat = 100
                else:
                    length_cat = 200
                recs.append(await self._wrap_record(
                    new_docs, "pair", ids, llm_indices=llm_indices,
                    statement_importance=importance,
                    conflicting_evidence_length=length_cat,
                    pair_proximity=cfg
                ))
                
            else:  # conditional
                # For conditional contradictions, generate three related documents
                doc = random.choice(corpus)
                docs, ids = await self.gen_cond_contrad(doc)
                # All documents are LLM-generated and all contribute to contradiction
                recs.append(await self._wrap_record(docs, "conditional", ids, llm_indices=[1,2,3]))

        # Create output directory if it doesn't exist
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file with timestamp
        out_path = out_dir / f"generated_{dt.datetime.utcnow().isoformat()}.jsonl"
        
        # Write records to file in JSONL format
        with out_path.open("w") as fout:
            for rec in recs:
                json.dump(rec, fout, ensure_ascii=False)
                fout.write("\n")
                
        return out_path

    async def build_self_only_dataset(self, corpus: list[str], n: int = 100, out_dir: pathlib.Path = pathlib.Path("data")) -> pathlib.Path:
        """
        Generate a dataset containing only self-contradiction examples.
        
        This method creates a dataset specifically for evaluating self-contradiction detection,
        as mentioned in the correction plan to properly evaluate this task that was previously
        incorrectly implemented.
        
        Args:
            corpus: List of source documents to use
            n: Number of self-contradiction examples to generate (default: 100)
            out_dir: Output directory for the dataset file
            
        Returns:
            Path to the generated JSONL file
        """
        recs = []
        
        print(f"Generating {n} self-contradiction examples...")
        
        for i in range(n):
            if i % 10 == 0:
                print(f"Generated {i}/{n} examples...")
                
            # For self-contradictions, select one document and add contradictory text
            doc = random.choice(corpus)
            new_doc, importance, length = await self.gen_self_contrad(doc)
            doc_list = [new_doc]  # Single document containing the contradiction
            
            # Convert length to category (50, 100, 200)
            if length <= 60:
                length_cat = 50
            elif length <= 150:
                length_cat = 100
            else:
                length_cat = 200
                
            recs.append(await self._wrap_record(
                doc_list, "self", [1], llm_indices=[1],
                statement_importance=importance,
                conflicting_evidence_length=length_cat
            ))

        # Create output directory if it doesn't exist
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file with timestamp and self-only marker
        out_path = out_dir / f"generated_self_only_{dt.datetime.utcnow().isoformat()}.jsonl"
        
        # Write records to file in JSONL format
        with out_path.open("w") as fout:
            for rec in recs:
                json.dump(rec, fout, ensure_ascii=False)
                fout.write("\n")
                
        print(f"Self-only dataset generated: {out_path}")
        return out_path

def _safe_extract_between(text: str, start: str, end: str) -> str:
    """
    Safely extract text between start and end tags.
    
    This helper function extracts content between specified tags in LLM responses,
    ensuring that the tag format described in Appendix A.1 is properly handled.
    
    Args:
        text: LLM response text to extract from
        start: Opening tag
        end: Closing tag
        
    Returns:
        Extracted text, or empty string if tags not found
    """
    import re
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""