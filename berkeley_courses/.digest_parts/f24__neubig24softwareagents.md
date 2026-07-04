### Agents for Software Development
*Speaker:* Graham Neubig

#### Key claims / techniques
- Software developers spend most of their time on bugfixing (36%), coding (17%), communication (14%), documents/reviews (10%), testing (8%), and other tasks (15%) (Meyer et al. 2019), suggesting coding agents should support more than just code writing.
- Two complementary support modes exist: synchronous development copilots (GitHub Copilot, Cursor) and autonomous development agents (SWE-Agent, Aider, Devin, OpenHands), with OpenHands-resolver targeting autonomous issue resolution.
- Coding benchmarks span simple Python functions (HumanEval/MBPP), broader StackOverflow-style libraries (CoNaLa/ODEX), data-science notebooks (ARCADE), real-world GitHub issues (SWEBench), and web-to-code generation (Design2Code).
- Pass@K is the standard execution-based metric for code generation; variance is reduced by generating N > K samples with C correct answers and computing the expected value.
- When execution is hard, lexical/semantic overlap metrics are used, including BLEU, CodeBLEU (syntax and semantic flow), and CodeBERTScore (CodeBERT-based BERTScore).
- Dataset leakage is a serious concern: ARCADE shows novel notebooks are harder than online ones, and LiveCodeBench finds some code LMs outperform on existing HumanEval problems while failing on newly collected ones.
- Effective coding agents must understand repository structure, read existing code, modify or produce code, and run/debug it; action-space designs include CodeAct (bash/Jupyter execution), SWE-Agent (specialized repo tools), and OpenHands (event stream with agent skills).
- File localization strategies include offloading to the user, equipping the agent with search tools, a-priori repo maps (Aider repomap, Agentless hierarchical search), and retrieval-augmented code generation.
- Planning and error-recovery approaches range from hard-coded pipelines (Agentless: localize file, localize function, generate patch, apply patch) to LLM-generated plans (CodeR), iterative revisiting (CoAct), and feedback from execution error messages (InterCode).
- Safety risks include accidental destructive actions (e.g., pushing to main, deleting tests to pass) and intentional misuse (hacking); mitigations include sandboxed execution (OpenHands Docker), least-privilege credentialing, and post-hoc action auditing.

#### Relevance hooks
- Strong connection to agent evaluation methodology: compares SWEBench, Pass@K, execution-based vs. overlap metrics, and discusses leakage and benchmark freshness.
- Relevant to evals-with-error-bars: notes that raw Pass@K is high variance and describes the standard N>K correction.
- Touches on RL reproducibility standards through its emphasis on dataset leakage, execution-based evaluation, and carefully constructed code-generation benchmarks.

#### Cited paper titles (verbatim only)
- Today was a Good Day: The Daily Life of Software Developers
- Why Software is Eating the World

Index row: f24 | neubig24softwareagents.pdf | Graham Neubig | Survey of coding-agent environments, benchmarks, action spaces, localization, planning, and safety | ok
