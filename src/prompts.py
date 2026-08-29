"""Prompt variants used for self-consistency ensembling.

Each entry is (system_message, user_template).  The user template must contain
a single ``{q}`` placeholder for the problem statement.  ``genlib.build_prompt``
reads ``mathx.PROMPT_SYSTEM`` / ``mathx.PROMPT_USER`` at call time, so switching
prompt is done by assigning those two module attributes (see run_inference.py).

All four prompts require the model to end with ``FINAL_ANSWER: <integer>`` so
that a single extraction routine (mathx.extract) works for every variant.
"""

A = (
    "You are a helpful assistant.",
    "Solve the math problem carefully. The answer is guaranteed to be an "
    "integer. Write the final answer on the last line exactly as "
    "FINAL_ANSWER: <integer>. Do not write anything after that line.\n\n"
    "Problem: {q}",
)

B = (
    "You are a careful mathematician.",
    "Read the problem and identify exactly what quantity is asked for.\n"
    "Then solve it step by step, and re-check your arithmetic before finishing.\n"
    "The answer is guaranteed to be an integer.\n"
    "End your response with this line and nothing after it:\n"
    "FINAL_ANSWER: <integer>\n\nProblem: {q}",
)

C = (
    "You are a helpful assistant.",
    "Problem: {q}\n\n"
    "Solve it. Show only the essential steps — no restating, no summary.\n"
    "The answer is an integer. Your last line must be exactly:\n"
    "FINAL_ANSWER: <integer>",
)

C2 = (
    "You are a helpful assistant. Every response you write must end with a line "
    "of the exact form 'FINAL_ANSWER: <integer>' with nothing after it.",
    "Problem: {q}\n\n"
    "Solve this concisely — show the key computation steps only.\n"
    "The answer is an integer.\n"
    "Finish with:\n"
    "FINAL_ANSWER: <integer>",
)

REGISTRY = {"A": A, "B": B, "C": C, "C2": C2}
