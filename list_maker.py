def generate_multicol_string(prompts):
    header = r"""\begin{multicols}{2}
\small
\begin{itemize}
"""

    footer = r"""
\end{itemize}
\end{multicols}
"""

    items = "\n".join([f"    \\item {prompt}" for prompt in prompts])

    return header + items + footer


with open("prompts.txt", "r") as f:
    prompts = f.read().splitlines()

multicols_content = generate_multicol_string(prompts)

print(multicols_content)