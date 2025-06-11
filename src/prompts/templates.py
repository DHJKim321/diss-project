'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

# Hard coded 0, 1 labels as we are only doing binary classification for this entire project
TEMPLATE_V1 = """You are a binary classifier. Answer ONLY with '0' or '1'.

Task: Determine if the following Reddit post discusses mental health (e.g., depression, anxiety, therapy).

Reply with:
- 1 = Yes
- 0 = No

Post:
{task_content}

Answer:"""

SYSTEM_PROMPT_V1 = """You are an AI assistant and your task is to perform binary classification on social media posts.
You need to identify when the Reddit post provided after <<<>>> discusses mental-health related symptoms such as depression and anxiety.
"""