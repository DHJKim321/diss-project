'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

# Hard coded 0, 1 labels as we are only doing binary classification for this entire project
TEMPLATE_V1 = """Answer with only one of the following digits: 0 or 1.
Respond with exactly one digit and nothing else.

Question: Does the following Reddit post contain mental-health-related discourse?

Post:
{task_content}

Answer:"""

SYSTEM_PROMPT_V1 = """You are an AI assistant and your task is to perform binary classification on social media posts.
You need to identify when the Reddit post provided after <<<>>> discusses mental-health related symptoms such as depression and anxiety.
"""