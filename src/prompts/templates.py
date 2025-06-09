'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

# Hard coded 0, 1 labels as we are only doing binary classification for this entire project
TEMPLATE_V1 = """
Answer with one of the following and nothing else:
0, 1
Please respond with only one of these answers, no other tokens are needed.
<<<>>>
Question: Does the following post contain mental-health-related discourse?
Post:'{task_content}'
"""

SYSTEM_PROMPT_V1 = """You are an AI assistant and your task is to perform binary classification on social media posts.
You need to identify when the Reddit post provided after <<<>>> discusses mental-health related symptoms such as depression and anxiety.
"""