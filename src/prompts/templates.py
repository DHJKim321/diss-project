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

TEMPLATE_V2 = """You are an AI Assistant that performs binary classification on social media data.
More specifically, your task is to detect whether each post contains discourse related to mental health symptoms such as, but not limited to, depression, anxiety, and suicidal ideation.
You will be provided with a Reddit post after <<<>>>.
Please classify the post as follows:
1: Yes, the post discusses mental health symptoms.
0: No, the post does not discuss mental health symptoms.

Only reply with '1' or '0' as your answer. Do not provide any additional information or explanations.
Post:<<<{task_content}>>>
"""