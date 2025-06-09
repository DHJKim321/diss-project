'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

# Hard coded 0, 1 labels as we are only doing binary classification for this entire project
TEMPLATE_V1 = """You are an AI assistant and your task is to streamline analysis of textual data.
You need to identify when the social media post provided after <<<>>> discusses mental-health related symptoms such as depression and anxiety.
In particular, you need to answer with one of the following:
0, 1
Please respond with only one of these answers, no other tokens are needed.
<<<>>>
Question: Does the following post contain mental-health-related discourse?
Post:'{task_content}'
"""