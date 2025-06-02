'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

TEMPLATE_V1 = """You are an AI assistant and your task is to streamline analysis of textual data.
You need to identify when the social media post provided after <<<>>> discusses the topic of interest, mental health, and more specifically, depression.
In particular, you need to answer with one of the following:
{classes}
Please respond with only one of these answers, no other tokens are needed.
<<<>>>
Question: Does the following post display signs of depression with regards to the author?
Post:'{task_content}'
"""