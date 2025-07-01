'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

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

Post:
{task_content}

Answer:"""

# CoT
TEMPLATE_V3 = """Let's think step by step:
1. What mental health-related terms or symptoms are present?
2. Does the post express personal experience, reflection, or help-seeking?
3. Based on the above, classify the post.

Answer with just '0' or '1'. Do not reply with any other text:

Post:
{task_content}

Answer:"""

# CoT + Roleplaying
TEMPLATE_V4 = """You are a compassionate mental-health expert and counselor. Read the Reddit post below, think step by step, and then provide a final answer only as '0' or '1'.

1. Identify any mention of feelings, symptoms, therapy, or mental states.
2. Evaluate whether it's a discussion of mental health.
3. Finally, respond strictly with:
   - 1 = Yes, discusses mental health
   - 0 = No

Post:
{task_content}

Answer:"""

# JSON Structure
TEMPLATE_V5 = """You will classify the post with no extra commentary. Output must be valid JSON.

{"post": "{task_content}", "analysis": "<brief analysis>", "prediction": <0_or_1>}

Where:
- "analysis" is 1–2 sentences, thinking aloud.
- "prediction" is strictly 0 or 1.

JSON:
"""