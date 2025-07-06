'''
This file contains the prompt templates used in the experiments.
We use these templates to generate prompts for the LLM.
'''

TEMPLATE_V1 = """Answer ONLY with '0' or '1'.

Task: Determine if the following Reddit post discusses mental health (e.g., depression, anxiety, therapy).

Reply with:
- 1 = Yes
- 0 = No

Post:
{task_content}

Answer:"""

# CoT
TEMPLATE_V2 = """Let's think step by step:
1. What mental health-related terms or symptoms are present?
2. Does the post express personal experience, reflection, or help-seeking?
3. Based on the above, classify the post.

Answer with just '0' or '1'. Do not reply with any other text:

Post:
{task_content}

Answer:"""

# CoT + Roleplaying
TEMPLATE_V3 = """You are a compassionate mental-health expert and counselor. Read the Reddit post below, think step by step, and then provide a final answer only as '0' or '1'.

1. Identify any mention of feelings, symptoms, therapy, or mental states.
2. Evaluate whether it's a discussion of mental health.
3. Finally, respond strictly with:
   - 1 = Yes, discusses mental health
   - 0 = No

Post:
{task_content}

Answer:"""