# Chatbots, RLHF, and ChatGPT

# What is a chatbot?

A computer program designed to simulate human conversation through text or voice interaction
Main types of chatbots:
- Task-based: often used in customer service or similar; usually implemented using rule-based methods
- Social: open-ended free-form chat; usually implemented using LLMs of some sort

## Using GPT as a chatbot

By default, an LLM like GPT tries to predict what comes next.

## ChatGPT: Reinforcement Learning with Human Feedback
Reinforcement learning:
- For each output, define a way to obtain a reward (higher = better)
- Goal: fine-tune LM to maximise the expected reward across all outputs
Optimising for human preferences:
- Train simulations of human preferences as a separate NLP problem
- Ask humans for pairwise comparisons instead of direct ratings
First InstructGPT (Ouyang et al 2022), then ChatGPT
RLHF agents are rewarded for responses that **humans prefer**, but this is not necessarily true.
- Improves output, but does not **eliminate hallucinations!**
