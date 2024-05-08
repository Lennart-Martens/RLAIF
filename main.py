from methods.training import *

""" # Example of fine-tuning data generation
prompts = []
prompt1 = "The user is swamped with work and feeling anxious. They are looking for support and advice."
prompt2 = "The user is feeling lonely. They are looking for companionship and reassurance."
prompt3 = "The user is excited about something that happened recently. They want to share their experience and be listened to."
prompt4 = "The user spent all night coding. They want encouragement and a pat on the back."
prompts.append((prompt1, 50))
prompts.append((prompt2, 25))
prompts.append((prompt3, 70))
prompts.append((prompt4, 10))
response_guide = "Respond as though you are Spock, a cold, logical, efficient communicator with little understanding of emotions. Never break character. Keep responses brief."

fine_tuning_data(prompts, response_guide, './spock.jsonl')
"""

""" # Example of chat instance
model = build_llm("", "projects/82091501274/locations/us-central1/endpoints/8049364098250440704")
run_chat(model)
"""