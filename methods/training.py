from vertexai.preview.generative_models import GenerativeModel, ChatSession, HarmCategory, HarmBlockThreshold
import time
from tqdm import tqdm


def build_llm(instructions:str="", version:str="gemini-1.0-pro"):
    """
    Builds an LLM from the VertexAI model garden.

    :param instructions: System instructions for the model
    :param version: The specific model to use
    :return: An instance of the specified model
    """
    safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
    }
    model = GenerativeModel(version, system_instruction=[instructions], safety_settings=safety)
    return model


def run_chat(model) -> int:
    """
    Starts a chat loop with the given model.

    :param model: The LLM to be chatted with
    :return: Zero, when the chat is successfully exited
    """
    chat = model.start_chat()
    while True:
        print()
        prompt = input("Enter your prompt (or 'q' to quit): ")

        if prompt=="q":
            break
        
        try:
            response = chat.send_message([prompt]).text
            print(response)
        except:
            print("Woah, slow down there!")
    return 0


def question_list(prompt:str, num_prompts:int=70, version:str="gemini-1.0-pro") -> list:
    """
    Builds a list of formatted prompts using an LLM

    :param prompt: The context in which these prompts could be given
    :param num_prompts: The ideal number of responses to generate
    :param version: The specific model to use
    :return: A formatted list of prompts
    """
    full_prompt = "Give me" + str(num_prompts) + " prompts a user might type in the following context: " + prompt
    instructions = "Give direct, specific answers to the question. Do not write anything but these answers. Seperate each answer with a pipe symbol like so: answer | answer | answer | ..."
    model = build_llm(instructions, version)
    
    unformatted_prompts = model.generate_content([full_prompt]).text
    split_prompts = unformatted_prompts.replace("\n", "").split("|")
    formatted_prompts = [x.strip() for x in split_prompts if x != ""]
    return formatted_prompts


def add_answers(prompts:list, model) -> list:
    """
    Pairs each prompt with a response given by the model

    :param prompts: The list of prompts to be answered
    :param model: The model that generates responses
    :return: A list of prompt-response tuples
    """
    pairs = []
    for prompt in tqdm(prompts):
        success = False
        while success is False:
            time.sleep(9)
            try:
                response = model.generate_content([prompt]).text
                response = response.replace("\n", "")
                success = True
            except:
                pass
        pairs.append((prompt, response))
    return pairs


def fine_tuning_data(query_request:list, response_instruction:str, jsonl_location:str) -> int:
    """
    Creates a JSONL file to perform supervised fine-tuning on a model

    :param query_request: A list of prompts and the number of times each should be generated
    :param response_instruction: Describes how prompts should be answered
    :param jsonl_location: The name and location of the file to be created
    :return: Zero, when the JSONL is generated successfully
    """
    prompts = []
    for request in tqdm(query_request):
        success = False
        while success is False:
            time.sleep(9)
            try:
                subset = question_list(request[0], request[1])
                success = True
            except:
                pass
        for result in subset:
            prompts.append(result)
    
    model = build_llm(response_instruction)
    pairs = add_answers(prompts, model)

    with open(jsonl_location, 'w', encoding="utf-8") as file:
        for pair in pairs:
            clean0 = pair[0].replace('"', '\\"')
            clean1 = pair[1].replace('"', '\\"')
            message1 = f'{{"role": "user", "content": "{ clean0 }"}}'
            message2 = f'{{"role": "model", "content": "{ clean1 }"}}'
            full = f'{{"messages": [{ message1 }, { message2 }]}}\n'
            file.write(full)
    
    return 0