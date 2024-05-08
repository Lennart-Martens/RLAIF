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
    for x in tqdm(prompts):
        success = False
        while success is False:
            time.sleep(9)
            try:
                response = model.generate_content([x]).text
                response = response.replace("\n", "")
                success = True
            except:
                pass
        pairs.append((x, response))
    return pairs


def fine_tuning_data(query_prompt:str, query_num:int, response_instruction:str, jsonl_location:str) -> int:
    """
    Creates a JSONL file to perform supervised fine-tuning on a model

    :param query_prompt: Describes the type of prompts to be generated
    :param query_num: The ideal number of prompts to be generated
    :param response_instruction: Describes how prompts should be answered
    :param jsonl_location: The name and location of the file to be created
    :return: Zero, when the JSONL is generated successfully
    """
    prompts = question_list(query_prompt, query_num)
    model = build_llm(response_instruction)
    pairs = add_answers(prompts, model)

    with open(jsonl_location, 'w', encoding="utf-8") as file:
        for pair in pairs:
            message1 = f'{{"role": "user", "content": "{ pair[0].replace('"', '\\"') }"}}'
            message2 = f'{{"role": "model", "content": "{ pair[1].replace('"', '\\"') }"}}'
            full = f'{{"messages": [{ message1 }, { message2 }]}}\n'
            file.write(full)
    
    return 0