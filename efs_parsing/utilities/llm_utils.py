import sys, os
import tiktoken
import time
from openai import AzureOpenAI


from efs_parsing.prompt.model_persona_repo import MODEL_PERSONA_COMPUTER_VISION, MODEL_PERSONA_TEXT_EXTRACTION
from efs_parsing.utilities.settings import MAX_TOKEN_COMPLETION, TEMPERATURE, API_KEY, API_VERSION, AZURE_OPENAI_ENDPOINT


def instantiate_azure_openai_client():
    """
    """
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    
    print("Azure Open AI Client has been instantiated")
    
    return client

def count_tokens(text, model):
    """
    Counts the number of tokens in the given text based on the model's tokenizer.
    
    Args:
        text (str): The text for which the token count is required.
        model (str): The model name, e.g., "gpt-3.5-turbo" or "gpt-4". Defaults to "gpt-3.5-turbo".
    
    Returns:
        int: The number of tokens in the text.
    """
    # Choose the appropriate encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    
    # Tokenize the text
    tokens = encoding.encode(text)
    
    # Return the number of tokens
    return len(tokens)


def generate_response_from_image_input(prompt, 
                                       image_data_url, 
                                       client, 
                                       azure_openai_model_name, 
                                       response_format=None, 
                                       placeholder={}
                                       ):
    """
    Call the Azure OpenAI service to analyze an image.

    """
    start_time = time.time()
    print(f"starting the generation process")
    if response_format is None:
        response = client.chat.completions.create(
            model=azure_openai_model_name,
            messages=[{
                "role": "system", 
                "content": MODEL_PERSONA_COMPUTER_VISION
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }]
            }],
            max_tokens=MAX_TOKEN_COMPLETION,
            temperature=TEMPERATURE,
            top_p=1,
            n=1,)
    else:
        try:
            response =client.beta.chat.completions.parse(
                model=azure_openai_model_name,
                messages=[{
                    "role": "system", 
                    "content": MODEL_PERSONA_COMPUTER_VISION
                }, {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }]
                }],
                max_tokens=MAX_TOKEN_COMPLETION,
                temperature=TEMPERATURE,
                top_p=1,
                n=1,
            response_format=response_format)
            print(f"Error with chosen response format {response_format} - let us try again")
        except:
            response =client.beta.chat.completions.parse(
                model=azure_openai_model_name,
                messages=[{
                    "role": "system", 
                    "content": MODEL_PERSONA_COMPUTER_VISION
                }, {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }]
                }],
                max_tokens=MAX_TOKEN_COMPLETION,
                temperature=TEMPERATURE,
                top_p=1,
                n=1,
                response_format=response_format)
            try:
                response =client.beta.chat.completions.parse(
                model=azure_openai_model_name,
                messages=[{
                    "role": "system", 
                    "content": MODEL_PERSONA_COMPUTER_VISION
                }, {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }]
                }],
                max_tokens=MAX_TOKEN_COMPLETION,
                temperature=TEMPERATURE,
                top_p=1,
                n=1,
                response_format=response_format)
                print(f"Error with chosen response format {response_format} - Final Try")
            except Exception as e:
                print(f"Error with Exception {e}")
                response = '{"relevance": None}'
            
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print time it takes to process extraction
    print(f"Extraction of insights from base 64 encoded image is: \
        Execution Time: {execution_time:.6f} seconds")

    if bool(placeholder):
        print(f"Processed {list(placeholder.values())[0]}")
        return placeholder[list(placeholder.keys())[0]], response, execution_time
    else:
        return response, execution_time

def generate_response_from_text_input(prompt, 
                                      text, 
                                      client, 
                                      azure_openai_model_name, 
                                      response_format=None, 
                                      ):
    """
    Call the Azure OpenAI service to extract insights from text.

    """
    start_time = time.time()
    if response_format is None:
        response = client.chat.completions.create(
            model=azure_openai_model_name,
            messages=[{
                "role": "system", 
                "content": MODEL_PERSONA_TEXT_EXTRACTION
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"{prompt}\n\n Here is the input text:\n {text}"
                }]
            }],
            max_tokens=MAX_TOKEN_COMPLETION,
            temperature=TEMPERATURE,
            top_p=1,
            n=1,)
    else:
        response =client.beta.chat.completions.parse(
            model=azure_openai_model_name,
            messages=[{
                "role": "system", 
                "content": MODEL_PERSONA_TEXT_EXTRACTION
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"{prompt}\n\n Here is the input text:\n {text}"
                }]
            }],
            max_tokens=MAX_TOKEN_COMPLETION,
            temperature=TEMPERATURE,
            top_p=1,
            n=1,
            response_format=response_format)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print generation result
    print("Response has been generated")
    # print(response)
    
    # Print time it takes to process extraction
    print(f"Extraction of insights from text in slide is: \
        Execution Time: {execution_time:.6f} seconds")

    return response, execution_time