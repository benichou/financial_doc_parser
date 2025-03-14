import sys, os


from efs_parsing.prompt.prompt_repo import process_math_extraction_prompt
from efs_parsing.prompt.prompt_repo import COMPLEX_SLIDE_EXTRACTION_PROMPT, PICTURE_ASSESSMENT_PROMPT, FULL_TEXT_EXTRACTION_PROMPT


def select_appropriate_extract_prompt(prompting_situation={}):
    """
    """
    if ("complex_slide" in prompting_situation.keys()) or ("math_equations" in prompting_situation.keys()):
        if "complex_slide" in prompting_situation.keys():
            if "math_equations" in prompting_situation.keys():
                prompt = process_math_extraction_prompt(prompting_situation)
            else:
                prompt = COMPLEX_SLIDE_EXTRACTION_PROMPT
        else:
            prompt = process_math_extraction_prompt(prompting_situation)
    else:
        if "picture_assessment" in prompting_situation.keys():
            prompt = PICTURE_ASSESSMENT_PROMPT
        
        if "full_text_extraction_and_analysis" in prompting_situation.keys():
            prompt = FULL_TEXT_EXTRACTION_PROMPT
    
    return prompt

def process_prompt_selection_for_page(page, page_for_full_base64_encoding):
    """
    """
    prompt_repo = {}
    if page in page_for_full_base64_encoding:
        complex_slide_analysis = page_for_full_base64_encoding[page]
        ## prompt to assess complex slides
        prompt_repo["prompt_complex_slide"] = select_appropriate_extract_prompt(complex_slide_analysis)
    ## prompt to assess pictures in the slide for extraction
    prompt_repo["prompt_picture_assessment"] = select_appropriate_extract_prompt(prompting_situation={"picture_assessment":True})
    ## prompt for full text extraction
    prompt_repo["prompt_full_text_extraction_and_analysis"] = select_appropriate_extract_prompt(prompting_situation={"full_text_extraction_and_analysis": True})

    return prompt_repo