import sys, os
import asyncio
import ast
from pathlib import Path
from pptx import Presentation


from efs_parsing.prompt.prompt_mgmt import process_prompt_selection_for_page
from efs_parsing.prompt.model_persona_repo import MODEL_PERSONA_COMPUTER_VISION, MODEL_PERSONA_TEXT_EXTRACTION

from efs_parsing.utilities.picture_utils import convert_slide_to_image, local_image_to_data_url, compress_and_resize_image 
from efs_parsing.utilities.picture_utils import create_img_markdown_placeholder, process_image_insight_extraction
from efs_parsing.utilities.llm_utils import count_tokens, generate_response_from_image_input, generate_response_from_text_input
from efs_parsing.utilities.llm_structured_output import SlideInfo, PictureProcessing, TextProcessing

from efs_parsing.utilities.settings import AZURE_OPENAI_MODEL, CONTEXT_WINDOW_LIMIT, AZURE_OPENAI_MODEL_NAME, IMAGE_COMPRESSION_FORMAT
from efs_parsing.utilities.settings import EXTRACTED_OUTPUT_ROOT



def extract_slide_content(page,
                          page_for_full_base64_encoding,
                          prompt_repo,
                          file_name,
                          file_path,
                          slide_images_output_folder_path,
                          client,
                          notes,
                          extracted_file_data,
                          document_shape_analyzer_object_name,
                          document_math_shape_analyzer_object_name):
    """
    """
    
    extracted_slide_content_repo = {}
    print(f"processing for page {page}")
    if page in page_for_full_base64_encoding:
        print(prompt_repo["prompt_complex_slide"])
        selected_prompt = prompt_repo["prompt_complex_slide"]
        
        print("Processing complex slide into base64 encoding to ensure accurate insights extraction")
        try:
            image_path = convert_slide_to_image(file_path, 
                                                slide_images_output_folder_path, 
                                                page, 
                                                file_name) ## TODO: RELIES ON COMTYPES --> 
                                                           ## WE NEED TO FIND AN EASY WAY TO NOT
                                                           ## USE COMTYPES FOR IMG CONVERSION 
                                                           ## EITHER DIRECTLY FROM PPTX OR PDF
        except:
            ## trying multiple times in case pptx opening did not work -- unstable behavior requires this handling
            try:
                print(f"Second try to convert slide {page} to image pptx is unstable")
                image_path = convert_slide_to_image(file_path, 
                                                    slide_images_output_folder_path, 
                                                    page, 
                                                    file_name) 
            except:
                try:
                    print(f"Third try to convert slide {page} to image pptx is unstable")
                    image_path = convert_slide_to_image(file_path, 
                                                        slide_images_output_folder_path, 
                                                        page, 
                                                        file_name) 
                
                except Exception as e:
                    print(f"WARNING: Error converting slide {page} to image: {e} WARNING")
        
        image_data_url = local_image_to_data_url(image_path)
        input_message = MODEL_PERSONA_COMPUTER_VISION + selected_prompt + image_data_url
        input_tokens = count_tokens(input_message, AZURE_OPENAI_MODEL)
        
        if input_tokens > CONTEXT_WINDOW_LIMIT:
            image_path = convert_slide_to_image(file_path, 
                                                slide_images_output_folder_path,
                                                page, 
                                                file_name,
                                                mode="secondary_aggresive_compression")
                
            image_data_url = local_image_to_data_url(image_path)
            input_message =  MODEL_PERSONA_COMPUTER_VISION + selected_prompt + image_data_url
            new_input_tokens = count_tokens(input_message, AZURE_OPENAI_MODEL)
            
            if new_input_tokens > CONTEXT_WINDOW_LIMIT:
                print("Full page base64 encoding is too large - page has been skipped")
                
        try: 
            slide_description_raw, execution_time = generate_response_from_image_input(prompt=selected_prompt, 
                                                                                   image_data_url=image_data_url, 
                                                                                   client=client,
                                                                                   azure_openai_model_name=AZURE_OPENAI_MODEL_NAME,
                                                                                   response_format=SlideInfo) # SlideInfo pydantic aspect for structured output
        except:
            print(f"Let us try insight extraction from image again")
            try:
                slide_description_raw, execution_time = generate_response_from_image_input(prompt=selected_prompt, 
                                                                                   image_data_url=image_data_url, 
                                                                                   client=client,
                                                                                   azure_openai_model_name=AZURE_OPENAI_MODEL_NAME,
                                                                                   response_format=SlideInfo)
            except Exception as e:
                print(f"Error extracting insights from image with error: {e}")
        
        markdown_slide_placeholder = create_img_markdown_placeholder(image_path, page)    

        slide_description = slide_description_raw.choices[0].message.content
        try:
            slide_description = ast.literal_eval(slide_description) # convert the previous string output to a readable python dict
        except:
            print(f"Extraction Generation issue for fully base 64 encoded slide number {page}")

        ## store the information
        extracted_slide_content_repo[page] = {}
        extracted_slide_content_repo[page]["fully_base64_converted"] = True
        extracted_slide_content_repo[page]["overview"] = slide_description['overview']
        extracted_slide_content_repo[page]["artefact_type"] = slide_description['artefact_type']
        extracted_slide_content_repo[page]["slide_title"] = slide_description['slide_title']
        extracted_slide_content_repo[page]["content_summary"] = slide_description['content_summary'] \
                                                                                       + "\n\n" + "## [Complex] Source Slide:" \
                                                                                       + "\n\n" + markdown_slide_placeholder \
                                                                                       + "\n\n" + "## Speaker Notes:" \
                                                                                       + "\n\n" + notes[page]            
        extracted_slide_content_repo[page]["image_path"] = image_path
        extracted_slide_content_repo[page]["speaker_notes"] = notes[page]
        extracted_slide_content_repo[page]["own_accounting_input_tokens"] = input_tokens
        extracted_slide_content_repo[page]["open_ai_accounting_output_tokens"] = slide_description_raw.usage.\
                                                                                                completion_tokens
        extracted_slide_content_repo[page]["open_ai_accounting_input_tokens"] = slide_description_raw.usage.\
                                                                                                prompt_tokens
        extracted_slide_content_repo[page]["open_ai_accounting_total_tokens"] = slide_description_raw.usage.\
                                                                                                total_tokens
        extracted_slide_content_repo[page]["processing_time_seconds"] = execution_time # in seconds
        
    else:
        ## extract all items pertaining to the said page
        page_elements = [docmt for docmt in extracted_file_data if docmt.metadata["page_number"] == page]
        title_cursor = 0
        image_cursor = 0
        tbl_cursor = 0
        parsed_textual_data = ""
        parsed_img_repo = {}
        
        ## concatenate elements
        for element in page_elements:
            
            if element.page_content not in [document_shape_analyzer_object_name,
                                            document_math_shape_analyzer_object_name]:
                if element.metadata["category"] in ['Title', 
                                                    'NarrativeText', 
                                                    'UncategorizedText',
                                                    'ListItem']:
                    
                    if element.metadata["category"] == "Title":
                        if len(element.page_content) < 2: #### make sure random letters or numbers or not added
                            continue
                        if title_cursor == 0:
                            text_identifier = f"Title: "
                            initial_break = ""
                            title_cursor += 1
                        else:
                            text_identifier = f"Subsection {title_cursor}: "
                            initial_break = "\n"
                            title_cursor += 1
                    
                    if element.metadata["category"] in ['NarrativeText', 'UncategorizedText']:
                        text_identifier = f"Paragraph: "
                        initial_break = "\n"
                        if len(element.page_content) < 2: ## make sure random letters or numbers or not added
                            continue ## we are making sure we keep only relevant texts and not stuffs like "1."
                    
                    if element.metadata["category"] == "ListItem":
                        if len(element.page_content) < 2: #### make sure random letters or numbers or not added
                            continue
                        text_identifier = f"Bullet point element: - "
                        initial_break = "\n"
                    
                    parsed_textual_data += initial_break + text_identifier + element.page_content
                    
                if element.metadata["category"] == "Image":
                    text_identifier = f"Placeholder_Image_{image_cursor+1}:"
                    parsed_img_repo[text_identifier] = {}
                    parsed_img_repo[text_identifier]["base64_encoding"] = element.metadata["image_base64"]
                    parsed_img_repo[text_identifier]["image_path"] = element.metadata["image_path"]
                    initial_break = "\n\n"
                    image_cursor += 1
                    parsed_textual_data += initial_break + text_identifier
                
                if element.metadata["category"] == "Table":
                    text_identifier = f"Table_{tbl_cursor+1}: "
                    initial_break = "\n\n"
                    tbl_cursor += 1
                    parsed_textual_data += initial_break + text_identifier + element.metadata["text_as_html"] + \
                                            initial_break
        parsed_textual_data += "\n\n" +  "Speaker notes:" + notes[page] # TODO: assess later if you need to put the speaker notes
            
        ## assess pictures in the slide for extraction
        selected_prompt = prompt_repo["prompt_picture_assessment"]
        
        ## create the correct image_data_url from the correct compressed image
        if bool(parsed_img_repo):
            for img in parsed_img_repo:
                image_path = parsed_img_repo[img]["image_path"]
                compressed_image_path = Path(str(slide_images_output_folder_path) + f"\\compressed_slide_{page}_for_{file_name}.{IMAGE_COMPRESSION_FORMAT}")
                image_path = compress_and_resize_image(image_path=image_path, 
                                                       output_path=compressed_image_path,
                                                       grayscale=True) ## this action reduces on average the token count by 30%
                image_data_url = local_image_to_data_url(image_path)
                input_message = MODEL_PERSONA_COMPUTER_VISION + selected_prompt + image_data_url
                input_tokens = count_tokens(input_message, AZURE_OPENAI_MODEL)
                
                if input_tokens > CONTEXT_WINDOW_LIMIT:
                    image_path = parsed_img_repo[img]["image_path"]
                    image_path = compress_and_resize_image(image_path=image_path, 
                                        output_path=compressed_image_path,
                                        quality=70,
                                        grayscale=True) ## this action reduces on average the token count by 30%
                    image_data_url = local_image_to_data_url(image_path)
                
                parsed_img_repo[img]['image_data_url'] = image_data_url
            
            # get the list of images for parallel processing
            img_repo = []
            for img in parsed_img_repo:
                single_img_repo = []
                single_img_repo.append(img)
                single_img_repo.append(parsed_img_repo[img]["image_data_url"])
                img_repo.append(single_img_repo)
        
        
            image_extraction_analysis = asyncio.run(process_image_insight_extraction(list_of_images=img_repo,
                                                                                     prompt=selected_prompt, 
                                                                                     client=client, 
                                                                                     azure_openai_model_name=AZURE_OPENAI_MODEL_NAME,
                                                                                     response_format=PictureProcessing
                                                                                     ))
            # collect the analysis in the img repo
            for extracted_image_insights in image_extraction_analysis:
                
                image_identifier = extracted_image_insights[0]
                image_analysis = extracted_image_insights[1]
                image_analysis_execution_time = extracted_image_insights[2]

                img_description = image_analysis.choices[0].message.content
                try:
                    img_description = ast.literal_eval(img_description) # convert the previous string output to a readable python dict
                except:
                    print(f"Extraction Generation issue for {img}")
                
                parsed_img_repo[image_identifier]["relevance"] = img_description["relevance"]
                parsed_img_repo[image_identifier]["relevance_explanation"] = img_description["relevance_explanation"]
                parsed_img_repo[image_identifier]["artefact_type"] = img_description["artefact_type"]
                parsed_img_repo[image_identifier]["content_summary"] = img_description["content_summary"]
            
        ## enrich the parsed textual with the correct images
        
            for img in parsed_img_repo:
                
                img_relevance = parsed_img_repo[img]["relevance"]
                
                if img_relevance == 'True':
                    parsed_textual_data = parsed_textual_data.replace(img, img + " " + parsed_img_repo[img]["image_path"] + "\n" + \
                                                parsed_img_repo[img]["content_summary"])
                if img_relevance == 'False':
                    parsed_textual_data = parsed_textual_data.replace(img, "")
        
        selected_prompt = prompt_repo["prompt_full_text_extraction_and_analysis"]
        input_message =  MODEL_PERSONA_TEXT_EXTRACTION + selected_prompt + parsed_textual_data
        input_tokens = count_tokens(input_message, AZURE_OPENAI_MODEL)
        
        # generate the content into a markdown format
        regenerated_text, execution_time = generate_response_from_text_input(prompt=selected_prompt, 
                                                                             text=parsed_textual_data, 
                                                                             client=client, 
                                                                             azure_openai_model_name=AZURE_OPENAI_MODEL_NAME, 
                                                                             response_format=TextProcessing, 
                                                                            )
        slide_description = regenerated_text.choices[0].message.content
        try:
            slide_description = ast.literal_eval(slide_description) # convert the previous string output to a readable python dict
        except:
            print(f"Extraction Generation issue for fully base 64 encoded slide number {page}")
        slide_description["content"] = slide_description["content"].replace(EXTRACTED_OUTPUT_ROOT, f"/{EXTRACTED_OUTPUT_ROOT}") ## IMPORTANT TO MAKE SURE YOU LL BE ABLE TO SEE THE IMAGES IN MARKDOWN
        
        ################### TODO TO REFACTOR AS SEPARATE FUNCTION ##########################################################
        ## correcting for hallucinated or missing image paths
        missed_paths = [itm["image_path"] for idx, itm in parsed_img_repo.items() if itm["image_path"] not in slide_description['content']]
        if bool(missed_paths):
            import re
            
            pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
            matches = re.findall(pattern, slide_description['content'])
            if bool(matches):
                if len(missed_paths) == 1: ## if only one path in the slide is supposed to be found
                    for placeholder_img, hallucinated_file_path in matches:
                        print(f"Hallunicated Text for Image: {placeholder_img}")
                        print(f"Hallunicated File Path: {hallucinated_file_path}")
                    slide_description['content'].replace(placeholder_img, "Placeholder Image") ## replace markdown image title
                    slide_description['content'].replace(hallucinated_file_path, f"/{missed_paths[0]}") ## replace markdown image path
                else:
                    for matched_item in matches:
                        slide_description['content'].replace(f"![{matched_item[0]}]({matched_item[1]})", "") ## remove hallucinated image paths
                    for missed_path in missed_paths:
                        slide_description['content'] += "\n\n" + f"![Corrected Placeholder Image](/{missed_path})"
        ################### TODO TO REFACTOR AS SEPARATE FUNCTION ##########################################################
        
        ## store the information
        extracted_slide_content_repo[page] = {}
        extracted_slide_content_repo[page]["fully_base64_converted"] = False
        extracted_slide_content_repo[page]["overview"] = slide_description['overview']
        extracted_slide_content_repo[page]["artefact_type"] = "Text-based slide"
        extracted_slide_content_repo[page]["slide_title"] = slide_description['slide_title']
        extracted_slide_content_repo[page]["content_summary"] = slide_description['content']             
        extracted_slide_content_repo[page]["image_path"] = [itm["image_path"] for idx, itm in parsed_img_repo.items()]
        extracted_slide_content_repo[page]["speaker_notes"] = notes[page]
        extracted_slide_content_repo[page]["own_accounting_input_tokens"] = input_tokens
        extracted_slide_content_repo[page]["open_ai_accounting_output_tokens"] = regenerated_text.usage.\
                                                                                                completion_tokens
        extracted_slide_content_repo[page]["open_ai_accounting_input_tokens"] = regenerated_text.usage.\
                                                                                                prompt_tokens
        extracted_slide_content_repo[page]["open_ai_accounting_total_tokens"] = regenerated_text.usage.\
                                                                                                total_tokens
        extracted_slide_content_repo[page]["processing_time_seconds"] = execution_time # in seconds

    return extracted_slide_content_repo


async def process_slide_extraction(list_pages,
                                   page_for_full_base64_encoding,
                                   file_name,
                                   file_path,
                                   slide_images_output_folder_path,
                                   client,
                                   notes,
                                   extracted_file_data,
                                   document_shape_analyzer_object_name,
                                   document_math_shape_analyzer_object_name):
    """
    """
    loop = asyncio.get_event_loop()
    tasks = []
    for page in list_pages:
        print(f"PAGE {page}")
        prompt_repo = process_prompt_selection_for_page(page, page_for_full_base64_encoding)
        if page in page_for_full_base64_encoding:
            print(prompt_repo["prompt_complex_slide"])
        tasks.append(loop.run_in_executor(None, extract_slide_content, page,
                                                                       page_for_full_base64_encoding,
                                                                       prompt_repo,
                                                                       file_name,
                                                                       file_path,
                                                                       slide_images_output_folder_path,
                                                                       client,
                                                                       notes,
                                                                       extracted_file_data,
                                                                       document_shape_analyzer_object_name,
                                                                       document_math_shape_analyzer_object_name
                                                                             ))
    responses = await asyncio.gather(*tasks)
    return responses


def extract_speaker_notes(ppt_path):
    """
    Extracts speaker notes from a PowerPoint presentation.
    
    Args:
        ppt_path (str): Path to the PowerPoint file.
    
    Returns:
        dict: A dictionary containing slide number and its associated speaker notes.
    """
    prs = Presentation(ppt_path)
    slide_notes = {}

    for slide_number, slide in enumerate(prs.slides):
        # Check if slide has speaker notes (notes_slide)
        notes_text = ""
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            notes_text = ""
            
            # Extract text from the notes slide (if any)
            if notes_slide.notes_text_frame is not None:
                for paragraph in notes_slide.notes_text_frame.paragraphs:
                    for run in paragraph.runs:
                        notes_text += run.text
                    notes_text += "\n"  # Add newline after each paragraph

            # Store speaker notes in dictionary (slide_number starts from 0, add 1 for human-friendly index)
        slide_notes[slide_number + 1] = notes_text.strip()
    
    return slide_notes
