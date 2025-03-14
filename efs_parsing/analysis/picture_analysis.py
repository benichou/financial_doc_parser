import sys, os
import time
import numpy as np
import math
from pathlib import Path
import ast

from efs_parsing.utilities.utils import chunk_list, flatten_list_of_list
from efs_parsing.utilities.picture_utils import compress_and_resize_image, local_image_to_data_url
from efs_parsing.utilities.llm_utils import count_tokens, generate_response_from_image_input
from efs_parsing.utilities.llm_structured_output import PictureProcessing

from efs_parsing.utilities.settings import AZURE_OPENAI_MODEL_NAME, AZURE_OPENAI_MODEL
from efs_parsing.utilities.settings import NON_REDUNDANT_PICTURE_SET_CHUNKING_SIZE, REDUNDANCY_PRESENCE_THRESHOLD, PICTURE_SIMILARITY, ALPHA
from efs_parsing.utilities.settings import IMAGE_COMPRESSION_QUALITY, IMAGE_COMPRESSION_FORMAT

from efs_parsing.prompt.prompt_repo import PICTURE_ASSESSMENT_PROMPT
from efs_parsing.prompt.model_persona_repo import MODEL_PERSONA_COMPUTER_VISION
# import custom_loader.unstructured.pdf_custom_loader as pdfucl 

import asyncio

# FOR RELATIVE IMPORT TESTING
# print("ok")
# print(PICTURE_SIMILARITY, ALPHA, IMAGE_COMPRESSION_FORMAT)
# print(generate_response_from_image_input)
# def picture_analysis_method():
# #     # print("Calling method from utils:")
# #     chunk_list()

# # picture_analysis_method()

def extract_picture_and_associated_metadata(extracted_file_data,
                                            parsing_method="docling"):
    """
    """
    picture_metadata_repo = {}
    if parsing_method == "unstructured":
        for element_index, element in enumerate(extracted_file_data):
            
            if element.metadata["category"] == pdfucl.ElementType.IMAGE:
                picture_metadata_repo[element_index] = element
        
        return picture_metadata_repo
    
    elif parsing_method == "docling":
        for element_index, element in enumerate(extracted_file_data.pictures):
            
            picture_metadata_repo[element_index] = element
        
        return picture_metadata_repo

def calc_bounding_box_euclidian_similarity_percentage(bbox1, bbox2):
    # bbox1 and bbox2 are tuples of (x1, y1, x2, y2)
    
    # Calculate centroid of each bounding box
    centroid1_x = (bbox1[0] + bbox1[2]) / 2
    centroid1_y = (bbox1[1] + bbox1[3]) / 2
    centroid2_x = (bbox2[0] + bbox2[2]) / 2
    centroid2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Compute Euclidean distance between centroids
    actual_distance = math.sqrt((centroid1_x - centroid2_x) ** 2 + (centroid1_y - centroid2_y) ** 2)

    # Calculate diagonal of the larger bounding box (d_max)
    diagonal1 = math.sqrt((bbox1[2] - bbox1[0]) ** 2 + (bbox1[3] - bbox1[1]) ** 2)
    diagonal2 = math.sqrt((bbox2[2] - bbox2[0]) ** 2 + (bbox2[3] - bbox2[1]) ** 2)
    max_diagonal = max(diagonal1, diagonal2)
    
    # Normalize and compute similarity percentage
    similarity_percentage = (1 - (actual_distance / max_diagonal)) * 100
    return max(0, min(similarity_percentage, 100))  # Ensure it's between 0% and 100%

def calc_bbox_similarity(element_1, element_2, parsing_method = "docling", alpha=ALPHA):
    
    if parsing_method == "unstructured":
        ## first bounding box coordinates
        x1 = element_1.metadata["coordinates"]["points"][0][0]
        x2 = element_1.metadata["coordinates"]["points"][2][0]
        y1 = element_1.metadata["coordinates"]["points"][0][1]
        y2 = element_1.metadata["coordinates"]["points"][2][1]
        bbox1 = (x1, y1, x2, y2)
        
        ## second bounding box coordinates
        x3 = element_2.metadata["coordinates"]["points"][0][0]
        x4 = element_2.metadata["coordinates"]["points"][2][0]
        y3 = element_2.metadata["coordinates"]["points"][0][1]
        y4 = element_2.metadata["coordinates"]["points"][2][1]
        bbox2 = (x3, y3, x4, y4)
        # Unpack the bounding box coordinates
        # x1, y1, x2, y2 = extracted_file_data[30].metadata["coordinates"]["points"][0], 
        # x3, y3, x4, y4 = extracted_file_data[55].metadata["coordinates"]
    elif parsing_method == "docling":
        x1 = element_1.l # left
        x2 = element_1.t # top
        y1 = element_1.r # right
        y2 = element_1.b # bottom
        bbox1 = (x1, y1, x2, y2)
        
        ## second bounding box coordinates
        x3 = element_2.l # left
        x4 = element_2.t # top
        y3 = element_2.r # right
        y4 = element_2.b # bottom
        bbox2 = (x3, y3, x4, y4)

    # Compute IoU (Intersection over Union)
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area
    ## -- Intersection Over Union = Area of Intersection / Area of Union --> if both are the same == 1 then overlap!
    iou = intersection_area / union_area if union_area != 0 else 0 # IoU (Overlap Measure)
    
    normalized_distance = calc_bounding_box_euclidian_similarity_percentage(bbox1, bbox2)
    # Compute final score
    iou_score = iou * 100
    
    similarity_score = alpha * iou_score + (1 - alpha) * normalized_distance

    return similarity_score

def analyze_overlap_and_euclidian_distance(picture_repo,
                                           parsing_method = "docling"):
    """
    """
    if parsing_method == "unstructured":
        
        analysis_start_time = time.time()
        picture_repo_analysis = {}
        for index_el_bb1, element_bbox_1 in picture_repo.items():
            picture_repo_analysis[index_el_bb1] = {}
            for index_el_bb2, element_bbox_2 in picture_repo.items():
                ## do not analyze an image bounding box and associated point coordinates with itself
                if index_el_bb1 == index_el_bb2:
                    continue
                else:
                    analysis_score = calc_bbox_similarity(element_bbox_1, element_bbox_2)
                picture_repo_analysis[index_el_bb1][index_el_bb2] = analysis_score
        
        analysis_end_time = time.time()
        analysis_processing_time = analysis_end_time - analysis_start_time
    
    elif parsing_method == "docling":
        
        analysis_start_time = time.time()
        picture_repo_analysis = {}
        for index_el_bb1, element_bbox_1 in picture_repo.items():
            picture_repo_analysis[index_el_bb1] = {}
            for index_el_bb2, element_bbox_2 in picture_repo.items():
                ## do not analyze an image bounding box and associated point coordinates with itself
                if index_el_bb1 == index_el_bb2:
                    continue
                else:
                    analysis_score = calc_bbox_similarity(element_bbox_1.prov[0].bbox, element_bbox_2.prov[0].bbox)
                picture_repo_analysis[index_el_bb1][index_el_bb2] = analysis_score
        
        analysis_end_time = time.time()
        analysis_processing_time = analysis_end_time - analysis_start_time
        
    print(f"The processing time to analyze pictures with overlap and their euclidian distance is \
        {analysis_processing_time} sec(s)")
    
    return picture_repo_analysis

def validate_list_of_redundant_irrelevant_pictures(picture_repo_analysis, 
                                                   total_number_pages):
    """
    """
    ## only keep the keys where the value is > PICTURE_SIMILARITY THRESHOLD 
    similar_picture_set = {}
    for index, analyzed_picture_set in picture_repo_analysis.items():
        similar_picture_set[index] = []
        for picture_key, picture_similarity_score in analyzed_picture_set.items():
            if picture_similarity_score > PICTURE_SIMILARITY:
                similar_picture_set[index].append(picture_key)
    
    ## keep unique sets
    ### first processing layer
    unique_redundant_picture_set_repo = []
    for picture_key, similar_picture_keys_set in similar_picture_set.items():
        if bool(similar_picture_keys_set): ## only analyze picture keys with similar pictures associated to them
            full_set = [key for key in similar_picture_keys_set]
            full_set.append(picture_key)
            # print(full_set)
            if bool(unique_redundant_picture_set_repo):
                ## Assess if the past set is similar to the new set that we are aiming to assess
                past_sets_repo = [past_set for past_set in unique_redundant_picture_set_repo if set(past_set) == set(full_set)]
                if not bool(past_sets_repo): # if it is not empty then it means the new set is indeed a unique set we can add
                    # print(f"Current <> All Past Full: {past_sets_repo}, {full_set}")
                    unique_redundant_picture_set_repo.append(full_set) 
            else:
                unique_redundant_picture_set_repo.append(full_set)
    
    ############################ CHAINING SETS OF REDUNDANT PICTURES THAT MIGHT BE CONNECTED TO EACH OTHER ############
    ### second processing layer to identify the sets that are intersecting entirely (for one of the sets)
    ## Are the sets connected in chains? If yes they should form one otherwise, they should stay separate
    ## because if sets share some redundant pictures then they are the same and should be merged
    connected_set_mapping = {}
    for idx_first_loop, similar_picture_keys_set_refined_first in enumerate(unique_redundant_picture_set_repo):
        for idx_second_loop, similar_picture_keys_set_refined_second in enumerate(unique_redundant_picture_set_repo):
            if idx_first_loop != idx_second_loop:
                for picture_key in similar_picture_keys_set_refined_first:
                    if idx_first_loop not in connected_set_mapping.keys():
                        connected_set_mapping[idx_first_loop] = []
                    if picture_key in similar_picture_keys_set_refined_second:
                            connected_set_mapping[idx_first_loop].append(idx_second_loop)
                            break # BREAK THIS LOOP and go back to UPPER LOOP: go to the next set!
    
    i = 1
    final_chained_sets_repo = {}
    ####### put the keys for the set together as a chain 
    for keys_one, chain_one in connected_set_mapping.items():
        
        if f"chain_{i}" not in final_chained_sets_repo.keys():
            final_chained_sets_repo[f"chain_{i}"] = []
        for keys_two, chain_two in connected_set_mapping.items():
            if keys_one != keys_two:
                if not bool(chain_one): # put the picture sets with no connection on their own
                    # print(f"ok {keys_one}")
                    final_chained_sets_repo[f"chain_{i}"].append(keys_one)
                    break
                else:
                    if keys_one in chain_two:
                        final_chained_sets_repo[f"chain_{i}"].append(chain_two)
        i += 1
    ## FINAL BLOCK TO CREATE THE CHAINS OF CONNECTED REDUNDANT PICTURES
    chain_list = []
    for _ , chain in final_chained_sets_repo.items():
        flattened_chain = flatten_list_of_list(chain)
        if not bool(chain_list):
            chain_list.append(list(flattened_chain))
        else:
            past_chain_repo = [past_set for past_set in chain_list if set(past_set) == set(flattened_chain)]
            if not bool(past_chain_repo):
                chain_list.append(list(flattened_chain))
    
    ## consolidate the chain of redundant sets into one final set, chains that correspond to one unique set 
    ## are their own set already
    refined_unique_redundant_picture_set_repo = []
    for chain in chain_list:
        consolidated_set = []
        
        for partial_set_key in chain:
            for picture_key in unique_redundant_picture_set_repo[partial_set_key]:
                consolidated_set.append(picture_key)
            # unique_redundant_picture_set_repo.pop(partial_set_key)
        refined_unique_redundant_picture_set_repo.append(list(set(consolidated_set)))
                
    ## keep redundant pictures that are across more than <REDUNDANCY_PRESENCE_THRESHOLD>% of the document 
    ## aka XX% of the total_number_of_pages
    
    redundancy_presence_threshold = REDUNDANCY_PRESENCE_THRESHOLD
    if total_number_pages > 50:
        redundancy_presence_threshold = redundancy_presence_threshold / 10 # if the file is very large then we will try to consider all redundant pictures
    minimum_number_pages_redundancy_presence = int(np.floor(total_number_pages * redundancy_presence_threshold))
    for idx, unique_redundant_picture_set in enumerate(refined_unique_redundant_picture_set_repo):
        if len(unique_redundant_picture_set) < minimum_number_pages_redundancy_presence:
            
            refined_unique_redundant_picture_set_repo.pop(idx)
            print("The redundant pictures were not redundant in the entire document so we do not consider them \
                    for discarding anymore and will be considered for knowledge extraction")
            
    return refined_unique_redundant_picture_set_repo


def analyze_picture_relevance(client,
                              compressed_images_output_folder_path,
                              picture_set,
                              extracted_file_data,
                              parsing_method = "docling"):
    """
    """
    representative_pic_key = picture_set[0] # trick that applies both to flag redundant irrelevant pictures and assess non redundant pictures
    ## Explanation of trick: instead of having a separate logic for redundant pictures and non redundant we can apply the same logic
    ## by making sure the picture set only represent one picture for processing non redundant picture set
    if parsing_method == "unstructured":
        ## retrieve the appropriate element from the list of langchain document with unstructured package metadata
        rep_picture_and_metadata = extracted_file_data[representative_pic_key]
        
        img_path = rep_picture_and_metadata.metadata["image_path"]
        page_number = rep_picture_and_metadata.metadata["page_number"]
        filename = rep_picture_and_metadata.metadata["filename"]
        
    elif parsing_method == "docling":
        rep_picture_and_metadata = extracted_file_data.pictures[representative_pic_key]
        
        img_path = rep_picture_and_metadata.image.uri
        page_number = rep_picture_and_metadata.prov[0].page_no
        filename = extracted_file_data.name
    
    compressed_img_path = compress_and_resize_image(image_path=img_path, 
                                                    output_path= Path(str(compressed_images_output_folder_path) + \
                                                    f"\\compressed_potential_redundant_pic_page_{page_number}_{filename}.{IMAGE_COMPRESSION_FORMAT}"), 
                                                    quality=IMAGE_COMPRESSION_QUALITY-10,
                                                    grayscale=True)
    
    image_data_url = local_image_to_data_url(compressed_img_path)
    input_message = MODEL_PERSONA_COMPUTER_VISION + PICTURE_ASSESSMENT_PROMPT + image_data_url
    input_tokens = count_tokens(input_message, AZURE_OPENAI_MODEL)
    
    image_description_raw, execution_time = generate_response_from_image_input(prompt=PICTURE_ASSESSMENT_PROMPT, 
                                                                               image_data_url=image_data_url, 
                                                                               client=client, 
                                                                               azure_openai_model_name=AZURE_OPENAI_MODEL_NAME,
                                                                               response_format=PictureProcessing)
    
    image_description = image_description_raw.choices[0].message.content
    image_description = ast.literal_eval(image_description)
    
    return (image_description, picture_set, input_tokens, execution_time)

async def analyze_picture_relevance_for_extraction(client,
                                                   compressed_images_output_folder_path,
                                                   element_set_of_pictures,
                                                   extracted_file_data):
    """
    """
    loop = asyncio.get_event_loop()
    tasks = []
    for picture_set in element_set_of_pictures:
        tasks.append(loop.run_in_executor(None, analyze_picture_relevance, client,
                                                                           compressed_images_output_folder_path,
                                                                           picture_set,
                                                                           extracted_file_data
                                                                             ))
    responses = await asyncio.gather(*tasks)
    
    return responses

def identify_pictures_relevance(client,
                                compressed_images_output_folder_path,
                                total_number_pages,
                                parsing_method,
                                extracted_file_data,
                                *args,
                                mode = "redundant_pictures" 
                                ):
    """
    """
    ## TODO: identify bboxes that are highly similar in terms of overlap and normalized euclidian distance
    ## --> create the repo of all images <key> (index in extracted data): <value> (point coordinates and image name)
    ## for each images and associated index + point coordinates go through all the others and assess similarity score
    ## if similarity > 90% then it is redundant images that should not be sent for extraction to LLMa 
    ## ALSO ADD logic to assess one sample of the redundant pictures to see if they should all be discarded or one 
    ## can be kept because it has meaning 
    
    ## before analyzing the pdf pages, we are going to assess the pictures of the pdf and see those
    ## that should not taken into account for knowledge extraction because they are irrelevant
    if mode == "redundant_pictures":
        picture_repo = extract_picture_and_associated_metadata(extracted_file_data,
                                                               parsing_method)
        picture_repo_analysis = analyze_overlap_and_euclidian_distance(picture_repo,
                                                                       parsing_method)

        element_set_of_pictures = validate_list_of_redundant_irrelevant_pictures(picture_repo_analysis, 
                                                                                 total_number_pages) ## set of REDUNDANT PICTURES
        
        
    elif mode == "non_redundant_pictures":
        element_set_of_pictures = [picture_key for picture_key in args]
        print(f"set of non redundant pictures considered {element_set_of_pictures}")
        
    redundant_pics_sets_relevance_analysis = asyncio.run(analyze_picture_relevance_for_extraction(client,
                                                                                                  compressed_images_output_folder_path,
                                                                                                  element_set_of_pictures,
                                                                                                  extracted_file_data)) 
    analysis_repo = {}
    for idx, analysis in enumerate(redundant_pics_sets_relevance_analysis):
        analysis_repo[idx+1] = analysis
    
    return analysis_repo


def analyze_non_redundant_pictures(client,
                                   non_redundant_pictures_set,
                                   compressed_images_output_folder_path,
                                   total_number_pages,
                                   parsing_method, 
                                   extracted_file_data_with_image_refs
                                   ):
    """
    """
    non_redundant_pictures_set_chunked = chunk_list(non_redundant_pictures_set, NON_REDUNDANT_PICTURE_SET_CHUNKING_SIZE)

    non_redundant_picture_analysis_repo = []
    for non_redundant_pictures_chunk in non_redundant_pictures_set_chunked:
        try:
            non_redundant_picture_analysis = identify_pictures_relevance(client,
                                                                        compressed_images_output_folder_path,
                                                                        total_number_pages,
                                                                        parsing_method,
                                                                        extracted_file_data_with_image_refs,
                                                                        *non_redundant_pictures_chunk,
                                                                        mode = "non_redundant_pictures"
                                                                        )
            print(f"NON REDUNDANT_PICTURE ANALYSIS {non_redundant_picture_analysis}")
        except Exception as e:
            print(f"Picture analysis errored for {extracted_file_data_with_image_refs.name} with exception as {e}")
            try:
                print(f"Second Try for set {non_redundant_pictures_chunk}")
                non_redundant_picture_analysis = identify_pictures_relevance(client,
                                                                            compressed_images_output_folder_path,
                                                                            total_number_pages,
                                                                            parsing_method,
                                                                            extracted_file_data_with_image_refs,
                                                                            *non_redundant_pictures_chunk,
                                                                            mode = "non_redundant_pictures"
                                                                            )
            except Exception as e:
                try:
                    print(f"Third Try for set {non_redundant_pictures_chunk} for {extracted_file_data_with_image_refs.name}")
                    non_redundant_picture_analysis = identify_pictures_relevance(client,
                                                                                compressed_images_output_folder_path,
                                                                                total_number_pages,
                                                                                parsing_method,
                                                                                extracted_file_data_with_image_refs,
                                                                                *non_redundant_pictures_chunk,
                                                                                mode = "non_redundant_pictures"
                                                                                )
                except:
                    print(f"Picture analysis errored FINAL for {extracted_file_data_with_image_refs.name} with exception as {e}")
                    non_redundant_picture_analysis = {}
                    for picture_id, picture_set in enumerate(non_redundant_pictures_chunk):
                        token_consumption = 0
                        execution_time = 0
                        non_redundant_picture_analysis[picture_id] = ({'relevance': None, 
                                                                       'relevance_explanation': None,
                                                                       'artefact_type': None, 
                                                                       'content_summary': None}, picture_set, 
                                                                        token_consumption, execution_time)
        
        non_redundant_picture_analysis_repo.append(non_redundant_picture_analysis)
    
    return non_redundant_picture_analysis_repo


