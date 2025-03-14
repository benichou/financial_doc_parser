import sys, os
import time
import asyncio
from fastapi.responses import JSONResponse

from efs_parsing.utilities.utils import create_final_output_folders
from efs_parsing.utilities.llm_utils import instantiate_azure_openai_client
from efs_parsing.extraction.content_extraction import extract_data_and_metadata_from_file


def process_extraction_job_metadata(responses,
                                    overall_start_time,
                                    overall_end_time,
                                    file_list
                                    ):
    """
    """
    
    total_token_consumption = 0
    
    responses = [item for item in responses if item] # Use a list comprehension to filter out empty dictionaries
    for response in responses:
        file_path = list(response.keys())[0]
        if file_path.endswith("pdf"):
            total_token_consumption += response[file_path]['total_token_consumption']
        if file_path.endswith("pptx"):
            total_token_consumption += response[file_path][-1]['file_processing_statistics']['total_token_consumption']
    
    overall_processing_time = overall_end_time - overall_start_time
    overall_process_metadata = {}
    overall_process_metadata["overall_processing_time"] = overall_processing_time
    overall_process_metadata["number_files_processed"] = len(file_list)
    overall_process_metadata["processing_time_per_files"] = overall_processing_time / len(file_list) \
                                                                                    if len(file_list) != 0 else float('inf')
    overall_process_metadata["overal_token_consumption"] = total_token_consumption
    overall_process_metadata["token_consumption_per_file"] = total_token_consumption / len(file_list) \
                                                                                    if len(file_list) != 0 else float('inf')
    overall_process_metadata["file_path_list"] = file_list
    
    return overall_process_metadata

async def extract_file_content(absolute_folder_path= "C:/Users/fbenichou/projects/gen-ai-coe/EFS_prototype/efs_parsing/efs_parsing/input_files/efs_files/Onboarding/8-Miscellaneous/SWIFT",
                               parallel_processing = True):
    """
    """
    
    # if not os.path.exists(absolute_folder_path):
    #     print(JSONResponse(content={"error": "Folder not found."}, status_code=400)) ## make it a return later
    
    file_list = [f for f in os.listdir(absolute_folder_path) if (f.lower().endswith("pdf")) or (f.lower().endswith("pptx"))]
    file_list = [os.path.join(absolute_folder_path, f) for f in file_list]

    word_doc_out_path, md_ind_pages_from_pptx, md_doc_out_path = create_final_output_folders()
    
    client = instantiate_azure_openai_client()

    overall_start_time = time.time()
    
    # start parallel processing of files for optimized parsing process (sequential processing would lead to 2X 
    ## processing time)
    if parallel_processing:
        loop = asyncio.get_event_loop()
        tasks = []
        
        for file_path in file_list:
            tasks.append(loop.run_in_executor(None, extract_data_and_metadata_from_file, file_path, 
                                                                                        client,
                                                                                        word_doc_out_path, 
                                                                                        md_ind_pages_from_pptx, 
                                                                                        md_doc_out_path
                                                                                ))
        responses = await asyncio.gather(*tasks)
    
    else:
        responses = []
        for file_path in file_list:
            response = extract_data_and_metadata_from_file(file_path, 
                                                client,
                                                word_doc_out_path, 
                                                md_ind_pages_from_pptx, 
                                                md_doc_out_path
                                                )
        responses.append(response)
    
    overall_end_time = time.time()
    overall_process_metadata = process_extraction_job_metadata(responses,
                                                               overall_start_time,
                                                               overall_end_time,
                                                               file_list
                                                              )
    
    responses.append(overall_process_metadata)
    print(f"Overall Performance: {overall_process_metadata}")
        
    return responses

if __name__ == "__main__":
    
    extracted_file_repo_data_and_metadata = asyncio.run(extract_file_content())
    
