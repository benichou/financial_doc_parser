import sys, os
from pathlib import Path

import re

from efs_parsing.utilities.settings import WORD_DOCUMENT_FINAL_OUTPUT_PATH, MARKDOWN_INDIVIDUAL_PAGES_FROM_PPTX_PAGES_OUTPUT_PATH
from efs_parsing.utilities.settings import MARKDOWN_DOCUMENT_FINAL_OUTPUT_PATH

def create_final_output_folders():
    """
    """
    
    ## create the output folders
    word_document_output_path = Path(os.getcwd() + WORD_DOCUMENT_FINAL_OUTPUT_PATH)
    markdown_individual_pages_from_pptx = Path(os.getcwd() + MARKDOWN_INDIVIDUAL_PAGES_FROM_PPTX_PAGES_OUTPUT_PATH)
    markdown_document_output_path = Path(os.getcwd() + MARKDOWN_DOCUMENT_FINAL_OUTPUT_PATH)
    
    os.makedirs(word_document_output_path, exist_ok=True)
    #print(f"appropriate output folder for final word output (.docx) has been created")
    os.makedirs(markdown_individual_pages_from_pptx, exist_ok=True)
    #print(f"appropriate output folder for file markdown individual pages has been created")
    os.makedirs(markdown_document_output_path, exist_ok=True)
    # print(f"appropriate output folder for final markdown output (.md) has been created")
    
    return word_document_output_path, markdown_individual_pages_from_pptx, markdown_document_output_path

def replace_between_markers(text, start_marker, end_marker, replacement_text="PLACEHOLDER"):
    # Define a function to use as a replacement
    def replacement(match):
        return f"{start_marker}{replacement_text}{end_marker}"
    
    # Regular expression to match content between markers
    pattern = re.escape(start_marker) + ".*?" + re.escape(end_marker)
    return re.sub(pattern, replacement, text, flags=re.DOTALL)


def chunk_list(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def flatten_list_of_list(list_of_lists):
    """
    """
    flattened_list = set(x if not isinstance(x, list) else tuple(x) \
                         for sublist in list_of_lists \
                         for x in (sublist if isinstance(sublist, list) else [sublist]))
    
    return flattened_list


def remove_clutter():
    """
    """
    # Define the root directory
    root_directory = os.getcwd()

    # Iterate through the files in the root directory
    for file_name in os.listdir(root_directory):
        file_path = os.path.join(root_directory, file_name)
        
        # Check if it's a file and ends with .png
        if os.path.isfile(file_path) and file_name.lower().endswith(".png"):
            try:
                os.remove(file_path)  # Remove the file
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
