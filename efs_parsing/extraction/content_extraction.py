import sys, os
import time
from pathlib import Path
import asyncio



from efs_parsing.utilities.settings import EXTRACTED_PPTX_FILE_OUTPUT_ROOT_FOLDER, EXTRACTED_PDF_FILE_OUTPUT_ROOT_FOLDER, PDF_PARSING_METHOD
from efs_parsing.utilities.utils import remove_clutter, flatten_list_of_list, replace_between_markers

from efs_parsing.custom_loader.unstructured import powerpoint_custom_loader as pptxucl
from efs_parsing.custom_loader.docling import pdf_custom_loader as pdfdcl

from efs_parsing.analysis.page_analysis import analyze_pptx_pages, analyze_pdf_pages, triage_pages_for_base64_encoding
from efs_parsing.analysis.picture_analysis import identify_pictures_relevance, analyze_non_redundant_pictures
from efs_parsing.extraction.pptx.slide_extraction import extract_speaker_notes, process_slide_extraction
from efs_parsing.extraction.pdf.pdf_extraction import setup_pipeline_conversion, save_pdf_document_artefacts 
from efs_parsing.extraction.pdf.pdf_extraction import instantiate_custom_parser, extract_content_complex_pdf_pages
from efs_parsing.conversion.efs_file_conversion import convert_to_markdown_and_docx_document, markdown_to_docx


def create_extracted_file_artefact_folders(doc_filename,
                                           format_type = "pdf"):
    """
    """
    doc_filename = doc_filename.replace(" ", "_") # make sure there are no spaces
    
    
    if format_type == "pdf":
        output_dir = Path(os.getcwd() + EXTRACTED_PDF_FILE_OUTPUT_ROOT_FOLDER)
        
        file_output_folder = Path(str(output_dir) + f"\\{doc_filename}")
        os.makedirs(file_output_folder, exist_ok=True)
        # print(f"appropriate output folder for extracted pdf file {doc_filename} has been created")
        
        ## create appropriate sub folders for the page output, the images output and the tables output
        # pages img output
        page_img_output_folder_path = Path(str(file_output_folder) + f"\\pages_img_output")
        os.makedirs(page_img_output_folder_path, exist_ok=True)
        # print(f"appropriate page images output folder for file {doc_filename} has been created")

        # images output folder path
        images_output_folder_path = Path(str(file_output_folder) + f"\\images_output")
        os.makedirs(images_output_folder_path, exist_ok=True)
        # print(f"appropriate images output folder for file {doc_filename} has been created")

        table_output_folder_path = Path(str(file_output_folder) + f"\\tables_output")
        os.makedirs(table_output_folder_path, exist_ok=True)
        # print(f"appropriate tables output folder for file {doc_filename} has been created")

        ## compressed_images output folder
        compressed_images_output_folder_path = Path(str(file_output_folder) + f"\\compressed_pdf_images")
        os.makedirs(compressed_images_output_folder_path, exist_ok=True)
        # print(f"appropriate compressed images folder for file {doc_filename} has been created")
        
        return file_output_folder, page_img_output_folder_path, images_output_folder_path, table_output_folder_path, \
            compressed_images_output_folder_path
        
    if format_type == "pptx":
        output_dir = Path(os.getcwd() + EXTRACTED_PPTX_FILE_OUTPUT_ROOT_FOLDER)
        
        file_output_folder = Path(str(output_dir) + f"\\{doc_filename}")
        os.makedirs(file_output_folder, exist_ok=True)
        # print(f"appropriate output folder for file {doc_filename} has been created")
        
        ## slide output folder (as extracted images)
        slide_images_output_folder_path = Path(str(file_output_folder) + f"\\pages_img_output")
        os.makedirs(slide_images_output_folder_path, exist_ok=True)
        # print(f"appropriate extracted slide folder for file {doc_filename} has been created")
        
        ## images output folder (as extracted images)
        images_output_folder_path = Path(str(file_output_folder) + f"\\images_output")
        os.makedirs(images_output_folder_path, exist_ok=True)
        # print(f"appropriate images folder for file {doc_filename} has been created")
        
        return file_output_folder, slide_images_output_folder_path, images_output_folder_path


def extract_data_and_metadata_from_file(file_path, 
                                        client,
                                        word_doc_out_path, 
                                        md_ind_pages_from_pptx, 
                                        md_doc_out_path):
    """
    """
    
    file_repo_data_and_metadata = {} # to keep all the processed items in memory for additional use in the future
    if file_path.lower().endswith(".pptx"):
        
        _, slide_images_output_folder_path, _ = create_extracted_file_artefact_folders(Path(file_path).stem,
                                                                                       format_type = "pptx")
        
        print("Processing a .pptx file")
        # loading
        beginning_time = time.time()
        print(f"Initial pptx parsing for {file_path}")
        pptx_loader = pptxucl.UnstructuredPowerPointLoader(file_path, mode="elements") 
        extracted_file_data = pptx_loader.load()
        file_name = extracted_file_data[0].metadata["filename"]

        # conduct slide analysis to triage for full page base 64 encoding
        file_pages_analytics = analyze_pptx_pages(extracted_file_data,
                                                  pptxucl.SHAPE_TYPE_MAPPING.items(),
                                                  pptxucl.DOCUMENT_SHAPE_ANALYZER_OBJECT_NAME,
                                                  pptxucl.DOCUMENT_MATH_SHAPE_ANALYZER_OBJECT_NAME,
                                                  pptxucl.EQUATION_SHAPES,
                                                  pptxucl.NOT_MATH_SHAPE_LABEL)
        print(f"PPTX: {file_pages_analytics}")
        # print(f"FILE PAGE ANALYTICS: {file_pages_analytics}")
        # triaging of slides for base64 encoding: complex slides with numerous figures and mathematical equations should # be converted to base64 encoding
        page_for_full_base64_encoding = triage_pages_for_base64_encoding(file_pages_analytics,
                                                                         parsing_mode= "pptx")
        print(f"PPTX: {page_for_full_base64_encoding}")
        # presentation speaker notes
        notes = extract_speaker_notes(file_path)
        ## TODO: implement feature to flag unnecessary pictures (logos and infographics)
        list_pages = list(file_pages_analytics.keys())
        
        ## PARALLEL SLIDE EXTRACTIION PROCESSING
        extracted_slide_content = asyncio.run(process_slide_extraction(list_pages,
                                                                       page_for_full_base64_encoding,
                                                                       file_name,
                                                                       file_path,
                                                                       slide_images_output_folder_path,
                                                                       client,
                                                                       notes,
                                                                       extracted_file_data,
                                                                       pptxucl.DOCUMENT_SHAPE_ANALYZER_OBJECT_NAME, 
                                                                       pptxucl.DOCUMENT_MATH_SHAPE_ANALYZER_OBJECT_NAME))
        
        
        # Sorting the list by the first key of each dictionary
        ordered_extracted_slide_content = sorted(extracted_slide_content, key=lambda d: next(iter(d.keys())))
        final_time = time.time()
        file_execution_time = final_time - beginning_time
        # Print time it takes to process extraction
        print(f"Parsing and Extraction of pptx is: \
                Execution Time: {file_execution_time:.6f} seconds")
        
        ## Once the file has all their pages extracted and analyzed we can now get the final markdown
        ## and final docx file
        ## Create the separate pages in markdown and create the final artefact in md and docx
        # for the single file being processed
        convert_to_markdown_and_docx_document(file_name,
                                              word_doc_out_path, 
                                              md_ind_pages_from_pptx, 
                                              md_doc_out_path,
                                              ordered_extracted_slide_content)
        
        file_execution_time = 0
        file_token_consumption = 0
        for page in ordered_extracted_slide_content:
            for idx, page_metadata in page.items():
                file_execution_time += page_metadata["processing_time_seconds"]
                file_token_consumption += page_metadata["open_ai_accounting_total_tokens"]
        
        ## keep all data and metadata regarding the slide and their different pages (includes execution time, tokens consumption..etc)
        ordered_extracted_slide_content[-1] = {"file_processing_statistics":{
                                                                            "file_execution_time": file_execution_time,
                                                                            "total_token_consumption": file_token_consumption
                                                                            }
                                              }
        file_repo_data_and_metadata[file_path] = ordered_extracted_slide_content
        
        # ## remove clutered files in root directory
        remove_clutter()
    
    if file_path.lower().endswith("pdf"):
        
        # file_list = ["C:/Users/fbenichou/projects/gen-ai-coe/EFS_prototype/EFS-POC-1 4/efs_poc-1/input_files/open-banking-consumer-protection.pdf"]
        
        beginning_time = time.time()
        doc_converter = setup_pipeline_conversion()
        print(f"Initial pdf parsing for {file_path}")
        conv_res = doc_converter.convert(Path(file_path))
        
        total_number_pages = conv_res.document.pages[list(conv_res.document.pages.keys())[-1]].page_no
        
        doc_filename = conv_res.input.file.stem

        ## analyze pdf pages
        file_pages_analytics = analyze_pdf_pages(conv_res,
                                                 pdfdcl.DEFAULT_EXPORT_LABELS)
        # print(f"FILE PAGE ANALYTICS: {file_pages_analytics}")
        
        ## create appropriate folder for file under appropriate pdf output
        file_output_folder, page_img_output_folder_path, images_output_folder_path, table_output_folder_path, \
                    compressed_images_output_folder_path = create_extracted_file_artefact_folders(doc_filename,
                                                                                                  format_type = "pdf")
        save_pdf_document_artefacts(conv_res, 
                                    doc_filename, 
                                    page_img_output_folder_path, 
                                    images_output_folder_path, 
                                    table_output_folder_path,
                                    pdfdcl.TableItem,
                                    pdfdcl.PictureItem
                                    )
        print(file_pages_analytics)
        page_for_full_base64_encoding = triage_pages_for_base64_encoding(file_pages_analytics,
                                                                         parsing_mode= "pdf")
        print(page_for_full_base64_encoding)
        
        if bool(page_for_full_base64_encoding):
            complex_pdf_pages_content = extract_content_complex_pdf_pages(client,
                                                                        page_for_full_base64_encoding,
                                                                        page_img_output_folder_path,
                                                                        doc_filename,
                                                                        compressed_images_output_folder_path)
        else:
            complex_pdf_pages_content = {}
            
        if bool(complex_pdf_pages_content):
            complex_pdf_pages_content_token_consumption = sum([page_content['total_tokens'] for _, page_content in complex_pdf_pages_content.items()])
            complex_pdf_pages_list = list(complex_pdf_pages_content.keys())
        else:
            complex_pdf_pages_content_token_consumption = 0
            complex_pdf_pages_list = []
        
        ## instantiate the custom docling object that supports custom parsing into markdown
        extracted_file_data_with_image_refs = instantiate_custom_parser(conv_res, 
                                                                        file_output_folder,
                                                                        pdfdcl.DoclingDocument(name="dummy"),
                                                                        pdfdcl.ImageRefMode.REFERENCED)
        
        
        ## before analyzing the pdf pages, we are going to assess the pictures of the pdf and see those
        ## that should not taken into account for knowledge extraction because they are irrelevant
        redundant_picture_set_analysis = identify_pictures_relevance(client,
                                                                     compressed_images_output_folder_path,
                                                                     total_number_pages,
                                                                     PDF_PARSING_METHOD,
                                                                     extracted_file_data_with_image_refs, 
                                                                     mode = "redundant_pictures")
        if bool(redundant_picture_set_analysis):
            redundant_picture_set_analysis_token_consumption = sum([analysis[2] for _, analysis in redundant_picture_set_analysis.items()])
        else:
            redundant_picture_set_analysis_token_consumption = 0
        
        ## mark all REDUNDANT irrelevant pictures as such in the docling object that stores the data 
        extracted_file_data_with_image_refs.classify_pictures_relevance(redundant_picture_set_analysis) ## this docling object method is NEW and was not part of the existing docling package
        
        ## assess for relevance of all other pictures that are not redundant as analyzed above
        redundant_pictures_list_of_lists = [item[1][1] for item in redundant_picture_set_analysis.items()]
        redundant_picture_set = flatten_list_of_list(redundant_pictures_list_of_lists)
        # indvidual list of unique picture to be analyzed + make sure the picture is not in any complex pdf pages
        non_redundant_pictures_set = [[pic_key] for pic_key in range(len(extracted_file_data_with_image_refs.pictures)) \
                                        if (pic_key not in redundant_picture_set) and 
                                        (extracted_file_data_with_image_refs.pictures[pic_key].prov[0].page_no \
                                        not in complex_pdf_pages_list)] 
        
        try:
            non_redundant_picture_analysis_repo = analyze_non_redundant_pictures(client,
                                                                                 non_redundant_pictures_set,
                                                                                 compressed_images_output_folder_path,
                                                                                 total_number_pages,
                                                                                 PDF_PARSING_METHOD, 
                                                                                 extracted_file_data_with_image_refs)
            
            if bool(non_redundant_picture_analysis_repo):
                non_redundant_picture_analysis_repo_token_consumption = sum([analysis[2] for _, analysis in non_redundant_picture_analysis_repo[0].items()])
            else:
                non_redundant_picture_analysis_repo_token_consumption = 0
        except Exception as e:
            print(f"Error analysing non redundant pictures: {e}")
            non_redundant_picture_analysis_repo = [{}]
            non_redundant_picture_analysis_repo_token_consumption = 0
        
        ## mark all NON REDUNDANT irrelevant pictures SET as such in the docling object that stores the data
        for picture_set_analysis in non_redundant_picture_analysis_repo:
            if bool(picture_set_analysis):
                extracted_file_data_with_image_refs.classify_pictures_relevance(picture_set_analysis)
            
        final_time = time.time()
        file_execution_time = final_time - beginning_time
        # Print time it takes to process extraction
        print(f"Parsing and Extraction of pdf file is: \
                Execution Time: {file_execution_time:.6f} seconds")
        md_filename = Path(str(md_doc_out_path) + f"\\{doc_filename}.md")
        markdown_output = extracted_file_data_with_image_refs.save_as_markdown(md_filename)
        ## remove elements from pages that are complex pages (aka pages that are too difficult to parse with the parser only)
        if bool(complex_pdf_pages_content):
            for page_number, complex_page_pdf in complex_pdf_pages_content.items():
                start_page_marker = f"<!-- Start of {doc_filename}-page-{page_number}.md -->"
                end_page_marker = f"<!-- End of {doc_filename}-page-{page_number}.md -->"
                title = "\n\n## " + complex_page_pdf['pdf_page_title'] + "\n\n"
                page_content = complex_page_pdf['pdf_page_content']
                final_reconstructed_page = title + page_content + "\n\n"
                markdown_output = replace_between_markers(markdown_output, start_page_marker, end_page_marker, final_reconstructed_page)
        
            try:
                with open(md_filename, "w") as fw:
                    fw.write(markdown_output)
            except:
                try:
                    with open(md_filename, "w", encoding="utf-8") as fw:
                        fw.write(markdown_output)
                except Exception as e:
                    print(f"Error creating the markdown output. Exception is {e}")
            
        final_docx_path = str(word_doc_out_path) + f"\\{doc_filename}_final_output_conversion.docx"
        markdown_to_docx(str(md_filename), final_docx_path)
        print(f"Converted to final docx file at:{final_docx_path}")
        
        ## keep all data and metadata regarding the slide and their different pages (includes execution time, tokens consumption..etc)
        file_repo_data_and_metadata[file_path] = {"extracted_markdown_output": markdown_output,
                                                  "file_execution_time": file_execution_time,
                                                  "total_token_consumption": complex_pdf_pages_content_token_consumption + \
                                                                             non_redundant_picture_analysis_repo_token_consumption + \
                                                                             redundant_picture_set_analysis_token_consumption
                                                                            }
        
        # ## remove clutered files in root directory
        remove_clutter()
        
    return file_repo_data_and_metadata
