import sys, os
from pathlib import Path
import asyncio
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import pandas as pd
import ast


from efs_parsing.utilities.settings import IMAGE_RESOLUTION_SCALE, GENERATE_PAGE_IMAGES, GENERATE_PICTURE_IMAGES, GENERATE_TABLE_IMAGES 
from efs_parsing.utilities.settings import IMAGE_COMPRESSION_FORMAT, AZURE_OPENAI_MODEL_NAME, DOCLING_TRANSFORMER_MODEL_PATH

from efs_parsing.utilities.llm_structured_output import SlideInfo
from efs_parsing.utilities.picture_utils import local_image_to_data_url, compress_and_resize_image, process_image_insight_extraction
from efs_parsing.utilities.picture_utils import create_img_markdown_placeholder

from efs_parsing.prompt.prompt_mgmt import select_appropriate_extract_prompt

def setup_pipeline_conversion():
    """
    """
    ## TODO: need to transfer this function and make it work for both pdf and word document format and images too
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = GENERATE_PAGE_IMAGES
    pipeline_options.generate_picture_images = GENERATE_PICTURE_IMAGES
    pipeline_options.generate_table_images = GENERATE_TABLE_IMAGES
    pipeline_options.artifacts_path = DOCLING_TRANSFORMER_MODEL_PATH

    doc_converter = DocumentConverter(
            allowed_formats=[
                    InputFormat.PDF,
                ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline, 
                                                 backend=PyPdfiumDocumentBackend,
                                                 pipeline_options=pipeline_options)
            }
        )
    
    return doc_converter

def save_pdf_document_artefacts(conv_res, 
                                doc_filename, 
                                page_img_output_folder_path, 
                                images_output_folder_path, 
                                table_output_folder_path,
                                docling_table_item,
                                docling_picture_item
                                    ):
    """
    """
    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename =  page_img_output_folder_path / f"{doc_filename}-{page_no}.png"
        
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, docling_table_item):
            table_counter += 1
            element_image_filename = (
                table_output_folder_path / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, docling_picture_item):
            picture_counter += 1
            element_image_filename = (
                images_output_folder_path / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")


    for table_ix, table in enumerate(conv_res.document.tables):
        
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as csv
        element_csv_filename = table_output_folder_path / f"{doc_filename}-table-{table_ix+1}.csv"
        
        table_df.to_csv(element_csv_filename)

        # Save the table as html
        element_html_filename = table_output_folder_path / f"{doc_filename}-table-{table_ix+1}.html"
        
        try:
            with element_html_filename.open("w") as fp:
                fp.write(table.export_to_html())
        except Exception as e:
            print(f"Error {e} with initial table saving")
            try:
                with element_html_filename.open("w", encoding="utf-8") as fp:
                    fp.write(table.export_to_html())
            except:
                print(f"Failure saving table from file {doc_filename} due to {e} with initial table saving")
            

def instantiate_custom_parser(conv_res, 
                              file_output_folder,
                              extracted_file_data,
                              docling_image_ref_mode):
    """
    """

    extracted_file_data.schema_name = conv_res.document.schema_name
    extracted_file_data.version = conv_res.document.version
    extracted_file_data.name = conv_res.document.name
    extracted_file_data.origin = conv_res.document.origin
    extracted_file_data.furniture = conv_res.document.furniture
    extracted_file_data.body = conv_res.document.body
    extracted_file_data.groups = conv_res.document.groups
    extracted_file_data.texts = conv_res.document.texts
    extracted_file_data.pictures = conv_res.document.pictures
    extracted_file_data.tables = conv_res.document.tables
    extracted_file_data.key_value_items = conv_res.document.key_value_items
    extracted_file_data.pages = conv_res.document.pages
    
    ## send the extracted specific tables and images into the md_filename folder
    md_filename = file_output_folder / f"image-references.md"
    artifacts_dir, reference_path = extracted_file_data._get_output_paths(md_filename)
    os.makedirs(artifacts_dir, exist_ok=True)
    ## create a copy of the docling object to ensure proper references path
    extracted_file_data_with_image_refs = extracted_file_data._make_copy_with_refmode(artifacts_dir, 
                                                                                      docling_image_ref_mode, 
                                                                                      reference_path=reference_path)

    return extracted_file_data_with_image_refs

    
    
def extract_content_complex_pdf_pages(client, 
                                      page_for_full_base64_encoding,
                                      page_img_output_folder_path,
                                      doc_filename,
                                      compressed_images_output_folder_path):
    """
    """
    
    selected_prompt = select_appropriate_extract_prompt({'complex_slide': True})
    page_img_repo = []
    img_path_repo = {}
    for page in page_for_full_base64_encoding:
        compressed_image_path = Path(str(compressed_images_output_folder_path) + f"\\compressed_pdf_page_{page}_for_{doc_filename}.{IMAGE_COMPRESSION_FORMAT}")
        image_path = compress_and_resize_image(image_path=page_img_output_folder_path / f"{doc_filename}-{page}.png", 
                                               output_path=compressed_image_path,
                                               quality=70,
                                               grayscale=True) ## this action reduces on average the token count by 30%
        img_path_repo[page] = image_path
        image_data_url = local_image_to_data_url(image_path)
        page_img_repo.append([page, image_data_url])
        
    
    image_extraction_analysis = asyncio.run(process_image_insight_extraction(list_of_images=page_img_repo,
                                                                             prompt=selected_prompt, 
                                                                             client=client, 
                                                                             azure_openai_model_name=AZURE_OPENAI_MODEL_NAME,
                                                                             response_format=SlideInfo
                                                                            ))
    extracted_page_content_repo = {}
    for pdf_page_analysis in image_extraction_analysis:
        pdf_page_number = pdf_page_analysis[0]
        img_page_path = img_path_repo[pdf_page_number]
        markdown_slide_placeholder = create_img_markdown_placeholder(img_page_path, pdf_page_number)
        extracted_page_content_repo[pdf_page_number] = {}
        
        pdf_page_content = pdf_page_analysis[1].choices[0].message.content
        usage_data = pdf_page_analysis[1].usage
        try:
            pdf_page_content = ast.literal_eval(pdf_page_content) # convert the previous string output to a readable python dict
        except:
            print(f"Extraction Generation issue for fully base 64 encoded slide number {page}")
        
        extracted_page_content_repo[pdf_page_number]["pdf_page_title"] = pdf_page_content["slide_title"]
        extracted_page_content_repo[pdf_page_number]["pdf_page_content"] = pdf_page_content["content_summary"] \
                                                                                        + "\n\n" + " ## [Complex] Source Page:" \
                                                                                        + "\n\n" + markdown_slide_placeholder
        extracted_page_content_repo[pdf_page_number]["completion_tokens"] = usage_data.completion_tokens
        extracted_page_content_repo[pdf_page_number]["prompt_tokens"] = usage_data.prompt_tokens
        extracted_page_content_repo[pdf_page_number]["total_tokens"] = usage_data.total_tokens
     
    return extracted_page_content_repo