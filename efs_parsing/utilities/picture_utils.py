import sys, os
from PIL import Image
import asyncio
import base64
from mimetypes import guess_type
import comtypes.client
import time
from pathlib import Path


from efs_parsing.utilities.settings import IMAGE_COMPRESSION_QUALITY, IMAGE_COMPRESSION_FORMAT, IMAGE_COMPRESSION_MAX_WIDTH, IMAGE_COMPRESSION_MAX_HEIGHT
from efs_parsing.utilities.llm_utils import generate_response_from_image_input

def create_img_markdown_placeholder(image_path, page):
    """
    """
    cwd = os.getcwd()
    relative_img_path = os.path.relpath(image_path, cwd)       
    full_path = "\\" + os.path.relpath(relative_img_path, cwd)
    full_path = full_path.replace("\\", "/")
    markdown_slide_placeholder = f"![Compressed_slide {page}]({full_path})"
    
    return markdown_slide_placeholder


def compress_and_resize_image(image_path, 
                              output_path, 
                              quality=IMAGE_COMPRESSION_QUALITY,
                              max_width=IMAGE_COMPRESSION_MAX_WIDTH, 
                              max_height=IMAGE_COMPRESSION_MAX_HEIGHT,
                              grayscale=False):
    """
    Resize an image while maintaining its aspect ratio.
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        max_width (int): Maximum width of the resized image.
        max_height (int): Maximum height of the resized image.
        quality (int): Compression quality (1-100, higher is better quality).
        grayscale(bool): decides to applying grayscale to further shrink image matrix or not
    Returns:
        str: Path to the resized image.
    """
    if grayscale:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
    else:
        image = Image.open(image_path)
    image.thumbnail((max_width, max_height))
    # Convert RGBA to RGB if needed
    if image.mode in ["RGBA", "P"]:
        image = image.convert("RGB")
    
    image.save(output_path, format=IMAGE_COMPRESSION_FORMAT, quality=quality)
    print("successful initial saving")
        
    return output_path

def local_image_to_data_url(image_path):
    """
    Convert a local image file to a data URL.

    Parameters:
    -----------
    image_path : str
        The path to the local image file to be converted.

    Returns:
    --------
    str
        A data URL representing the image, suitable for embedding in HTML or other web contexts.
    """
    # Get mime type
    mime_type, _ = guess_type(image_path)

    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

def save_pptx_slide(pptx_path, 
                    slide_index, 
                    image_path):
    """
    """
    try:
        comtypes.CoInitialize()
        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
        powerpoint.Visible = 1
        
        # Set AutomationSecurity to Low (1) to allow macros and interactive content
        powerpoint.AutomationSecurity = 1  # msoAutomationSecurityLow

        pptx_path = pptx_path.replace("/", "\\") ## comtype path need to be absolute path with \\
        # Open the PowerPoint presentation
        presentation = powerpoint.Presentations.Open(pptx_path)

        # Get the slide by index
        slide = presentation.Slides(slide_index)  

       # Retry logic for exporting slide
        retries = 1
        for attempt in range(retries):
            try:
                slide.Export(image_path, "PNG")
                print(f"Slide exported successfully to {image_path} - slide index {slide_index}")
                break
            except comtypes.COMError as e:
                print(f"Error exporting slide: {e} - slide index {slide_index}")
                if attempt < retries - 1:
                    print(f"Retrying... ({attempt + 1}/{retries})")
                    time.sleep(2)  # Wait before retrying
                else:
                    image_path = Path(image_path)
                    image_path = image_path.with_name(f"slide_{slide_index}_for_file.pptx.png")
                    try:
                        slide.Export(str(image_path), "PNG")
                        print(f"Slide exported successfully to {image_path} - slide index {slide_index}")
                    except:
                        print("Failed to export after retries and renaming")
    except Exception as e:
        print(f"Complete failure exporting slide: {e} - - slide index {slide_index}")
    finally:
        # Close the presentation and PowerPoint application
        if 'presentation' in locals():
            presentation.Close()
        if 'powerpoint' in locals():
            powerpoint.Quit()
        
        comtypes.CoUninitialize()
    
    return image_path


def convert_slide_to_image(pptx_path,
                           slide_images_output_folder_path,
                           slide_index, 
                           file_name,
                           mode="initial_compression"):
    """
    Converts a specific slide from the PowerPoint presentation to an image.

    Args:
        ppt_path (str): Path to the PowerPoint file.
        slide_index (int): The index of the slide to convert (0-based index).

    Returns:
        Image object: PIL Image object containing the slide as an image.
    """
    corrected_file_name = file_name.replace(" ", "_")
    image_path = str(slide_images_output_folder_path) + f"\\slide_{slide_index}_for_{corrected_file_name}.png" ## comtype path need to be absolute path with \\
    compressed_image_path = str(slide_images_output_folder_path) + f"\\compressed_slide_{slide_index}_for_{corrected_file_name}.{IMAGE_COMPRESSION_FORMAT}"
    
    if mode == "initial_compression":
        # Initialize PowerPoint application (Windows only)
        try:
            image_path = save_pptx_slide(pptx_path, slide_index, image_path)
        except Exception as e:
            try:
                print(f"Error: {e} - let us try again {file_name} slide_index - {slide_index}")
                image_path = save_pptx_slide(pptx_path, slide_index, image_path)
            except Exception as e:
                try:
                    print(f"Error: {e} - let us try again third time {file_name} slide_index - {slide_index}")
                    image_path = save_pptx_slide(pptx_path, slide_index, image_path)
                except:
                    print(f"Error: {e} - {file_name} slide_index - {slide_index}")
                    
        ## compress image to make sure we are not going over the context window limit
    
        image_path = compress_and_resize_image(image_path=image_path, 
                                               output_path=compressed_image_path)
    if mode == "secondary_aggresive_compression":
        ## ALWAYS CALLED A SECOND TIME
        
        image_path = compress_and_resize_image(image_path=compressed_image_path, 
                                               output_path=compressed_image_path,
                                               quality=85,
                                               max_width=768, 
                                               max_height=432,
                                               grayscale=True)

    return image_path

## parallelizing image processing in the same slide
async def process_image_insight_extraction(list_of_images,
                                           prompt, 
                                           client, 
                                           azure_openai_model_name, 
                                           response_format
                                           ):
    """
    """
    loop = asyncio.get_event_loop()
    tasks = []
    for items in list_of_images:
        img_key = items[0]
        # print(f"Processing Image Placeholder {img_key}")
        img_data_url = items[1]
        tasks.append(loop.run_in_executor(None, generate_response_from_image_input, prompt, 
                                                                                    img_data_url, 
                                                                                    client, 
                                                                                    azure_openai_model_name, 
                                                                                    response_format,
                                                                                    {'img_key':img_key}
                                                                                    ))
    responses = await asyncio.gather(*tasks)
    return responses