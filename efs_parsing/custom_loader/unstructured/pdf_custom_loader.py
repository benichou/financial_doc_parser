from __future__ import annotations

import base64
import os
import re
import tempfile

from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, List, Optional, Tuple, cast, Any

import pdf2image

from unstructured.documents.elements import ElementType
from unstructured.logger import logger
from unstructured.partition.common.common import convert_to_bytes, exactly_one
from unstructured.partition.utils.config import env_config
from PIL import Image as PILImage



if TYPE_CHECKING:
    from unstructured_inference.inference.elements import TextRegion
    from unstructured_inference.inference.layout import DocumentLayout, PageLayout
    from unstructured_inference.inference.layoutelement import LayoutElement

    from unstructured.documents.elements import Element


from unstructured.partition.pdf_image.pdf_image_utils import convert_pdf_to_image, pad_bbox, write_image
import pytesseract
import unstructured_pytesseract

DOCUMENT_SHAPE_ANALYZER_OBJECT_NAME = 'PAGE_SHAPE_ANALYTICS_HIGH_LEVEL' ## tweak to package

############################### MODIFICATION TO THE SAVE_ELEMENTS FUNCTION TO ALLOW SAVING OF IMAGES LOCALLY ALONGSIDE
############################### SAVING OF IMAGE BASE 64 ENCODING #####################################################
#
def save_elements(
    elements: List["Element"],
    starting_page_number: int,
    element_category_to_save: str,
    pdf_image_dpi: int,
    filename: str = "",
    file: bytes | IO[bytes] | None = None,
    is_image: bool = False,
    extract_image_block_to_payload: bool = False,
    output_dir_path: str | None = None,
):
    """
    Saves specific elements from a PDF as images either to a directory or embeds them in the
    element's payload.

    This function processes a list of elements partitioned from a PDF file. For each element of
    a specified category, it extracts and saves the image. The images can either be saved to
    a specified directory or embedded into the element's payload as a base64-encoded string.
    """

    # Determine the output directory path
    # if extract_image_block_to_payload:  ### used to be 'if not extract_image_block_to_payload'
    print("checking directory existence")
    output_dir_path = output_dir_path or (
        str(Path(env_config.GLOBAL_WORKING_PROCESS_DIR) / "extracted_images/pdf/")
        if env_config.GLOBAL_WORKING_DIR_ENABLED
        else str(Path.cwd() / "extracted_images/pdf/")
    )

    os.makedirs(output_dir_path, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        if is_image:
            if file is None:
                image_paths = [filename]
            else:
                if isinstance(file, bytes):
                    file_data = file
                else:
                    file.seek(0)
                    file_data = file.read()

                tmp_file_path = os.path.join(temp_dir, "tmp_file")
                with open(tmp_file_path, "wb") as tmp_file:
                    tmp_file.write(file_data)
                image_paths = [tmp_file_path]
        else:
            _image_paths = convert_pdf_to_image(
                filename,
                file,
                pdf_image_dpi,
                output_folder=temp_dir,
                path_only=True,
            )
            image_paths = cast(List[str], _image_paths)

        figure_number = 0
        for el in elements:
            if el.category != element_category_to_save:
                continue

            coordinates = el.metadata.coordinates
            if not coordinates or not coordinates.points:
                continue

            points = coordinates.points
            x1, y1 = points[0]
            x2, y2 = points[2]
            h_padding = env_config.EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD
            v_padding = env_config.EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD
            padded_bbox = cast(
                Tuple[int, int, int, int], pad_bbox((x1, y1, x2, y2), (h_padding, v_padding))
            )

            # The page number in the metadata may have been offset
            # by starting_page_number. Make sure we use the right
            # value for indexing!
            assert el.metadata.page_number
            metadata_page_number = el.metadata.page_number
            page_index = metadata_page_number - starting_page_number

            figure_number += 1
            try:
                image_path = image_paths[page_index]
                image = PILImage.open(image_path)
                cropped_image = image.crop(padded_bbox)
                if extract_image_block_to_payload:
                    buffered = BytesIO()
                    cropped_image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue())
                    img_base64_str = img_base64.decode()
                    el.metadata.image_base64 = img_base64_str
                    el.metadata.image_mime_type = "image/jpeg"
                    ## ADDED THE FOLLOWING CODE TOO THAT NEEDS TO BE REMOVED LATER
                    basename = "table" if el.category == ElementType.TABLE else "image"
                    assert output_dir_path
                    output_f_path = os.path.join(
                        output_dir_path,
                        f"{Path(filename).stem}_page_{metadata_page_number}_{basename}_{figure_number}.jpg", # tweak from package 
                    )
                    write_image(cropped_image, output_f_path)
                    print(f"saved image at {output_f_path}")
                    # add image path to element metadata
                    el.metadata.image_path = output_f_path
                else:
                    basename = "table" if el.category == ElementType.TABLE else "figure"
                    assert output_dir_path
                    output_f_path = os.path.join(
                        output_dir_path,
                        f"{Path(filename).stem}_page_{metadata_page_number}_{basename}_{figure_number}.jpg", # tweak from package
                    )
                    write_image(cropped_image, output_f_path)
                    print(f"saved image at {output_f_path}")
                    # add image path to element metadata
                    el.metadata.image_path = output_f_path
            except (ValueError, IOError):
                print("FAILURE")
                logger.warning("Image Extraction Error: Skipping the failed image", exc_info=True)



## below tweak to package
# Overwrite the function in the module
import sys
sys.modules['unstructured.partition.pdf_image.pdf_image_utils'].save_elements = save_elements

############################### MODIFICATION TO THE SAVE_ELEMENTS FUNCTION TO ALLOW SAVING OF IMAGES LOCALLY ALONGSIDE
############################### SAVING OF IMAGE BASE 64 ENCODING #####################################################

############################### MODIFICATION OF PARTITION_PDF WITH DIFFERENT PARAMETERS TO RUN HI RES EXTRACTION ######
############################### WHILE SAVING IMAGES IN PRE-DETERMINED FILE LOCATION ###################################

from unstructured.chunking import add_chunking_strategy
from unstructured.file_utils.filetype import add_metadata_with_filetype
from unstructured.documents.elements import (
    CoordinatesMetadata,
    Element,
    ElementMetadata,
    ElementType,
    Image,
    PageBreak,
    Text,
    Title,
    process_metadata,
)


from unstructured.partition.utils.constants import (
    OCR_AGENT_PADDLE,
    SORT_MODE_DONT,
    SORT_MODE_XY_CUT,
    OCRMode,
    PartitionStrategy,
)

from unstructured.partition.common.common import (
    add_element_metadata,
    exactly_one,
    get_page_image_metadata,
    normalize_layout_element,
    spooled_to_bytes_io_if_needed,
)
from unstructured.partition.common.lang import (
    check_language_args,
    prepare_languages_for_tesseract,
    tesseract_to_paddle_language,
)
from unstructured.partition.pdf_image.pdfminer_processing import (
    clean_pdfminer_inner_elements,
    get_links_in_element,
    merge_inferred_with_extracted_layout,
)

from unstructured.file_utils.model import FileType
from unstructured.partition.pdf import (partition_pdf_or_image, 
                                        extractable_elements, 
                                        _process_uncategorized_text_elements, 
                                        _partition_pdf_with_pdfparser, 
                                        check_pdf_hi_res_max_pages_exceeded, 
                                        _partition_pdf_or_image_with_ocr
                                        )

import os
import re
import warnings
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Optional, cast

from unstructured.partition.utils.sorting import coord_has_valid_points, sort_page_elements

from pi_heif import register_heif_opener

from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.inference.layoutelement import LayoutElement

from unstructured.chunking import add_chunking_strategy


from unstructured.partition.pdf_image.pdf_image_utils import (
    check_element_types_to_extract,
    save_elements,
)

from unstructured.file_utils.filetype import add_metadata_with_filetype
from unstructured.file_utils.model import FileType
from unstructured.logger import logger
from unstructured.nlp.patterns import PARAGRAPH_PATTERN

from unstructured.partition.common.metadata import get_last_modified_date
from unstructured.partition.pdf_image.analysis.layout_dump import (
    ExtractedLayoutDumper,
    FinalLayoutDumper,
    ObjectDetectionLayoutDumper,
    OCRLayoutDumper,
)
from unstructured.utils import first, requires_dependencies
from unstructured.partition.pdf_image.analysis.tools import save_analysis_artifiacts
from unstructured.documents.coordinates import PixelSpace, PointSpace
from unstructured.partition.strategies import determine_pdf_or_image_strategy, validate_strategy
from unstructured.partition.pdf_image.form_extraction import run_form_extraction

from unstructured.partition.utils.config import env_config
from unstructured.partition.utils.constants import (
    OCR_AGENT_PADDLE,
    PartitionStrategy,
)


if TYPE_CHECKING:
    pass

@requires_dependencies("unstructured_inference")
def default_hi_res_model() -> str:
    # a light config for the hi res model; this is not defined as a constant so that no setting of
    # the default hi res model name is done on importing of this submodule; this allows (if user
    # prefers) for setting env after importing the sub module and changing the default model name

    from unstructured_inference.models.base import DEFAULT_MODEL

    return os.environ.get("UNSTRUCTURED_HI_RES_MODEL_NAME", DEFAULT_MODEL)

@process_metadata()
@add_metadata_with_filetype(FileType.PDF)
@add_chunking_strategy
def partition_pdf(
    filename: Optional[str] = None,
    file: Optional[IO[bytes]] = None,
    include_page_breaks: bool = True, ## CHANGED
    strategy: str = "hi_res", # CHANGED PartitionStrategy.AUTO
    infer_table_structure: bool = True, ## CHANGED
    ocr_languages: Optional[str] = None,  # changing to optional for deprecation
    languages: Optional[list[str]] = None,
    metadata_filename: Optional[str] = None,  # used by decorator
    metadata_last_modified: Optional[str] = None,
    chunking_strategy: Optional[str] = None,  # used by decorator
    hi_res_model_name: Optional[str] = None,
    extract_images_in_pdf: bool = True, ## CHANGED
    extract_image_block_types = ["Image", "Table"], #Optional[list[str]] = None,
    extract_image_block_output_dir = os.path.join(os.getcwd(), "extracted_images/pdf"),# Optional[str] = None,
    extract_image_block_to_payload: bool = True, # this prevents saving under path -- for now we cannot have both the base64 image and the image saved in a path apparently - need to chnage code
    starting_page_number: int = 1,
    extract_forms: bool = False,
    form_extraction_skip_tables: bool = True,
    **kwargs: Any,
) -> list[Element]:
    """Parses a pdf document into a list of interpreted elements.
    Parameters
    ----------
    filename
        A string defining the target filename path.
    file
        A file-like object as bytes --> open(filename, "rb").
    strategy
        The strategy to use for partitioning the PDF. Valid strategies are "hi_res",
        "ocr_only", and "fast". When using the "hi_res" strategy, the function uses
        a layout detection model to identify document elements. When using the
        "ocr_only" strategy, partition_pdf simply extracts the text from the
        document using OCR and processes it. If the "fast" strategy is used, the text
        is extracted directly from the PDF. The default strategy `auto` will determine
        when a page can be extracted using `fast` mode, otherwise it will fall back to `hi_res`.
    infer_table_structure
        Only applicable if `strategy=hi_res`.
        If True, any Table elements that are extracted will also have a metadata field
        named "text_as_html" where the table's text content is rendered into an html string.
        I.e., rows and cells are preserved.
        Whether True or False, the "text" field is always present in any Table element
        and is the text content of the table (no structure).
    languages
        The languages present in the document, for use in partitioning and/or OCR. To use a language
        with Tesseract, you'll first need to install the appropriate Tesseract language pack.
    metadata_last_modified
        The last modified date for the document.
    hi_res_model_name
        The layout detection model used when partitioning strategy is set to `hi_res`.
    extract_images_in_pdf
        Only applicable if `strategy=hi_res`.
        If True, any detected images will be saved in the path specified by
        'extract_image_block_output_dir' or stored as base64 encoded data within metadata fields.
        Deprecation Note: This parameter is marked for deprecation. Future versions will use
        'extract_image_block_types' for broader extraction capabilities.
    extract_image_block_types
        Only applicable if `strategy=hi_res`.
        Images of the element type(s) specified in this list (e.g., ["Image", "Table"]) will be
        saved in the path specified by 'extract_image_block_output_dir' or stored as base64
        encoded data within metadata fields.
    extract_image_block_to_payload
        Only applicable if `strategy=hi_res`.
        If True, images of the element type(s) defined in 'extract_image_block_types' will be
        encoded as base64 data and stored in two metadata fields: 'image_base64' and
        'image_mime_type'.
        This parameter facilitates the inclusion of element data directly within the payload,
        especially for web-based applications or APIs.
    extract_image_block_output_dir
        Only applicable if `strategy=hi_res` and `extract_image_block_to_payload=False`.
        The filesystem path for saving images of the element type(s)
        specified in 'extract_image_block_types'.
    extract_forms
        Whether the form extraction logic should be run
        (results in adding FormKeysValues elements to output).
    form_extraction_skip_tables
        Whether the form extraction logic should ignore regions designated as Tables.
    """
    print(f"Partitioning file {filename}")
    exactly_one(filename=filename, file=file)

    languages = check_language_args(languages or [], ocr_languages)

    return partition_pdf_or_image(
        filename=filename,
        file=file,
        include_page_breaks=include_page_breaks,
        strategy=strategy,
        infer_table_structure=infer_table_structure,
        languages=languages,
        metadata_last_modified=metadata_last_modified,
        hi_res_model_name=hi_res_model_name,
        extract_images_in_pdf=extract_images_in_pdf,
        extract_image_block_types=extract_image_block_types,
        extract_image_block_output_dir=extract_image_block_output_dir,
        extract_image_block_to_payload=extract_image_block_to_payload,
        starting_page_number=starting_page_number,
        extract_forms=extract_forms,
        form_extraction_skip_tables=form_extraction_skip_tables,
        **kwargs,
    )

def partition_pdf_or_image(
    filename: str = "",
    file: Optional[bytes | IO[bytes]] = None,
    is_image: bool = False,
    include_page_breaks: bool = False,
    strategy: str = PartitionStrategy.AUTO,
    infer_table_structure: bool = False,
    languages: Optional[list[str]] = None,
    metadata_last_modified: Optional[str] = None,
    hi_res_model_name: Optional[str] = None,
    extract_images_in_pdf: bool = False,
    extract_image_block_types: Optional[list[str]] = None,
    extract_image_block_output_dir: Optional[str] = None,
    extract_image_block_to_payload: bool = False,
    starting_page_number: int = 1,
    extract_forms: bool = False,
    form_extraction_skip_tables: bool = True,
    **kwargs: Any,
) -> list[Element]:
    """Parses a pdf or image document into a list of interpreted elements."""
    # TODO(alan): Extract information about the filetype to be processed from the template
    # route. Decoding the routing should probably be handled by a single function designed for
    # that task so as routing design changes, those changes are implemented in a single
    # function.

    if languages is None:
        languages = ["eng"]

    # init ability to process .heic files
    register_heif_opener()

    validate_strategy(strategy, is_image)

    last_modified = get_last_modified_date(filename) if filename else None

    extracted_elements = []
    pdf_text_extractable = False
    if not is_image:
        try:
            extracted_elements = extractable_elements(
                filename=filename,
                file=spooled_to_bytes_io_if_needed(file),
                languages=languages,
                metadata_last_modified=metadata_last_modified or last_modified,
                starting_page_number=starting_page_number,
                **kwargs,
            )
            pdf_text_extractable = any(
                isinstance(el, Text) and el.text.strip()
                for page_elements in extracted_elements
                for el in page_elements
            )
        except Exception as e:
            logger.debug(e)
            logger.info("PDF text extraction failed, skip text extraction...")

    strategy = determine_pdf_or_image_strategy(
        strategy,
        is_image=is_image,
        pdf_text_extractable=pdf_text_extractable,
        infer_table_structure=infer_table_structure,
        extract_images_in_pdf=extract_images_in_pdf,
        extract_image_block_types=extract_image_block_types,
    )

    if file is not None:
        file.seek(0)

    ocr_languages = prepare_languages_for_tesseract(languages)
    if env_config.OCR_AGENT == OCR_AGENT_PADDLE:
        ocr_languages = tesseract_to_paddle_language(ocr_languages)

    if strategy == PartitionStrategy.HI_RES:
        # NOTE(robinson): Catches a UserWarning that occurs when detection is called
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elements = _partition_pdf_or_image_local(
                filename=filename,
                file=spooled_to_bytes_io_if_needed(file),
                is_image=is_image,
                infer_table_structure=infer_table_structure,
                include_page_breaks=include_page_breaks,
                languages=languages,
                ocr_languages=ocr_languages,
                metadata_last_modified=metadata_last_modified or last_modified,
                hi_res_model_name=hi_res_model_name,
                pdf_text_extractable=pdf_text_extractable,
                extract_images_in_pdf=extract_images_in_pdf,
                extract_image_block_types=extract_image_block_types,
                extract_image_block_output_dir=extract_image_block_output_dir,
                extract_image_block_to_payload=extract_image_block_to_payload,
                starting_page_number=starting_page_number,
                extract_forms=extract_forms,
                form_extraction_skip_tables=form_extraction_skip_tables,
                **kwargs,
            )
            out_elements = _process_uncategorized_text_elements(elements)

    elif strategy == PartitionStrategy.FAST:
        out_elements = _partition_pdf_with_pdfparser(
            extracted_elements=extracted_elements,
            include_page_breaks=include_page_breaks,
            **kwargs,
        )

        return out_elements

    elif strategy == PartitionStrategy.OCR_ONLY:
        # NOTE(robinson): Catches file conversion warnings when running with PDFs
        with warnings.catch_warnings():
            elements = _partition_pdf_or_image_with_ocr(
                filename=filename,
                file=file,
                include_page_breaks=include_page_breaks,
                languages=languages,
                ocr_languages=ocr_languages,
                is_image=is_image,
                metadata_last_modified=metadata_last_modified or last_modified,
                starting_page_number=starting_page_number,
                **kwargs,
            )
            out_elements = _process_uncategorized_text_elements(elements)

    return out_elements

def document_to_element_list(
    document: DocumentLayout,
    sortable: bool = False,
    include_page_breaks: bool = False,
    last_modification_date: Optional[str] = None,
    infer_list_items: bool = True,
    source_format: Optional[str] = None,
    detection_origin: Optional[str] = None,
    sort_mode: str = SORT_MODE_XY_CUT,
    languages: Optional[list[str]] = None,
    starting_page_number: int = 1,
    layouts_links: Optional[list[list]] = None,
    **kwargs: Any,
) -> list[Element]:
    """Converts a DocumentLayout object to a list of unstructured elements."""
    elements: list[Element] = []

    num_pages = len(document.pages)
    for page_number, page in enumerate(document.pages, start=starting_page_number):
        page_elements: list[Element] = []

        page_image_metadata = get_page_image_metadata(page)
        image_format = page_image_metadata.get("format")
        image_width = page_image_metadata.get("width")
        image_height = page_image_metadata.get("height")

        translation_mapping: list[tuple["LayoutElement", Element]] = []

        links = (
            layouts_links[page_number - starting_page_number]
            if layouts_links and layouts_links[0]
            else None
        )

        for layout_element in page.elements:
            if image_width and image_height and hasattr(layout_element.bbox, "coordinates"):
                coordinate_system = PixelSpace(width=image_width, height=image_height)
            else:
                coordinate_system = None

            element = normalize_layout_element(
                layout_element,
                coordinate_system=coordinate_system,
                infer_list_items=infer_list_items,
                source_format=source_format if source_format else "html",
            )
            if isinstance(element, list):
                for el in element:
                    if last_modification_date:
                        el.metadata.last_modified = last_modification_date
                    el.metadata.page_number = page_number
                page_elements.extend(element)
                translation_mapping.extend([(layout_element, el) for el in element])
                continue
            else:

                element.metadata.links = (
                    get_links_in_element(links, layout_element.bbox) if links else []
                )

                if last_modification_date:
                    element.metadata.last_modified = last_modification_date
                element.metadata.text_as_html = getattr(layout_element, "text_as_html", None)
                element.metadata.table_as_cells = getattr(layout_element, "table_as_cells", None)

                if (isinstance(element, Title) and element.metadata.category_depth is None) and any(
                    getattr(el, "type", "") in ["Headline", "Subheadline"] for el in page.elements
                ):
                    element.metadata.category_depth = 0

                page_elements.append(element)
                translation_mapping.append((layout_element, element))
            coordinates = (
                element.metadata.coordinates.points if element.metadata.coordinates else None
            )

            el_image_path = (
                layout_element.image_path if hasattr(layout_element, "image_path") else None
            )

            add_element_metadata(
                element,
                page_number=page_number,
                filetype=image_format,
                coordinates=coordinates,
                coordinate_system=coordinate_system,
                category_depth=element.metadata.category_depth,
                image_path=el_image_path,
                detection_origin=detection_origin,
                languages=languages,
                **kwargs,
            )

        for layout_element, element in translation_mapping:
            if hasattr(layout_element, "parent") and layout_element.parent is not None:
                element_parent = first(
                    (el for l_el, el in translation_mapping if l_el is layout_element.parent),
                )
                element.metadata.parent_id = element_parent.id
        sorted_page_elements = page_elements
        if sortable and sort_mode != SORT_MODE_DONT:
            sorted_page_elements = sort_page_elements(page_elements, sort_mode)

        if include_page_breaks and page_number < num_pages + starting_page_number:
            sorted_page_elements.append(PageBreak(text=""))
        
        #####################  ADDED CODE ELEMENTS #################################################################
        elements_analytics = [x.to_dict()["type"] for x in sorted_page_elements]
        element_metadata = ElementMetadata(
            page_number=sorted_page_elements[0].to_dict()["metadata"]["page_number"],
            last_modified=sorted_page_elements[0].to_dict()["metadata"]["last_modified"],
            emphasized_text_contents=elements_analytics
        )
        sorted_page_elements.append(Text(text=DOCUMENT_SHAPE_ANALYZER_OBJECT_NAME, metadata=element_metadata)) ## added the following line
        #####################  ADDED CODE ELEMENTS #################################################################
        elements.extend(sorted_page_elements)
        
    return elements

RE_MULTISPACE_INCLUDING_NEWLINES = re.compile(pattern=r"\s+", flags=re.DOTALL)

@requires_dependencies("unstructured_inference")
def _partition_pdf_or_image_local(
    filename: str = "",
    file: Optional[bytes | IO[bytes]] = None,
    is_image: bool = False,
    infer_table_structure: bool = False,
    include_page_breaks: bool = False,
    languages: Optional[list[str]] = None,
    ocr_languages: Optional[str] = None,
    ocr_mode: str = OCRMode.FULL_PAGE.value,
    model_name: Optional[str] = None,  # to be deprecated in favor of `hi_res_model_name`
    hi_res_model_name: Optional[str] = None,
    pdf_image_dpi: Optional[int] = None,
    metadata_last_modified: Optional[str] = None,
    pdf_text_extractable: bool = False,
    extract_images_in_pdf: bool = False,
    extract_image_block_types: Optional[list[str]] = None,
    extract_image_block_output_dir: Optional[str] = None,
    extract_image_block_to_payload: bool = False,
    analysis: bool = False,
    analyzed_image_output_dir_path: Optional[str] = None,
    starting_page_number: int = 1,
    extract_forms: bool = False,
    form_extraction_skip_tables: bool = True,
    pdf_hi_res_max_pages: Optional[int] = None,
    **kwargs: Any,
) -> list[Element]:
    """Partition using package installed locally"""
    from unstructured_inference.inference.layout import (
        process_data_with_model,
        process_file_with_model,
    )

    from unstructured.partition.pdf_image.ocr import process_data_with_ocr, process_file_with_ocr
    from unstructured.partition.pdf_image.pdfminer_processing import (
        process_data_with_pdfminer,
        process_file_with_pdfminer,
    )

    if not is_image:
        check_pdf_hi_res_max_pages_exceeded(
            filename=filename, file=file, pdf_hi_res_max_pages=pdf_hi_res_max_pages
        )

    hi_res_model_name = hi_res_model_name or model_name or default_hi_res_model()
    if pdf_image_dpi is None:
        pdf_image_dpi = 200

    od_model_layout_dumper: Optional[ObjectDetectionLayoutDumper] = None
    extracted_layout_dumper: Optional[ExtractedLayoutDumper] = None
    ocr_layout_dumper: Optional[OCRLayoutDumper] = None
    final_layout_dumper: Optional[FinalLayoutDumper] = None

    skip_analysis_dump = env_config.ANALYSIS_DUMP_OD_SKIP

    if file is None:
        inferred_document_layout = process_file_with_model(
            filename,
            is_image=is_image,
            model_name=hi_res_model_name,
            pdf_image_dpi=pdf_image_dpi,
        )

        extracted_layout, layouts_links = (
            process_file_with_pdfminer(filename=filename, dpi=pdf_image_dpi)
            if pdf_text_extractable
            else ([], [])
        )

        if analysis:
            if not analyzed_image_output_dir_path:
                if env_config.GLOBAL_WORKING_DIR_ENABLED:
                    analyzed_image_output_dir_path = str(
                        Path(env_config.GLOBAL_WORKING_PROCESS_DIR) / "annotated"
                    )
                else:
                    analyzed_image_output_dir_path = str(Path.cwd() / "annotated")
            os.makedirs(analyzed_image_output_dir_path, exist_ok=True)
            if not skip_analysis_dump:
                od_model_layout_dumper = ObjectDetectionLayoutDumper(
                    layout=inferred_document_layout,
                    model_name=hi_res_model_name,
                )
                extracted_layout_dumper = ExtractedLayoutDumper(
                    layout=extracted_layout,
                )
                ocr_layout_dumper = OCRLayoutDumper()
        # NOTE(christine): merged_document_layout = extracted_layout + inferred_layout
        merged_document_layout = merge_inferred_with_extracted_layout(
            inferred_document_layout=inferred_document_layout,
            extracted_layout=extracted_layout,
            hi_res_model_name=hi_res_model_name,
        )

        final_document_layout = process_file_with_ocr(
            filename,
            merged_document_layout,
            extracted_layout=extracted_layout,
            is_image=is_image,
            infer_table_structure=infer_table_structure,
            ocr_languages=ocr_languages,
            ocr_mode=ocr_mode,
            pdf_image_dpi=pdf_image_dpi,
            ocr_layout_dumper=ocr_layout_dumper,
        )
    else:
        inferred_document_layout = process_data_with_model(
            file,
            is_image=is_image,
            model_name=hi_res_model_name,
            pdf_image_dpi=pdf_image_dpi,
        )

        if hasattr(file, "seek"):
            file.seek(0)

        extracted_layout, layouts_links = (
            process_data_with_pdfminer(file=file, dpi=pdf_image_dpi)
            if pdf_text_extractable
            else ([], [])
        )

        if analysis:
            if not analyzed_image_output_dir_path:
                if env_config.GLOBAL_WORKING_DIR_ENABLED:
                    analyzed_image_output_dir_path = str(
                        Path(env_config.GLOBAL_WORKING_PROCESS_DIR) / "annotated"
                    )
                else:
                    analyzed_image_output_dir_path = str(Path.cwd() / "annotated")
            if not skip_analysis_dump:
                od_model_layout_dumper = ObjectDetectionLayoutDumper(
                    layout=inferred_document_layout,
                    model_name=hi_res_model_name,
                )
                extracted_layout_dumper = ExtractedLayoutDumper(
                    layout=extracted_layout,
                )
                ocr_layout_dumper = OCRLayoutDumper()

        # NOTE(christine): merged_document_layout = extracted_layout + inferred_layout
        merged_document_layout = merge_inferred_with_extracted_layout(
            inferred_document_layout=inferred_document_layout,
            extracted_layout=extracted_layout,
            hi_res_model_name=hi_res_model_name,
        )

        if hasattr(file, "seek"):
            file.seek(0)
        final_document_layout = process_data_with_ocr(
            file,
            merged_document_layout,
            extracted_layout=extracted_layout,
            is_image=is_image,
            infer_table_structure=infer_table_structure,
            ocr_languages=ocr_languages,
            ocr_mode=ocr_mode,
            pdf_image_dpi=pdf_image_dpi,
            ocr_layout_dumper=ocr_layout_dumper,
        )

    final_document_layout = clean_pdfminer_inner_elements(final_document_layout)

    for page in final_document_layout.pages:
        for el in page.elements:
            el.text = el.text or ""

    elements = document_to_element_list(
        final_document_layout,
        sortable=True,
        include_page_breaks=include_page_breaks,
        last_modification_date=metadata_last_modified,
        # NOTE(crag): do not attempt to derive ListItem's from a layout-recognized "list"
        # block with NLP rules. Otherwise, the assumptions in
        # unstructured.partition.common::layout_list_to_list_items often result in weird chunking.
        infer_list_items=False,
        languages=languages,
        starting_page_number=starting_page_number,
        layouts_links=layouts_links,
        **kwargs,
    )

    extract_image_block_types = check_element_types_to_extract(extract_image_block_types)
    #  NOTE(christine): `extract_images_in_pdf` would deprecate
    #  (but continue to support for a while)
    if extract_images_in_pdf:
        # print("ONE TWO THREE")
        save_elements(
            elements=elements,
            starting_page_number=starting_page_number,
            element_category_to_save=ElementType.IMAGE,
            filename=filename,
            file=file,
            is_image=is_image,
            pdf_image_dpi=pdf_image_dpi,
            extract_image_block_to_payload=extract_image_block_to_payload,
            output_dir_path=extract_image_block_output_dir,
        )

    for el_type in extract_image_block_types:
        # print("ONE TWO")
        # print(el_type)
        if extract_images_in_pdf and el_type == ElementType.TABLE: ### NEED TO CHANGE THIS
            print("saving tables too") #### CHANGE!!!!!

        save_elements(
            elements=elements,
            starting_page_number=starting_page_number,
            element_category_to_save=el_type,
            filename=filename,
            file=file,
            is_image=is_image,
            pdf_image_dpi=pdf_image_dpi,
            extract_image_block_to_payload=extract_image_block_to_payload,
            output_dir_path=extract_image_block_output_dir,
        )

    out_elements = []
    for el in elements:
        if isinstance(el, PageBreak) and not include_page_breaks:
            continue

        if isinstance(el, Image):
            out_elements.append(cast(Element, el))
        # NOTE(crag): this is probably always a Text object, but check for the sake of typing
        elif isinstance(el, Text):
            el.text = re.sub(
                RE_MULTISPACE_INCLUDING_NEWLINES,
                " ",
                el.text or "",
            ).strip()
            if el.text or isinstance(el, PageBreak):
                out_elements.append(cast(Element, el))

    if extract_forms:
        forms = run_form_extraction(
            file=file,
            filename=filename,
            model_name=hi_res_model_name,
            elements=out_elements,
            skip_table_regions=form_extraction_skip_tables,
        )
        out_elements.extend(forms)

    if analysis:
        if not skip_analysis_dump:
            final_layout_dumper = FinalLayoutDumper(
                layout=out_elements,
            )
        layout_dumpers = []
        if od_model_layout_dumper:
            layout_dumpers.append(od_model_layout_dumper)
        if extracted_layout_dumper:
            layout_dumpers.append(extracted_layout_dumper)
        if ocr_layout_dumper:
            layout_dumpers.append(ocr_layout_dumper)
        if final_layout_dumper:
            layout_dumpers.append(final_layout_dumper)
        save_analysis_artifiacts(
            *layout_dumpers,
            filename=filename,
            file=file,
            is_image=is_image,
            analyzed_image_output_dir_path=analyzed_image_output_dir_path,
            skip_bboxes=env_config.ANALYSIS_BBOX_SKIP,
            skip_dump_od=env_config.ANALYSIS_DUMP_OD_SKIP,
            draw_grid=env_config.ANALYSIS_BBOX_DRAW_GRID,
            draw_caption=env_config.ANALYSIS_BBOX_DRAW_CAPTION,
            resize=env_config.ANALYSIS_BBOX_RESIZE,
            format=env_config.ANALYSIS_BBOX_FORMAT,
        )

    return out_elements


## below tweak to package
# Overwrite the function in the module
import sys
sys.modules['unstructured.partition.pdf'].partition_pdf = partition_pdf
sys.modules['unstructured.partition.pdf'].partition_pdf_or_image = partition_pdf_or_image
sys.modules['unstructured.partition.pdf']._partition_pdf_or_image_local = _partition_pdf_or_image_local


############################### MODIFICATION OF PARTITION_PDF WITH DIFFERENT PARAMETERS TO RUN HI RES EXTRACTION ######
############################### WHILE SAVING IMAGES IN PRE-DETERMINED FILE LOCATION ###################################

############################################# SETTING THE PATH TO THE POPPLER AND TESSERACT FOR OCR PROCESSING WHEN SELECTED ##
##############################
##############################
############################################# INCLUDING THE PATH TO THE CORRECT POPPLER ##############################

# Define the custom Poppler path
CUSTOM_POPPLER_PATH = os.path.join(os.getcwd(), "poppler-24.08.0\\Library\\bin")

# Store the original function
original_convert_from_path = pdf2image.convert_from_path

# Define the new function with the custom default poppler_path
def custom_convert_from_path(pdf_path, poppler_path=CUSTOM_POPPLER_PATH, *args, **kwargs):
    return original_convert_from_path(pdf_path, poppler_path=poppler_path, *args, **kwargs)

# Monkey patch the pdf2image.convert_from_path function
pdf2image.convert_from_path = custom_convert_from_path

############################################# INCLUDING THE PATH TO TESSERACT EXE  ###################################

CUSTOM_TESSERACT_PATH = os.path.join(os.getcwd(), "tesseract\\tesseract.exe")
## add tesseract to PATH for OCR processing
pytesseract.pytesseract.tesseract_cmd = CUSTOM_TESSERACT_PATH
unstructured_pytesseract.pytesseract.tesseract_cmd = CUSTOM_TESSERACT_PATH

#######################################################################################################################

from langchain_community.document_loaders import UnstructuredPDFLoader
