from __future__ import annotations
import os, sys

from docling_core.types.doc.labels import DocItemLabel

################################################## CHANGE TO DEFAULT EXPORT LABELS IN DOCLING CORE ####################


DEFAULT_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.FOOTNOTE, ## new addition
    DocItemLabel.CAPTION, ## new addition
    DocItemLabel.PAGE_FOOTER, ## new addition
    DocItemLabel.PAGE_HEADER, ## new addition
}

## adding the following 4 DocItemLabel to be processed as part of the conversion to have a more thorough
## extraction 
## please note that page footer are not taken into account because page footers "never" empirically
## provide data worth extracting for knowledge extraction 
sys.modules['docling_core.types.doc.document'].DEFAULT_EXPORT_LABELS = DEFAULT_EXPORT_LABELS ## new default labels


############################################## CHANGE TO THE PICTURE ITEM CLASS IN DOCLING CORE TO ACCOMODATE PICTURE
############################################## RELEVANCE ASSESSMENT #################################################

from docling_core.types.doc.document import FloatingItem, PictureDataType, DocumentOrigin, GroupItem, ImageRef 
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, TableItem, KeyValueItem
from docling_core.types.doc.document import PageItem, NodeItem, RefItem, ProvenanceItem, TableData, DocItem
import base64
import hashlib
import os
import sys
import typing
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from pydantic import (
    AnyUrl
)
from docling_core.types.doc.tokens import DocumentToken

"""Models for the Docling Document data type."""

import base64
import copy
import hashlib
import json
import mimetypes
import os
import re
import sys
import textwrap
import typing
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union
from urllib.parse import unquote

import pandas as pd
import yaml
from PIL import Image as PILImage
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)
from tabulate import tabulate
from typing_extensions import Annotated, Self

from docling_core.search.package import VERSION_PATTERN
from docling_core.types.doc import Size
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.labels import DocItemLabel, GroupLabel
from docling_core.types.doc.tokens import DocumentToken
from docling_core.types.doc.utils import relative_path

Uint64 = typing.Annotated[int, Field(ge=0, le=(2**64 - 1))]
LevelNumber = typing.Annotated[int, Field(ge=1, le=100)]
CURRENT_VERSION: Final = "1.0.0"

EXTRACTED_OUTPUT_ROOT = "extracted_output"


# class ImageRef(BaseModel):
#     """ImageRef."""

#     mimetype: str
#     dpi: int
#     size: Size
#     uri: Union[AnyUrl, Path] = Field(union_mode="left_to_right")
#     _pil: Optional[PILImage.Image] = None

#     @property
#     def pil_image(self) -> Optional[PILImage.Image]:
#         """Return the PIL Image."""
#         if self._pil is not None:
#             return self._pil

#         if isinstance(self.uri, AnyUrl):
#             if self.uri.scheme == "data":
#                 encoded_img = str(self.uri).split(",")[1]
#                 decoded_img = base64.b64decode(encoded_img)
#                 self._pil = PILImage.open(BytesIO(decoded_img))
#             elif self.uri.scheme == "file":
#                 self._pil = PILImage.open(unquote(str(self.uri.path)))
#             # else: Handle http request or other protocols...
#         elif isinstance(self.uri, Path):
#             self._pil = PILImage.open(self.uri)

#         return self._pil

#     @field_validator("mimetype")
#     @classmethod
#     def validate_mimetype(cls, v):
#         """validate_mimetype."""
#         # Check if the provided MIME type is valid using mimetypes module
#         if v not in mimetypes.types_map.values():
#             raise ValueError(f"'{v}' is not a valid MIME type")
#         return v

#     @classmethod
#     def from_pil(cls, image: PILImage.Image, dpi: int) -> Self:
#         """Construct ImageRef from a PIL Image."""
#         print("Creating the Image and ImageRef!!")
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         img_uri = f"data:image/png;base64,{img_str}"
#         return cls(
#             mimetype="image/png",
#             dpi=dpi,
#             size=Size(width=image.width, height=image.height),
#             uri=img_uri,
#             _pil=image,
#         )
        

# sys.modules['docling_core.types.doc.document'].ImageRef = ImageRef ## new Picture Item class that accomodates picture_relevance and better picture referencing



class PictureItem(FloatingItem):
    """PictureItem."""

    label: typing.Literal[DocItemLabel.PICTURE] = DocItemLabel.PICTURE
    picture_relevance: Optional[str] = None ## ADDITION TO CORE CODE FOR PICTURE PROCESSING #################### CODE CHANGE
    relevance_explanation: Optional[str] = None ## ADDITION TO CORE CODE FOR PICTURE PROCESSING #################### CODE CHANGE
    artefact_type: Optional[str] = None ## ADDITION TO CORE CODE FOR PICTURE PROCESSING #################### CODE CHANGE
    content_summary: Optional[str] = None ## ADDITION TO CORE CODE FOR PICTURE PROCESSING #################### CODE CHANGE
    annotations: List[PictureDataType] = []

    # Convert the image to Base64
    def _image_to_base64(self, pil_image, format="PNG"):
        """Base64 representation of the image."""
        buffered = BytesIO()
        pil_image.save(buffered, format=format)  # Save the image to the byte stream
        img_bytes = buffered.getvalue()  # Get the byte data
        img_base64 = base64.b64encode(img_bytes).decode(
            "utf-8"
        )  # Encode to Base64 and decode to string
        return img_base64

    def _image_to_hexhash(self) -> Optional[str]:
        """Hexash from the image."""
        if self.image is not None and self.image._pil is not None:
            # Convert the image to raw bytes
            image_bytes = self.image._pil.tobytes()

            # Create a hash object (e.g., SHA-256)
            hasher = hashlib.sha256()

            # Feed the image bytes into the hash object
            hasher.update(image_bytes)

            # Get the hexadecimal representation of the hash
            return hasher.hexdigest()

        return None
    ################################################################ CODE CHANGE TO EXPORT TO MARKDOWN #################
    def export_to_markdown(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        image_placeholder: str = "<!-- image -->",
    ) -> str:
        """Export picture to Markdown format."""
        default_response = "\n" + image_placeholder + "\n"
        error_response = (
            "\n<!-- 🖼️❌ Image not available. "
            "Please use `PdfPipelineOptions(generate_picture_images=True)`"
            " --> \n"
        )

        if image_mode == ImageRefMode.PLACEHOLDER:
            return default_response

        elif image_mode == ImageRefMode.EMBEDDED:

            # short-cut: we already have the image in base64
            if (
                isinstance(self.image, ImageRef)
                and isinstance(self.image.uri, AnyUrl)
                and self.image.uri.scheme == "data"
            ):
                text = f"\n![Image]({self.image.uri})\n"
                return text

            # get the self.image._pil or crop it out of the page-image
            img = self.get_image(doc)

            if img is not None:
                imgb64 = self._image_to_base64(img)
                text = f"\n![Image](data:image/png;base64,{imgb64})\n"

                return text
            else:
                return error_response

        elif image_mode == ImageRefMode.REFERENCED:
            if not isinstance(self.image, ImageRef) or (
                isinstance(self.image.uri, AnyUrl) and self.image.uri.scheme == "data"
            ):
                return default_response

            # text = f"\n![Image]({str(self.image.uri)})\n" change to code
            current_directory = Path(os.getcwd())
            file_path = self.image.uri
            ## Code change to make sure the actual proper reference is added
            text = f"\n![Image]({str(file_path.relative_to(current_directory)).replace(EXTRACTED_OUTPUT_ROOT, f"\{EXTRACTED_OUTPUT_ROOT}").replace("\\", "/")})\n" ## code change
            print("You are using the modified version of Picture Item")
            print(f"Improved reference: {text}")
            return text

        else:
            return default_response

    def export_to_html(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
    ) -> str:
        """Export picture to HTML format."""
        text = ""
        if add_caption and len(self.captions):
            text = self.caption_text(doc)

        caption_text = ""
        if len(text) > 0:
            caption_text = f"<figcaption>{text}</figcaption>"

        default_response = f"<figure>{caption_text}</figure>"

        if image_mode == ImageRefMode.PLACEHOLDER:
            return default_response

        elif image_mode == ImageRefMode.EMBEDDED:
            # short-cut: we already have the image in base64
            if (
                isinstance(self.image, ImageRef)
                and isinstance(self.image.uri, AnyUrl)
                and self.image.uri.scheme == "data"
            ):
                img_text = f'<img src="{self.image.uri}">'
                return f"<figure>{caption_text}{img_text}</figure>"

            # get the self.image._pil or crop it out of the page-image
            img = self.get_image(doc)

            if img is not None:
                imgb64 = self._image_to_base64(img)
                img_text = f'<img src="data:image/png;base64,{imgb64}">'

                return f"<figure>{caption_text}{img_text}</figure>"
            else:
                return default_response

        elif image_mode == ImageRefMode.REFERENCED:

            if not isinstance(self.image, ImageRef) or (
                isinstance(self.image.uri, AnyUrl) and self.image.uri.scheme == "data"
            ):
                return default_response

            img_text = f'<img src="{str(self.image.uri)}">'
            return f"<figure>{caption_text}{img_text}</figure>"

        else:
            return default_response

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "\n",
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,  # not used at the moment
        add_page_index: bool = True,
    ):
        r"""Export picture to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str:  (Default value = "\n")
        :param xsize: int:  (Default value = 100)
        :param ysize: int:  (Default value = 100)
        :param add_location: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param # not used at the momentadd_page_index: bool:  (Default value = True)

        """
        body = f"{DocumentToken.BEG_FIGURE.value}{new_line}"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_caption and len(self.captions):
            text = self.caption_text(doc)

            if len(text):
                body += f"{DocumentToken.BEG_CAPTION.value}"
                body += f"{text.strip()}"
                body += f"{DocumentToken.END_CAPTION.value}"
                body += f"{new_line}"

        body += f"{DocumentToken.END_FIGURE.value}{new_line}"

        return body


sys.modules['docling_core.types.doc.document'].PictureItem = PictureItem ## new Picture Item class that accomodates picture_relevance and better picture referencing


class DoclingDocument(BaseModel):
    """DoclingDocument."""

    _HTML_DEFAULT_HEAD: str = r"""<head>
    <link rel="icon" type="image/png"
    href="https://ds4sd.github.io/docling/assets/logo.png"/>
    <meta charset="UTF-8">
    <title>
    Powered by Docling
    </title>
    <style>
    html {
    background-color: LightGray;
    }
    body {
    margin: 0 auto;
    width:800px;
    padding: 30px;
    background-color: White;
    font-family: Arial, sans-serif;
    box-shadow: 10px 10px 10px grey;
    }
    figure{
    display: block;
    width: 100%;
    margin: 0px;
    margin-top: 10px;
    margin-bottom: 10px;
    }
    img {
    display: block;
    margin: auto;
    margin-top: 10px;
    margin-bottom: 10px;
    max-width: 640px;
    max-height: 640px;
    }
    table {
    min-width:500px;
    background-color: White;
    border-collapse: collapse;
    cell-padding: 5px;
    margin: auto;
    margin-top: 10px;
    margin-bottom: 10px;
    }
    th, td {
    border: 1px solid black;
    padding: 8px;
    }
    th {
    font-weight: bold;
    }
    table tr:nth-child(even) td{
    background-color: LightGray;
    }
    </style>
    </head>"""
    print("WARNING: Using modified version of Docling Document: WARNING")
    schema_name: typing.Literal["DoclingDocument"] = "DoclingDocument"
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        CURRENT_VERSION
    )
    name: str  # The working name of this document, without extensions
    # (could be taken from originating doc, or just "Untitled 1")
    origin: Optional[DocumentOrigin] = (
        None  # DoclingDocuments may specify an origin (converted to DoclingDocument).
        # This is optional, e.g. a DoclingDocument could also be entirely
        # generated from synthetic data.
    )

    furniture: GroupItem = GroupItem(
        name="_root_", self_ref="#/furniture"
    )  # List[RefItem] = []
    body: GroupItem = GroupItem(name="_root_", self_ref="#/body")  # List[RefItem] = []

    groups: List[GroupItem] = []
    texts: List[Union[SectionHeaderItem, ListItem, TextItem]] = []
    pictures: List[PictureItem] = []
    tables: List[TableItem] = []
    key_value_items: List[KeyValueItem] = []

    pages: Dict[int, PageItem] = {}  # empty as default

    def add_group(
        self,
        label: Optional[GroupLabel] = None,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
    ) -> GroupItem:
        """add_group.

        :param label: Optional[GroupLabel]:  (Default value = None)
        :param name: Optional[str]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if not parent:
            parent = self.body

        group_index = len(self.groups)
        cref = f"#/groups/{group_index}"

        group = GroupItem(self_ref=cref, parent=parent.get_ref())
        if name is not None:
            group.name = name
        if label is not None:
            group.label = label

        self.groups.append(group)
        parent.children.append(RefItem(cref=cref))

        return group

    def add_list_item(
        self,
        text: str,
        enumerated: bool = False,
        marker: Optional[str] = None,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_list_item.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        marker = marker or "-"

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            enumerated=enumerated,
            marker=marker,
        )
        if prov:
            list_item.prov.append(prov)

        self.texts.append(list_item)
        parent.children.append(RefItem(cref=cref))

        return list_item

    def add_text(
        self,
        label: DocItemLabel,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_text.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        # Catch a few cases that are in principle allowed
        # but that will create confusion down the road
        if label in [DocItemLabel.TITLE]:
            return self.add_title(text=text, orig=orig, prov=prov, parent=parent)

        elif label in [DocItemLabel.LIST_ITEM]:
            return self.add_list_item(text=text, orig=orig, prov=prov, parent=parent)

        elif label in [DocItemLabel.SECTION_HEADER]:
            return self.add_heading(text=text, orig=orig, prov=prov, parent=parent)

        else:

            if not parent:
                parent = self.body

            if not orig:
                orig = text

            text_index = len(self.texts)
            cref = f"#/texts/{text_index}"
            text_item = TextItem(
                label=label,
                text=text,
                orig=orig,
                self_ref=cref,
                parent=parent.get_ref(),
            )
            if prov:
                text_item.prov.append(prov)

            self.texts.append(text_item)
            parent.children.append(RefItem(cref=cref))

            return text_item

    def add_table(
        self,
        data: TableData,
        caption: Optional[Union[TextItem, RefItem]] = None,  # This is not cool yet.
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        label: DocItemLabel = DocItemLabel.TABLE,
    ):
        """add_table.

        :param data: TableData:
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param label: DocItemLabel:  (Default value = DocItemLabel.TABLE)

        """
        if not parent:
            parent = self.body

        table_index = len(self.tables)
        cref = f"#/tables/{table_index}"

        tbl_item = TableItem(
            label=label, data=data, self_ref=cref, parent=parent.get_ref()
        )
        if prov:
            tbl_item.prov.append(prov)
        if caption:
            tbl_item.captions.append(caption.get_ref())

        self.tables.append(tbl_item)
        parent.children.append(RefItem(cref=cref))

        return tbl_item

    def add_picture(
        self,
        annotations: List[PictureDataType] = [],
        image: Optional[ImageRef] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_picture.

        :param data: List[PictureData]: (Default value = [])
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        picture_index = len(self.pictures)
        cref = f"#/pictures/{picture_index}"

        fig_item = PictureItem(
            label=DocItemLabel.PICTURE,
            annotations=annotations,
            image=image,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            fig_item.prov.append(prov)
        if caption:
            fig_item.captions.append(caption.get_ref())

        self.pictures.append(fig_item)
        parent.children.append(RefItem(cref=cref))

        return fig_item

    def add_title(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_title.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        text_item = TextItem(
            label=DocItemLabel.TITLE,
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            text_item.prov.append(prov)

        self.texts.append(text_item)
        parent.children.append(RefItem(cref=cref))

        return text_item

    def add_heading(
        self,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_heading.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        section_header_item = SectionHeaderItem(
            level=level,
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            section_header_item.prov.append(prov)

        self.texts.append(section_header_item)
        parent.children.append(RefItem(cref=cref))

        return section_header_item

    def num_pages(self):
        """num_pages."""
        return len(self.pages.values())

    def validate_tree(self, root) -> bool:
        """validate_tree."""
        res = []
        for child_ref in root.children:
            child = child_ref.resolve(self)
            if child.parent.resolve(self) != root:
                return False
            res.append(self.validate_tree(child))

        return all(res) or len(res) == 0

    def iterate_items(
        self,
        root: Optional[NodeItem] = None,
        with_groups: bool = False,
        traverse_pictures: bool = False,
        page_no: Optional[int] = None,
        _level: int = 0,  # fixed parameter, carries through the node nesting level
    ) -> typing.Iterable[Tuple[NodeItem, int]]:  # tuple of node and level
        """iterate_elements.

        :param root: Optional[NodeItem]:  (Default value = None)
        :param with_groups: bool:  (Default value = False)
        :param traverse_pictures: bool:  (Default value = True)
        :param page_no: Optional[int]:  (Default value = None)
        :param _level:  (Default value = 0)
        :param # fixed parameter:
        :param carries through the node nesting level:
        """
        if not root:
            root = self.body

        # Yield non-group items or group items when with_groups=True
        if not isinstance(root, GroupItem) or with_groups:
            if isinstance(root, DocItem):
                if page_no is None or any(
                    prov.page_no == page_no for prov in root.prov
                ):
                    yield root, _level
            else:
                yield root, _level

        # Handle picture traversal - only traverse children if requested
        if isinstance(root, PictureItem) and not traverse_pictures:
            return

        # Traverse children
        for child_ref in root.children:
            child = child_ref.resolve(self)
            if isinstance(child, NodeItem):
                yield from self.iterate_items(
                    child,
                    with_groups=with_groups,
                    traverse_pictures=traverse_pictures,
                    page_no=page_no,
                    _level=_level + 1,
                )

    def _clear_picture_pil_cache(self):
        """Clear cache storage of all images."""
        for item, level in self.iterate_items(with_groups=False):
            if isinstance(item, PictureItem):
                if item.image is not None and item.image._pil is not None:
                    item.image._pil.close()

    def _list_images_on_disk(self) -> List[Path]:
        """List all images on disk."""
        result: List[Path] = []

        for item, level in self.iterate_items(with_groups=False):
            if isinstance(item, PictureItem):
                if item.image is not None:
                    if (
                        isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "file"
                        and item.image.uri.path is not None
                    ):
                        local_path = Path(unquote(item.image.uri.path))
                        result.append(local_path)
                    elif isinstance(item.image.uri, Path):
                        result.append(item.image.uri)

        return result

    def _with_embedded_pictures(self) -> "DoclingDocument":
        """Document with embedded images.

        Creates a copy of this document where all pictures referenced
        through a file URI are turned into base64 embedded form.
        """
        result: DoclingDocument = copy.deepcopy(self)

        for ix, (item, level) in enumerate(result.iterate_items(with_groups=True)):
            if isinstance(item, PictureItem):

                if item.image is not None:
                    if (
                        isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "file"
                    ):
                        assert isinstance(item.image.uri.path, str)
                        tmp_image = PILImage.open(str(unquote(item.image.uri.path)))
                        item.image = ImageRef.from_pil(tmp_image, dpi=item.image.dpi)

                    elif isinstance(item.image.uri, Path):
                        tmp_image = PILImage.open(str(item.image.uri))
                        item.image = ImageRef.from_pil(tmp_image, dpi=item.image.dpi)

        return result

    def _with_pictures_refs(
        self, image_dir: Path, reference_path: Optional[Path] = None
    ) -> "DoclingDocument":
        """Document with images as refs.

        Creates a copy of this document where all picture data is
        saved to image_dir and referenced through file URIs.
        """
        result: DoclingDocument = copy.deepcopy(self)

        img_count = 0
        image_dir.mkdir(parents=True, exist_ok=True)

        if image_dir.is_dir():
            for item, level in result.iterate_items(with_groups=False):
                if isinstance(item, PictureItem):

                    if (
                        item.image is not None
                        and isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "data"
                        and item.image.pil_image is not None
                    ):
                        img = item.image.pil_image

                        hexhash = item._image_to_hexhash()

                        # loc_path = image_dir / f"image_{img_count:06}.png"
                        if hexhash is not None:
                            loc_path = image_dir / f"image_{img_count:06}_{hexhash}.png"

                            img.save(loc_path)
                            if reference_path is not None:
                                obj_path = relative_path(
                                    reference_path.resolve(), loc_path.resolve()
                                )
                            else:
                                obj_path = loc_path

                            item.image.uri = Path(obj_path)

                        # if item.image._pil is not None:
                        #    item.image._pil.close()

                    img_count += 1

        return result

    def print_element_tree(self):
        """Print_element_tree."""
        for ix, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                print(" " * level, f"{ix}: {item.label.value} with name={item.name}")
            elif isinstance(item, DocItem):
                print(" " * level, f"{ix}: {item.label.value}")

    def export_to_element_tree(self) -> str:
        """Export_to_element_tree."""
        texts = []
        for ix, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                texts.append(
                    " " * level + f"{ix}: {item.label.value} with name={item.name}"
                )
            elif isinstance(item, DocItem):
                texts.append(" " * level + f"{ix}: {item.label.value}")

        return "\n".join(texts)

    def save_as_json(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        indent: int = 2,
    ):
        """Save as json."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
        )

        out = new_doc.export_to_dict()
        with open(filename, "w") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Path) -> "DoclingDocument":
        """load_from_json.

        :param filename: The filename to load a saved DoclingDocument from a .json.
        :type filename: Path

        :returns: The loaded DoclingDocument.
        :rtype: DoclingDocument

        """
        with open(filename, "r") as f:
            return cls.model_validate_json(f.read())

    def save_as_yaml(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        default_flow_style: bool = False,
    ):
        """Save as yaml."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
        )

        out = new_doc.export_to_dict()
        with open(filename, "w") as fw:
            yaml.dump(out, fw, default_flow_style=default_flow_style)

    def export_to_dict(
        self,
        mode: str = "json",
        by_alias: bool = True,
        exclude_none: bool = True,
    ) -> Dict:
        """Export to dict."""
        out = self.model_dump(mode=mode, by_alias=by_alias, exclude_none=exclude_none)

        return out

    def save_as_markdown(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        delim: str = "\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        strict_text: bool = False,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.REFERENCED,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
    ):
        """Save to markdown."""
        # artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        # if image_mode == ImageRefMode.REFERENCED:
        #     os.makedirs(artifacts_dir, exist_ok=True)

        # new_doc = self._make_copy_with_refmode(
        #     artifacts_dir, image_mode, reference_path=reference_path
        # )
        
        # print("AHAHAHAHA")
        md_out = self.export_to_markdown(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            strict_text=strict_text,
            image_placeholder=image_placeholder,
            image_mode=image_mode,
            indent=indent,
            text_width=text_width,
            page_no=page_no,
        )

        try:
            with open(filename, "w") as fw:
                fw.write(md_out)
        except:
            try:
                with open(filename, "w", encoding="utf-8") as fw:
                    fw.write(md_out)
            except Exception as e:
                print(f"Error creating the markdown output. Exception is {e}")
        
        return md_out ## code addition

    def export_to_markdown(  # noqa: C901
        self,
        delim: str = "\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        strict_text: bool = False,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
    ) -> str:
        r"""Serialize to Markdown.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole document.

        :param delim: Delimiter to use when concatenating the various
                Markdown parts. (Default value = "\n").
        :type delim: str = "\n"
        :param from_element: Body slicing start index (inclusive).
                (Default value = 0).
        :type from_element: int = 0
        :param to_element: Body slicing stop index
                (exclusive). (Default value = maxint).
        :type to_element: int = sys.maxsize
        :param labels: The set of document labels to include in the export.
        :type labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS
        :param strict_text: bool: Whether to only include the text content
            of the document. (Default value = False).
        :type strict_text: bool = False
        :param image_placeholder: The placeholder to include to position
            images in the markdown. (Default value = "\<!-- image --\>").
        :type image_placeholder: str = "<!-- image -->"
        :param image_mode: The mode to use for including images in the
            markdown. (Default value = ImageRefMode.PLACEHOLDER).
        :type image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
        :param indent: The indent in spaces of the nested lists.
            (Default value = 4).
        :type indent: int = 4
        :returns: The exported Markdown representation.
        :rtype: str
        """
        ## TODO: NEED TO UPDATE THE DOCSTRING HERE
        print("WARNING: using modified version of export_to_markdown from the modified DoclingDocument class: WARNING")
        mdtexts: list[str] = []
        list_nesting_level = 0  # Track the current list nesting level
        previous_level = 0  # Track the previous item's level
        in_list = False  # Track if we're currently processing list items
        first_page = self.pages[list(self.pages.keys())[0]].page_no ## code change
        traversed_page = first_page ## code change
        file_name = self.name ## code change
        last_item = len([x for x  in enumerate(self.iterate_items(self.body, with_groups=True))])
        
        for ix, (item, level) in enumerate(
            self.iterate_items(self.body, with_groups=True, page_no=page_no)
        ):
            # print(ix, last_item)
            ################################### CODE CHANGE TO ORIGINAL DOCLING ############################################################
            #
            #
            #
            ##################### LOGIC TO IMPLEMENT PAGE BREAKS IN DOCLING ###############################################################
            if not bool(mdtexts): # if empty mention it is the start of the first page
                mdtexts.append(f"<!-- Start of {file_name}-page-{traversed_page}.md -->" + "\n")
            
            if not isinstance(item, GroupItem):
                page_item = item.prov[0].page_no
                
                if traversed_page != page_item:
                    mdtexts.append(f"<!-- End of {file_name}-page-{traversed_page}.md -->" + "\n")
                    traversed_page = page_item
                    mdtexts.append(f"---" + "\n")
                    mdtexts.append(f"<!-- Start of {file_name}-page-{traversed_page}.md -->" + "\n")
            
            ###########################################################################################################
            
            
            # If we've moved to a lower level, we're exiting one or more groups
            if level < previous_level:
                # Calculate how many levels we've exited
                level_difference = previous_level - level
                # Decrement list_nesting_level for each list group we've exited
                list_nesting_level = max(0, list_nesting_level - level_difference)

            previous_level = level  # Update previous_level for next iteration

            if ix < from_element or to_element <= ix:
                continue  # skip as many items as you want

            # Handle newlines between different types of content
            if (
                len(mdtexts) > 0
                and not isinstance(item, (ListItem, GroupItem))
                and in_list
            ):
                mdtexts[-1] += "\n"
                in_list = False

            if isinstance(item, GroupItem) and item.label in [
                GroupLabel.LIST,
                GroupLabel.ORDERED_LIST,
            ]:

                if list_nesting_level == 0:  # Check if we're on the top level.
                    # In that case a new list starts directly after another list.
                    mdtexts.append("\n")  # Add a blank line

                # Increment list nesting level when entering a new list
                list_nesting_level += 1
                in_list = True
                continue

            elif isinstance(item, GroupItem):
                continue

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                in_list = False
                marker = "" if strict_text else "#"
                text = f"{marker} {item.text}"
                mdtexts.append(text.strip() + "\n")

            elif (
                isinstance(item, TextItem)
                and item.label in [DocItemLabel.SECTION_HEADER]
            ) or isinstance(item, SectionHeaderItem):
                in_list = False
                marker = ""
                if not strict_text:
                    marker = "#" * level
                    if len(marker) < 2:
                        marker = "##"
                text = f"{marker} {item.text}\n"
                mdtexts.append(text.strip() + "\n")

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:
                in_list = False
                text = f"```\n{item.text}\n```\n"
                mdtexts.append(text)

            elif isinstance(item, ListItem) and item.label in [DocItemLabel.LIST_ITEM]:
                in_list = True
                # Calculate indent based on list_nesting_level
                # -1 because level 1 needs no indent
                list_indent = " " * (indent * (list_nesting_level - 1))

                marker = ""
                if strict_text:
                    marker = ""
                elif item.enumerated:
                    marker = item.marker
                else:
                    marker = "-"  # Markdown needs only dash as item marker.

                text = f"{list_indent}{marker} {item.text}"
                mdtexts.append(text)
            
            ###################################################################### CODE CHANGE TO ORIGINAL DOCLING #######################
            #
            #
            #
            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CAPTION]:
                in_list = False
                if len(item.text) and text_width > 0:
                    wrapped_text = textwrap.fill(text, width=text_width)
                    mdtexts.append(wrapped_text + "\n")
                elif len(item.text):
                    text = f"{item.text}\n"
                    mdtexts.append(text)
            
            elif isinstance(item, TextItem) and item.label in [DocItemLabel.FOOTNOTE]: ## code addition
                in_list = False
                if len(item.text) and text_width > 0:
                    # wrapped_text = textwrap.fill(text, width=text_width)
                    wrapped_text = textwrap.fill(f"Footnote at page {item.prov[0].page_no} (source page): {item.text}\n", width=text_width)
                    mdtexts.append(wrapped_text + "\n")
                elif len(item.text):
                    text = f"Footnote at page {item.prov[0].page_no} (source page): {item.text}\n"
                    mdtexts.append(text)
            
            elif isinstance(item, TextItem) and (item.label in labels) and (item.label not in [DocItemLabel.CAPTION, 
                                                                                               DocItemLabel.FOOTNOTE, 
                                                                                               DocItemLabel.PAGE_FOOTER]): ## code addition
                in_list = False
                if len(item.text) and text_width > 0:
                    wrapped_text = textwrap.fill(text, width=text_width)
                    mdtexts.append(wrapped_text + "\n")
                elif len(item.text) > 1: # code modification from native docling to avoid single character parsing like page number
                    text = f"{item.text}\n"
                    mdtexts.append(text)
                    
            ###################################################################### CODE CHANGE TO ORIGINAL DOCLING #######################
            # elif isinstance(item, TextItem) and (item.label in labels):
            #     in_list = False
            #     if len(item.text) and text_width > 0:
            #         wrapped_text = textwrap.fill(text, width=text_width)
            #         mdtexts.append(wrapped_text + "\n")
            #     elif len(item.text):
            #         text = f"{item.text}\n"
            #         mdtexts.append(text)

            ## CODE ADDITION HERE TOO
            elif isinstance(item, TableItem) and not strict_text:
                in_list = False
                caption_text = item.caption_text(self)
                if mdtexts[-1] != (f"<!-- Start of Page {traversed_page} -->" + "\n"): ## code change
                    if bool(mdtexts):
                        if caption_text + "\n" != mdtexts[-1]: # do not add the captions that were individually identified before
                            mdtexts.append(caption_text)
                # mdtexts.append(item.caption_text(self))
                md_table = item.export_to_markdown()
                mdtexts.append("\n" + md_table + "\n")

            elif isinstance(item, PictureItem) and (not strict_text) and (item.picture_relevance == True): ## last argument is change to original package
                in_list = False
                caption_text = item.caption_text(self)
                if mdtexts[-1] != (f"<!-- Start of Page {traversed_page} -->" + "\n"): ## code change
                    if bool(mdtexts): # rule only applies only when it is not empty
                        if caption_text + "\n" != mdtexts[-1]: # do not add the captions that were individually identified before
                            mdtexts.append(caption_text)
                # mdtexts.append(item.caption_text(self))

                line = item.export_to_markdown(
                    doc=self,
                    image_placeholder=image_placeholder,
                    image_mode=image_mode,
                )

                mdtexts.append(line)
                mdtexts.append("\n" + f"Artefact Type: {item.artefact_type}" + "\n" ) ## code change to original package
                mdtexts.append("\n" + f"This is a summary of the artefact : {item.content_summary}" + "\n" ) ## code change to original package

            elif isinstance(item, DocItem) and item.label in labels and (item.label not in [DocItemLabel.PAGE_FOOTER]): ## change here
                # in_list = False
                # text = "<missing-text>"
                # mdtexts.append(text) ## code change
                pass
            
        if ix == (last_item - 1):
            mdtexts.append("\n" + f"<!-- End of {file_name}-page-{traversed_page}.md -->" + "\n")

        mdtext = (delim.join(mdtexts)).strip()
        mdtext = re.sub(
            r"\n\n\n+", "\n\n", mdtext
        )  # remove cases of double or more empty lines.

        # Our export markdown doesn't contain any emphasis styling:
        # Bold, Italic, or Bold-Italic
        # Hence, any underscore that we print into Markdown is coming from document text
        # That means we need to escape it, to properly reflect content in the markdown
        # However, we need to preserve underscores in image URLs
        # to maintain their validity
        # For example: ![image](path/to_image.png) should remain unchanged
        # def escape_underscores(text):
        #     """Escape underscores but leave them intact in the URL.."""
        #     # Firstly, identify all the URL patterns.
        #     url_pattern = r"!\[.*?\]\((.*?)\)"
        #     parts = []
        #     last_end = 0

        #     for match in re.finditer(url_pattern, text):
        #         # Text to add before the URL (needs to be escaped)
        #         before_url = text[last_end : match.start()]
        #         parts.append(re.sub(r"(?<!\\)_", r"\_", before_url))

        #         # Add the full URL part (do not escape)
        #         parts.append(match.group(0))
        #         last_end = match.end()

        #     # Add the final part of the text (which needs to be escaped)
        #     if last_end < len(text):
        #         parts.append(re.sub(r"(?<!\\)_", r"\_", text[last_end:]))

        #     return "".join(parts)

        # mdtext = escape_underscores(mdtext)

        return mdtext

    def export_to_text(  # noqa: C901
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = 1000000,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
    ) -> str:
        """export_to_text."""
        return self.export_to_markdown(
            delim,
            from_element,
            to_element,
            labels,
            strict_text=True,
            image_placeholder="",
        )

    def save_as_html(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = _HTML_DEFAULT_HEAD,
    ):
        """Save to HTML."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
        )

        html_out = new_doc.export_to_html(
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            image_mode=image_mode,
            page_no=page_no,
            html_lang=html_lang,
            html_head=html_head,
        )

        with open(filename, "w") as fw:
            fw.write(html_out)

    def _get_output_paths(
        self, filename: Path, artifacts_dir: Optional[Path] = None
    ) -> Tuple[Path, Optional[Path]]:
        if artifacts_dir is None:
            # Remove the extension and add '_pictures'
            artifacts_dir = filename.with_suffix("")
            artifacts_dir = artifacts_dir.with_name(artifacts_dir.name + "_artifacts")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = filename.parent
        return artifacts_dir, reference_path

    def _make_copy_with_refmode(
        self,
        artifacts_dir: Path,
        image_mode: ImageRefMode,
        reference_path: Optional[Path] = None,
    ):
        new_doc = None
        if image_mode == ImageRefMode.PLACEHOLDER:
            new_doc = self
        elif image_mode == ImageRefMode.REFERENCED:
            new_doc = self._with_pictures_refs(
                image_dir=artifacts_dir, reference_path=reference_path
            )
        elif image_mode == ImageRefMode.EMBEDDED:
            new_doc = self._with_embedded_pictures()
        else:
            raise ValueError("Unsupported ImageRefMode")
        return new_doc

    def export_to_html(  # noqa: C901
        self,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = _HTML_DEFAULT_HEAD,
    ) -> str:
        r"""Serialize to HTML."""

        def close_lists(
            curr_level: int,
            prev_level: int,
            in_ordered_list: List[bool],
            html_texts: list[str],
        ):

            if len(in_ordered_list) == 0:
                return (in_ordered_list, html_texts)

            while curr_level < prev_level and len(in_ordered_list) > 0:
                if in_ordered_list[-1]:
                    html_texts.append("</ol>")
                else:
                    html_texts.append("</ul>")

                prev_level -= 1
                in_ordered_list.pop()  # = in_ordered_list[:-1]

            return (in_ordered_list, html_texts)

        head_lines = ["<!DOCTYPE html>", f'<html lang="{html_lang}">', html_head]
        html_texts: list[str] = []

        prev_level = 0  # Track the previous item's level

        in_ordered_list: List[bool] = []  # False

        for ix, (item, curr_level) in enumerate(
            self.iterate_items(self.body, with_groups=True, page_no=page_no)
        ):
            # If we've moved to a lower level, we're exiting one or more groups
            if curr_level < prev_level and len(in_ordered_list) > 0:
                # Calculate how many levels we've exited
                # level_difference = previous_level - level
                # Decrement list_nesting_level for each list group we've exited
                # list_nesting_level = max(0, list_nesting_level - level_difference)

                in_ordered_list, html_texts = close_lists(
                    curr_level=curr_level,
                    prev_level=prev_level,
                    in_ordered_list=in_ordered_list,
                    html_texts=html_texts,
                )

            prev_level = curr_level  # Update previous_level for next iteration

            if ix < from_element or to_element <= ix:
                continue  # skip as many items as you want

            if (isinstance(item, DocItem)) and (item.label not in labels):
                continue  # skip any label that is not whitelisted

            if isinstance(item, GroupItem) and item.label in [
                GroupLabel.ORDERED_LIST,
            ]:

                text = "<ol>"
                html_texts.append(text.strip())

                # Increment list nesting level when entering a new list
                in_ordered_list.append(True)

            elif isinstance(item, GroupItem) and item.label in [
                GroupLabel.LIST,
            ]:

                text = "<ul>"
                html_texts.append(text.strip())

                # Increment list nesting level when entering a new list
                in_ordered_list.append(False)

            elif isinstance(item, GroupItem):
                continue

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:

                text = f"<h1>{item.text}</h1>"
                html_texts.append(text.strip())

            elif isinstance(item, SectionHeaderItem):

                section_level: int = item.level + 1

                text = f"<h{(section_level)}>{item.text}</h{(section_level)}>"
                html_texts.append(text.strip())

            elif isinstance(item, TextItem) and item.label in [
                DocItemLabel.SECTION_HEADER
            ]:

                section_level = curr_level

                if section_level <= 1:
                    section_level = 2

                if section_level >= 6:
                    section_level = 6

                text = f"<h{section_level}>{item.text}</h{section_level}>"
                html_texts.append(text.strip())

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:

                text = f"<pre>{item.text}</pre>"
                html_texts.append(text)

            elif isinstance(item, ListItem):

                text = f"<li>{item.text}</li>"
                html_texts.append(text)

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.LIST_ITEM]:

                text = f"<li>{item.text}</li>"
                html_texts.append(text)

            elif isinstance(item, TextItem) and item.label in labels:

                text = f"<p>{item.text}</p>"
                html_texts.append(text.strip())

            elif isinstance(item, TableItem):

                text = item.export_to_html(doc=self, add_caption=True)
                html_texts.append(text)

            elif isinstance(item, PictureItem):

                html_texts.append(
                    item.export_to_html(
                        doc=self, add_caption=True, image_mode=image_mode
                    )
                )

            elif isinstance(item, DocItem) and item.label in labels:
                continue

        html_texts.append("</html>")

        lines = []
        lines.extend(head_lines)
        for i, line in enumerate(html_texts):
            lines.append(line.replace("\n", "<br>"))

        delim = "\n"
        html_text = (delim.join(lines)).strip()

        return html_text

    def save_as_document_tokens(
        self,
        filename: Path,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_label: bool = True,
        add_table_cell_text: bool = True,
        # specifics
        page_no: Optional[int] = None,
        with_groups: bool = True,
    ):
        r"""Save the document content to a DocumentToken format."""
        out = self.export_to_document_tokens(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            xsize=xsize,
            ysize=ysize,
            add_location=add_location,
            add_content=add_content,
            add_page_index=add_page_index,
            # table specific flags
            add_table_cell_location=add_table_cell_location,
            add_table_cell_label=add_table_cell_label,
            add_table_cell_text=add_table_cell_text,
            # specifics
            page_no=page_no,
            with_groups=with_groups,
        )

        with open(filename, "w") as fw:
            fw.write(out)

    def export_to_document_tokens(
        self,
        delim: str = "\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_label: bool = True,
        add_table_cell_text: bool = True,
        # specifics
        page_no: Optional[int] = None,
        with_groups: bool = True,
        newline: bool = True,
    ) -> str:
        r"""Exports the document content to a DocumentToken format.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole main_text.

        :param delim: str:  (Default value = "\n\n")
        :param from_element: int:  (Default value = 0)
        :param to_element: Optional[int]:  (Default value = None)
        :param labels: set[DocItemLabel]
        :param xsize: int:  (Default value = 100)
        :param ysize: int:  (Default value = 100)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param add_page_index: bool:  (Default value = True)
        :param # table specific flagsadd_table_cell_location: bool
        :param add_table_cell_label: bool:  (Default value = True)
        :param add_table_cell_text: bool:  (Default value = True)
        :returns: The content of the document formatted as a DocTags string.
        :rtype: str
        """

        def close_lists(
            curr_level: int,
            prev_level: int,
            in_ordered_list: List[bool],
            result: str,
            delim: str,
        ):

            if len(in_ordered_list) == 0:
                return (in_ordered_list, result)

            while curr_level < prev_level and len(in_ordered_list) > 0:
                if in_ordered_list[-1]:
                    result += f"</ordered_list>{delim}"
                else:
                    result += f"</unordered_list>{delim}"

                prev_level -= 1
                in_ordered_list.pop()  # = in_ordered_list[:-1]

            return (in_ordered_list, result)

        if newline:
            delim = "\n"
        else:
            delim = ""

        prev_level = 0  # Track the previous item's level

        in_ordered_list: List[bool] = []  # False

        result = f"{DocumentToken.BEG_DOCUMENT.value}{delim}"

        for ix, (item, curr_level) in enumerate(
            self.iterate_items(self.body, with_groups=True)
        ):

            # If we've moved to a lower level, we're exiting one or more groups
            if curr_level < prev_level and len(in_ordered_list) > 0:
                # Calculate how many levels we've exited
                # level_difference = previous_level - level
                # Decrement list_nesting_level for each list group we've exited
                # list_nesting_level = max(0, list_nesting_level - level_difference)

                in_ordered_list, result = close_lists(
                    curr_level=curr_level,
                    prev_level=prev_level,
                    in_ordered_list=in_ordered_list,
                    result=result,
                    delim=delim,
                )

            prev_level = curr_level  # Update previous_level for next iteration

            if ix < from_element or to_element <= ix:
                continue  # skip as many items as you want

            if (isinstance(item, DocItem)) and (item.label not in labels):
                continue  # skip any label that is not whitelisted

            if isinstance(item, GroupItem) and item.label in [
                GroupLabel.ORDERED_LIST,
            ]:

                result += f"<ordered_list>{delim}"
                in_ordered_list.append(True)

            elif isinstance(item, GroupItem) and item.label in [
                GroupLabel.LIST,
            ]:

                result += f"<unordered_list>{delim}"
                in_ordered_list.append(False)

            elif isinstance(item, SectionHeaderItem):

                result += item.export_to_document_tokens(
                    doc=self,
                    new_line=delim,
                    xsize=xsize,
                    ysize=ysize,
                    add_location=add_location,
                    add_content=add_content,
                    add_page_index=add_page_index,
                )

            elif isinstance(item, TextItem) and (item.label in labels):

                result += item.export_to_document_tokens(
                    doc=self,
                    new_line=delim,
                    xsize=xsize,
                    ysize=ysize,
                    add_location=add_location,
                    add_content=add_content,
                    add_page_index=add_page_index,
                )

            elif isinstance(item, TableItem) and (item.label in labels):

                result += item.export_to_document_tokens(
                    doc=self,
                    new_line=delim,
                    xsize=xsize,
                    ysize=ysize,
                    add_caption=True,
                    add_location=add_location,
                    add_content=add_content,
                    add_cell_location=add_table_cell_location,
                    add_cell_label=add_table_cell_label,
                    add_cell_text=add_table_cell_text,
                    add_page_index=add_page_index,
                )

            elif isinstance(item, PictureItem) and (item.label in labels):

                result += item.export_to_document_tokens(
                    doc=self,
                    new_line=delim,
                    xsize=xsize,
                    ysize=ysize,
                    add_caption=True,
                    add_location=add_location,
                    add_content=add_content,
                    add_page_index=add_page_index,
                )

        result += DocumentToken.END_DOCUMENT.value

        return result
    
    def classify_pictures_relevance(self,
                                    picture_set_analysis):
        """
        TODO: NEED TO DESCRIBE THIS METHOD NEW METHOD NOT NATIVE TO DOCLING
        """
        for set_idx, analysis_set in picture_set_analysis.items():
            pictures_set = analysis_set[1]
            
            if analysis_set[0]["relevance"] in ["FALSE", 'False']: ## FALSE or False because the picture relevance classification are determined by multimodal LLM that can be unstable even with structured output 
                print(f"Picture set {set_idx} is deemed IRRELEVANT")
                for picture_key in pictures_set:
                    print(f"Picture with key {picture_key}: path: {self.pictures[picture_key].image.uri} \
                        has been marked as irrelevant")
                    self.pictures[picture_key].picture_relevance = False
            if analysis_set[0]["relevance"] in ["TRUE", 'True']:
                print(f"Picture set {set_idx} is deemed RELEVANT")
                for picture_key in pictures_set:
                    print(f"Picture with key {picture_key}: path: {self.pictures[picture_key].image.uri} \
                        has been marked as relevant")
                    self.pictures[picture_key].picture_relevance = True
                    self.pictures[picture_key].relevance_explanation = analysis_set[0]["relevance_explanation"] 
                    self.pictures[picture_key].artefact_type = analysis_set[0]["artefact_type"] 
                    self.pictures[picture_key].content_summary = analysis_set[0]["content_summary"] 
        
        print("Picture relevance classification is complete")

    def _export_to_indented_text(
        self, indent="  ", max_text_len: int = -1, explicit_tables: bool = False
    ):
        """Export the document to indented text to expose hierarchy."""
        result = []

        def get_text(text: str, max_text_len: int):

            middle = " ... "

            if max_text_len == -1:
                return text
            elif len(text) < max_text_len + len(middle):
                return text
            else:
                tbeg = int((max_text_len - len(middle)) / 2)
                tend = int(max_text_len - tbeg)

                return text[0:tbeg] + middle + text[-tend:]

        for i, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                result.append(
                    indent * level
                    + f"item-{i} at level {level}: {item.label}: group {item.name}"
                )

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, SectionHeaderItem):
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, ListItem) and item.label in [DocItemLabel.LIST_ITEM]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, TextItem):
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, TableItem):

                result.append(
                    indent * level
                    + f"item-{i} at level {level}: {item.label} with "
                    + f"[{item.data.num_rows}x{item.data.num_cols}]"
                )

                for _ in item.captions:
                    caption = _.resolve(self)
                    result.append(
                        indent * (level + 1)
                        + f"item-{i} at level {level + 1}: {caption.label}: "
                        + f"{caption.text}"
                    )

                if explicit_tables:
                    grid: list[list[str]] = []
                    for i, row in enumerate(item.data.grid):
                        grid.append([])
                        for j, cell in enumerate(row):
                            if j < 10:
                                text = get_text(text=cell.text, max_text_len=16)
                                grid[-1].append(text)

                    result.append("\n" + tabulate(grid) + "\n")

            elif isinstance(item, PictureItem):

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}"
                )

                for _ in item.captions:
                    caption = _.resolve(self)
                    result.append(
                        indent * (level + 1)
                        + f"item-{i} at level {level + 1}: {caption.label}: "
                        + f"{caption.text}"
                    )

            elif isinstance(item, DocItem):
                result.append(
                    indent * (level + 1)
                    + f"item-{i} at level {level}: {item.label}: ignored"
                )

        return "\n".join(result)

    def add_page(
        self, page_no: int, size: Size, image: Optional[ImageRef] = None
    ) -> PageItem:
        """add_page.

        :param page_no: int:
        :param size: Size:

        """
        pitem = PageItem(page_no=page_no, size=size, image=image)

        self.pages[page_no] = pitem
        return pitem

    @field_validator("version")
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this document version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, CURRENT_VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(
                f"incompatible version {v} with schema version {CURRENT_VERSION}"
            )
        else:
            return CURRENT_VERSION

    @model_validator(mode="after")  # type: ignore
    @classmethod
    def validate_document(cls, d: "DoclingDocument"):
        """validate_document."""
        if not d.validate_tree(d.body) or not d.validate_tree(d.furniture):
            raise ValueError("Document hierachy is inconsistent.")

        return d

# sys.modules['docling_core.types.doc.document'].DoclingDocument = DoclingDocument ## new Docling Document object with new export_to_markdown method


################################################## ADDING STANDARD PDF PIPELINE ######################################

import logging
import sys
from pathlib import Path
from typing import Optional

from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import AssembledUnit, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.ds_glm_model import GlmModel, GlmOptions
from docling.models.easyocr_model import EasyOcrModel
from docling.models.layout_model import LayoutModel
from docling.models.ocr_mac_model import OcrMacModel
from docling.models.page_assemble_model import PageAssembleModel, PageAssembleOptions
from docling.models.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.rapid_ocr_model import RapidOcrModel
from docling.models.table_structure_model import TableStructureModel
from docling.models.tesseract_ocr_cli_model import TesseractOcrCliModel
from docling.models.tesseract_ocr_model import TesseractOcrModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

# _log = logging.getLogger(__name__)


class StandardPdfPipeline(PaginatedPipeline):
    _layout_model_path = "model_artifacts/layout"
    _table_model_path = "model_artifacts/tableformer"

    def __init__(self, pipeline_options: PdfPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: PdfPipelineOptions

        if pipeline_options.artifacts_path is None:
            self.artifacts_path = self.download_models_hf()
        else:
            self.artifacts_path = Path(pipeline_options.artifacts_path)

        keep_images = (
            self.pipeline_options.generate_page_images
            or self.pipeline_options.generate_picture_images
            or self.pipeline_options.generate_table_images
        )

        self.glm_model = GlmModel(options=GlmOptions())

        if (ocr_model := self.get_ocr_model()) is None:
            raise RuntimeError(
                f"The specified OCR kind is not supported: {pipeline_options.ocr_options.kind}."
            )

        self.build_pipe = [
            # Pre-processing
            PagePreprocessingModel(
                options=PagePreprocessingOptions(
                    images_scale=pipeline_options.images_scale
                )
            ),
            # OCR
            ocr_model,
            # Layout model
            LayoutModel(
                artifacts_path=self.artifacts_path
                / StandardPdfPipeline._layout_model_path,
                accelerator_options=pipeline_options.accelerator_options,
            ),
            # Table structure model
            TableStructureModel(
                enabled=pipeline_options.do_table_structure,
                artifacts_path=self.artifacts_path
                / StandardPdfPipeline._table_model_path,
                options=pipeline_options.table_structure_options,
                accelerator_options=pipeline_options.accelerator_options,
            ),
            # Page assemble
            PageAssembleModel(options=PageAssembleOptions(keep_images=keep_images)),
        ]

        self.enrichment_pipe = [
            # Other models working on `NodeItem` elements in the DoclingDocument
        ]

    @staticmethod
    def download_models_hf(
        local_dir: Optional[Path] = None, force: bool = False
    ) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        download_path = snapshot_download(
            repo_id="ds4sd/docling-models",
            force_download=force,
            local_dir=local_dir,
            revision="v2.1.0",
        )

        return Path(download_path)

    def get_ocr_model(self) -> Optional[BaseOcrModel]:
        if isinstance(self.pipeline_options.ocr_options, EasyOcrOptions):
            return EasyOcrModel(
                enabled=self.pipeline_options.do_ocr,
                options=self.pipeline_options.ocr_options,
                accelerator_options=self.pipeline_options.accelerator_options,
            )
        elif isinstance(self.pipeline_options.ocr_options, TesseractCliOcrOptions):
            return TesseractOcrCliModel(
                enabled=self.pipeline_options.do_ocr,
                options=self.pipeline_options.ocr_options,
            )
        elif isinstance(self.pipeline_options.ocr_options, TesseractOcrOptions):
            return TesseractOcrModel(
                enabled=self.pipeline_options.do_ocr,
                options=self.pipeline_options.ocr_options,
            )
        elif isinstance(self.pipeline_options.ocr_options, RapidOcrOptions):
            return RapidOcrModel(
                enabled=self.pipeline_options.do_ocr,
                options=self.pipeline_options.ocr_options,
                accelerator_options=self.pipeline_options.accelerator_options,
            )
        elif isinstance(self.pipeline_options.ocr_options, OcrMacOptions):
            if "darwin" != sys.platform:
                raise RuntimeError(
                    f"The specified OCR type is only supported on Mac: {self.pipeline_options.ocr_options.kind}."
                )
            return OcrMacModel(
                enabled=self.pipeline_options.do_ocr,
                options=self.pipeline_options.ocr_options,
            )
        return None

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            page._backend = conv_res.input._backend.load_page(page.page_no)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()

        return page

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        all_elements = []
        all_headers = []
        all_body = []

        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            for p in conv_res.pages:
                if p.assembled is not None:
                    for el in p.assembled.body:
                        all_body.append(el)
                    for el in p.assembled.headers:
                        all_headers.append(el)
                    for el in p.assembled.elements:
                        all_elements.append(el)

            conv_res.assembled = AssembledUnit(
                elements=all_elements, headers=all_headers, body=all_body
            )

            conv_res.document = self.glm_model(conv_res)

            # Generate page images in the output
            if self.pipeline_options.generate_page_images:
                for page in conv_res.pages:
                    assert page.image is not None
                    page_no = page.page_no + 1
                    conv_res.document.pages[page_no].image = ImageRef.from_pil(
                        page.image, dpi=int(72 * self.pipeline_options.images_scale)
                    )

            # Generate images of the requested element types
            if (
                self.pipeline_options.generate_picture_images
                or self.pipeline_options.generate_table_images
            ):
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, DocItem) or len(element.prov) == 0:
                        continue
                    if (
                        element.label == DocItemLabel.PICTURE
                        and self.pipeline_options.generate_picture_images
                    ) or (
                        element.label == DocItemLabel.TABLE
                        and self.pipeline_options.generate_table_images
                    ):
                        page_ix = element.prov[0].page_no - 1
                        page = conv_res.pages[page_ix]
                        assert page.size is not None
                        assert page.image is not None

                        crop_bbox = (
                            element.prov[0]
                            .bbox.scaled(scale=scale)
                            .to_top_left_origin(page_height=page.size.height * scale)
                        )

                        cropped_im = page.image.crop(crop_bbox.as_tuple())
                        element.image = ImageRef.from_pil(
                            cropped_im, dpi=int(72 * scale)
                        )
                        # print("test")

        return conv_res

    @classmethod
    def get_default_options(cls) -> PdfPipelineOptions:
        return PdfPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)


sys.modules['docling.pipeline.standard_pdf_pipeline'].StandardPdfPipeline = StandardPdfPipeline





