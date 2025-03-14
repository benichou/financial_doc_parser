<!-- CREDITS: Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<div align="center">
    <img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 30%;"
    src="https://github.com/benichou/financial_doc_parser/blob/main/efs_parsing/assets/logo/financial_documents_logo.png"
    alt="Our logo">
    </img>

  <h3 align="center">Financial Services Information Parsing</h3>

  <p align="center">
    Repository Documentation for the Financial Information Parsing
    
</div>

## Table of Contents

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Click to Expand</summary>
  <ol>
    <li>
      <a href="#about-the-repo">About The Repo</a>
      <ul>
        <li><a href="#repository-structure">Repository Structure</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Repo

This repo aims to build the base capabilities to extract information from financial documents, with a particular concern
for financial information and mathematical equations.

The repo limits ingesting to pptx and pdf files, parses these files and converts the parsed output into markdown and word documents.

This limited scope of files to extract information from (that is PPTX and PDF) has been agreed upon with financial business team.

Please note that the pptx parsing is conducted by using the `unstructured` open source library while the pdf parsing is done 
with `docling` from IBM.

Initially, the intent was to build the pdf parsing with `unstructured` but this changed given the greatly improved parsing peformance provided
by `docling`. For future considerations, we could parse pptx files by using docling to have a more unified pipeline. For now, it has not been conducted to 
limit the amount of rework on the pptx parsing (given a number of heuristics are in place to handle complex slides and complex math equations).

- [Unstructured Home Page Link on Github](https://github.com/Unstructured-IO)
- [Docling Home Page Link on Github](https://github.com/DS4SD/docling)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure

This repository contains the following structure to help you navigate and understand the project:

The repo tree to update since I have made it a package for code reusability
```plaintext
📂 Financial_Doc_Parser/
├── 📁 analysis/                              # analysis of the file pages and pictures to augment parsing
│   ├── 🐍 page_analysis.py                   # series of functions that analyze pages of a file to determine base64 encoding
│   └── 🐍 picture_analysis.py                # series of functions that analyze whether pictures should be kept in the parsing output
├── 📁 assets/                                # asset files - for documentation only 
│   └── logo/                                 # logo/static files (e.g., images, fonts)
│       └── 🖼️ efs_logo.png                   # logo for readme.md documentation
├── 📁 conversion/                            # analysis of the file pages and pictures to augment parsing
│   ├── 🐍 efs_file_conversion.py             # conversion of parsed output to markdown (applies to pptx only) and word document | Supports Human in the Loop       
├── 📁 custom_loader/                         # custom loaders to support specific parsing requirements of financial files for both pptx and pdf
│   ├── 📁 docling/                           # modified docling parser (modified to support the specific parsing needs for Financial Documents)  
|   |   └── 🐍 pdf_custom_loader.py           # Docling parser to support more specialized needs beyond the native capabilities provided by native docling            
│   └── 📁 unstructured/                      # modified unstructured parser (modified to suppor the specific parsing needs for Financial Documents)
|       ├── 🐍 pdf_custom_loader.py           # deprecated/disregard
|       └── 🐍 powerpoint_custom_loader.py    # Unstrutured parser to support more specialized needs beyond the native capabilities provided by native unstructured
├── 📁 extracted_output/                      # local destination of saving the different parsed individual artefacts for individual files 
│   |── 📁 pdf/                               # individual pdf files parsed artefacts (images, page images, table output)
|   |   └── 📁 pdf_file_1/                    # single individual parsed file
|   |       ├── 📁 compressed_pdf_images/     # stores the pictures deemed relevant for the business and compressed
|   |       ├── 📁 image-references_artifacts/ # stores the original pictures from document after first pass  
|   |       ├── 📁 images_output/             # stores the original pictures from document after second pass with modified parser 
|   |       ├── 📁 pages_img_output/          # stores the original pages of the doc and converted to an image
|   |       └── 📁 tables_output/             # stores the parsed tables in html, csv, and html format
│   └── 📁 pptx/                              # modified unstructured parser (modified to suppor the specific parsing needs for Financial Documents) 
|       └── 📁 pptx_file_1/                   # single individual parsed file
|           ├── 📁 images_output/             # stores the pictures deemed relevant for the business and compressed
|           └── 📁 pages_img_output/          # stores the original pictures from document including the compressed ones (used for base64 encoding and use by LLM for parsing)
├── 📁 extraction/                            # where the functionalities for parsing pdf and pptx are housed
│   |── 📁 pdf/                               # parsing functionalities for pdf files
|   |   └── 🐍 pdf_extraction.py              # includes docling pipeline setup, saving of the pdf document artefacts, custom parser instantiation, complex pdf pages extraction
│   |── 📁 pptx/                              # parsing functionalities for pptx files
|   |   └── 🐍 slide_extraction.py            # includes the extract slide content and the extract speaker note functionalities
|   └── 🐍 content_extraction.py              # includes the creation of the appropriate folders for file artefacts, and orchestrates all functionalities to parse pdf/pptx file 
├── 📁 output_summary/                        # Automation and utility scripts
|   |── 📁 docx/                              # stores the docx files stemming from parsing output
|   |   └── 📁 final/                         # stores the individual docx output (stemming from the parsing output)
|   └── 📁 markdown/                          # stores the markdown files stemming from parsing output
|       ├── 📁 final/                         # where the final markdown files are stored
|       └── 📁 pptx_pages_converted_to_markdown/ # where the separate final markdown pages are stored 
├── 📁 poppler-24.08.0/                       # poppler artefact aimed to initially support tesseract for pptx parsing with unstructued - not needed anymore - deprecated 
├── 📁 prompt                                 # prompt folder to manage prompts and llm personas
|    ├── 🐍 model_persona_repo.py             # manage the llm personas
|    ├── 🐍 prompt_mgmt.py                    # handles manipulation of prompts to apply the correct one in the appropriate context
|    └── 🐍 prompt_repo.py                    # stores all prompts for the Financial Documents solution
├── 📁 tesseract/                             # aimed to initially support tesseract for pptx parsing with unstructured - not needed anymore - deprecated
├── 📁 utilities/                             # stores all necessary utilities functions (functions used often across the modules without being task/domain specific)
|   ├── 🐍 llm_structured_output.py           # stores the structured output classes
|   ├── 🐍 llm_utils.py                       # stores the llm api calls (only Azure Open AI at the time of writing)
|   ├── 🐍 picture_utils.py                   # includes the picture utilities functions
|   └── 🐍 utils.py                           # includes some python utilities used for data wrangling and file manipulation
├── 📄 .gitignore                             # Git ignore rules
├── 🐍 main.py                                # THE POINT OF ENTRY OF THIS SOLUTION: MAIN.PY TO EXECUTE AT ROOT | PARALLEL OR SEQUENTIAL PROCESSING OPTIONS
├── 📄 README.md                              # Repository overview and guide
├── 📄 requirements.txt                       # the list of python packages to create appropriate python virtual environment
└── 🐍 settings.py                            # solution parameters (pdf/pptx parsing parameters,base64 encoding triaging parameters, image compression parameters, aoi cred,...)

```

Note: Please note that you will need to create your own .env file to store the azure open ai credentials, at the root of the repo. 
Details are provided under the Installation section

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

- [![Python][Python]][Python]
- [![Langchain][Langchain]][Langchain]
- Docling & Unstructured
- [![Openai][Openai]][Openai]
- [![Azure][Azure]][Azure]
- [![Git][Git]][Git]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Prerequisites

Make sure you have the following:
- `git`
- `Python`
- Credentials to run the api calls to the LLM
- Donwload the `ds4sd_docling-models` (version: `v2.1.0`) and save it to a know path (update the `DOCLING_TRANSFORMER_MODEL_PATH`
variable in `settings.py` accordingly). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

1. Make sure to have a correct Open AI API key and add below to the `.env` file.
```
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_TYPE = ""
AZURE_OPENAI_API_VERSION = ""
AZURE_OPENAI_MODEL_NAME = ""
AZURE_OPEN_AI_MODEL = ""
CONTEXT_WINDOW_LIMIT = (expects an integet, not a string)
```

2. Clone the repo

```sh
git clone https://fbenichou@dev.azure.com/fbenichou/fbenichou/_git/efs_parsing
```

3. Python version 

Make sure the python version is 3.12 (3.12.7 if possible)

4. Create a new venv environment

```sh
python -m venv efs_environment
```

5. Activate conda environment

```sh
efs_environment\Scripts\activate
```

6. Install the necessary dependencies by running the following command (make sure you are at the root of the repo)

```sh
pip install -r requirements.txt
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

1. To run locally (after the prerequisites are covered): 
  ```sh
  cd efs_poc-1
  efs_environment\Scripts\activate
  python efs_parsing/main.py
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## High Level Logic

![High Level Logic](/efs_parsing/assets/diagrams/high_level_logic.png)


## Roadmap

- [] Migrate from unstructured to docling for pptx parsing
- [] Enable all other supported docling format types
- [] Make this solution a base capability

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Franck Benichou - franck.benichou@sciencespo.fr

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&logoColor=blue
[Langchain]: https://img.shields.io/badge/langchain-000000?style=for-the-badge&logo=langchain&logoColor=blue
[Openai]: https://img.shields.io/badge/openai-000001?style=for-the-badge&logo=openai&logoColor=orange
[Azure]: https://img.shields.io/badge/azure-000001?style=for-the-badge&logo=azuredevops&logoColor=blue
[Git]: https://img.shields.io/badge/git-000001?style=for-the-badge&logo=git&logoColor=white