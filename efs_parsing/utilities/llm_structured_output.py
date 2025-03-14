##################################################################################################################
## STRUCTURED OUTPUT DEFINITION --> FOR PPTX SLIDES AND PDF PAGES 
##
##################################################################################################################
from pydantic import BaseModel

# Define the Pydantic model for the description of a slide that 
# has been converted to base64 encoded because it was too complex
class SlideInfo(BaseModel):
   overview: str
   artefact_type: str
   slide_title: str
   content_summary: str
   
class PictureProcessing(BaseModel):
   relevance: str
   relevance_explanation: str
   artefact_type: str
   content_summary: str

class TextProcessing(BaseModel):
   overview: str
   slide_title: str
   content: str