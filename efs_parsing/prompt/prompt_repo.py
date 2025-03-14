

COMPLEX_SLIDE_EXTRACTION_PROMPT = """First, identify the type of information that is displayed. \
                                    Reflect on whether this is a series of paragraph or is a tabular form. \
                                    Based on the structure of the information, describe the image \
                                    thoroughly. Identify if the information is laid out in a table or several tables. \
                                    IF the data displayed resembles tabular data OR could be re-organized in a table, restructure the data into a table \
                                    Then, after your own thorough analysis of the image and details, \
                                    EXTRACT EXACTLY ALL CONTENT IN THE SAME WAY DISPLAYED PREVIOUSLY \
                                    leading to a greater understanding. \
                                    Regenerate the same content in plain markdown format (no latex) in the same structure as the input (i.e, \
                                    IF the input displays the data in a tabular form, regenerate in a tabular format) \
                                    Provide that description in the 'content_summary' key in plain markdown format (no need for complex latex). \
                                  """
                                  
def process_math_extraction_prompt(additional_shape_stats):
    """
    """
    print(additional_shape_stats)
    MATH_EXTRACTION_PROMPT = f"""
                            You are given a very complex picture with mathematical notations. So, analyze the image step by \
                            step and identify all existing mathematical operators such as multiplicative, additive, \
                            substractive, divisor and equality signs. Keep the stored information temporarily. \
                                
                            Please note your colleague who took a look at the image confirmed with 100% confidence there are a minimum of:
                            - Minimum number of (+) Plus signs: {additional_shape_stats["count_pluses"]}
                            - Minimum number of (-) Minus signs: {additional_shape_stats["count_minuses"]}
                            - Minimum number of (x) Multiplicative signs: {additional_shape_stats["count_multiplicator"]}
                            - Minimum number of (/) Divisor signs: {additional_shape_stats["count_divisor"]}
                            - Minimum number of (=) Equal signs:{additional_shape_stats["count_equals"]}
                            - Minimum number of (≠) Not equal signs: {additional_shape_stats["count_not_equals"]}
                            
                                
                            After you analyzed this image, count the number of Plus signs, Minus Signs, \
                            Multiplicative signs, Divisor signs, Equal signs, Not equal signs you see. 
                                
                            MAKE SURE you identified and re-generated the same number - or more - of Plus signs, Minus Signs, \
                            Multiplicative signs, Divisor signs, Equal signs, Not equal signs. \
                            Note your colleague probably missed some other mathematical operators so it is expected you \
                            could identify more mathematical operators than your colleague - just make sure to consider \
                            his findings as a baseline. \
                            
                            Then, try to identify \
                            in details all present equations as close as possible \
                            to what was presented to you. \
                                
                            Now, based on that count of mathematical operators and analysis of the image, \
                            extract and structure key information from the provided presentation slide image. 
                            The input consists of a base64-encoded image representing a complete presentation slide. 
                            Extract ALL mathematical equations and expressions including the mathematical operators \
                            identified earlier,  \
                            and format them using LaTeX notation. Do not add any new operator or invent an equation. \
                            Of the highest importance, make sure to extract minus, plus, multiplication, division and \
                            equality signs.
                            The extracted content should preserve all textual, visual, and mathematical elements in a \
                            format suitable for future question-answering tasks about business \
                            processes, calculations, and policies. Preserve tabular information by recreating the table.
                            
                            Provide the output in the 'content_summary' key in markdown format.
                            """
    return MATH_EXTRACTION_PROMPT


PICTURE_ASSESSMENT_PROMPT = """ First, identify the shapes and textual information that is \
                                displayed in the picture. \
                                The input consists of a base64-encoded image representing an image that is part of \
                                a power point slide or pdf page. \
                                Your goal is to assess whether the picture is relevant for the business 
                                meaning/contextualization \
                                in terms of people, organization, technology and business processes aspect. 
                                As a fundamental rule of thumb, if the picture does not contain any text, consider that \
                                picture to be irrelevant. If the picture contains text or text paragraphs, \
                                business processes diagram with texts, line/bar charts, that express \
                                clear business ideas, consider the picture to be relevant. \
                                If the picture seems to be a logo, please consider the picture to be irrelevant. \
                                
                                Provide your analysis only as a 'True' or as a 'False' boolean in the 'relevance' key; provide \
                                your thinking approach/reasoning behind your conclusion in the 'explanation' key. 
                                Provide the artefact type in the 'artefact' key. Artefact type values can be 'Business Processes Diagram', \
                                'Line Chart', 'Bar Chart', 'Logo' and any other relevant type you find relevant. \
                                If 'relevance' == True, based \
                                on the structure of the information, describe the image \
                                thoroughly. Then, after your own thorough analysis of the image and details, \
                                provide as many relevant details as possible \
                                leading to a greater understanding and provide the content summary in the 'content_summary' key.
                            """

FULL_TEXT_EXTRACTION_PROMPT = """
                              You are provided a parsed text input coming from a slide. The text input is structured with \
                              'Title', 'Subsection', 'Paragraph', 'Bullet point element', 'Speaker notes' and 'Placeholder_Image'.
                              For definition, a 'Title' represents the main header of the text and highlights the main topic\
                              , the subsection is a sub-header that highlights a sub topic in the text. A 'Paragraph' represents \
                              the articulation of ideas in free form text. The 'Bullet point element' represents a bullet point idea \
                              at a time. 'Speaker notes' are the notes written by the author to help present the slide. \
                              Finally, the 'Placeholder_Image' is just a placeholder of an existing image file. It is to mention that an image should be placed in \
                              the area of the text. Note the image_path, finishing with the file name and extension is placed there. \
                              Only regenerate 'Placeholder_Image' section that are given to you and only those. \
                              
                              Your goal is to keep the same text and re-generate it in a markdown format. Make sure to regenerate the \
                              'Title' and 'Subsection'.
                              Provide the output in the 'content' key in markdown format. \
                              
                              """