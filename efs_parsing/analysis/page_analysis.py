import sys, os

from efs_parsing.utilities.settings import BASE64_ENCODING_AUTO_SHAPE_TRIAGING, BASE64_ENCODING_CHART_TRIAGING, BASE64_ENCODING_CANVAS_TRIAGING
from efs_parsing.utilities.settings import BASE64_ENCODING_DIAGRAM_TRIAGING, BASE64_ENCODING_MATH_SHAPE_TRIAGING, BASE64_ENCODING_PICTURE_TRIAGING
from efs_parsing.utilities.settings import BASE64_ENCODING_LINE_TRIAGING, BASE64_ENCODING_SECTION_HEADER_TRIAGING

def analyze_pdf_pages(extracted_file_data,
                      default_export_labels):
    """
    TO COMPLETE 
    
    ## count number of power point auto shapes for each page
    ## count number of mathematical formulas for each page
    ## Be careful you are manipulating a docling object
    
    """
    file_page_analytics_repo = {}
    included_shape_list = [fig._value_
                            for fig in default_export_labels if fig
                            in ["section_header"]]
    
    for text_itm in extracted_file_data.document.texts:
        page_number = text_itm.prov[0].page_no
        if page_number not in file_page_analytics_repo.keys():
            file_page_analytics_repo[page_number] = {'section_header_count': 0}
        
        if text_itm.label in included_shape_list:
            file_page_analytics_repo[page_number]['section_header_count'] = file_page_analytics_repo[page_number]['section_header_count'] \
                + 1
    
    return file_page_analytics_repo


def analyze_pptx_pages(extracted_file_data,
                       shape_mapping_items,
                       document_shape_analyzer_object_name,
                       document_math_shape_analyzer_object_name,
                       equation_shapes,
                       not_math_shape_label):
    """
    TO COMPLETE 
    
    ## count number of power point auto shapes for each page
    ## count number of mathematical formulas for each page
    """
    file_page_analytics_repo = {}
    excluded_shape_list = [fig[1]
                            for fig in shape_mapping_items if fig[1] 
                            not in ["FREEFORM", "AUTO_SHAPE", 
                            "LINE", "CANVAS", "DIAGRAM", "CHART"]]

    for document_items in extracted_file_data:
        
        if document_items.page_content == document_shape_analyzer_object_name:
            
            page_number = document_items.metadata["page_number"]
            file_page_analytics_repo[page_number] = {}
            ## count number of items for all eligible shapes
            total_count_shapes = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig not in excluded_shape_list])
            count_auto_shape = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig == "AUTO_SHAPE"])
            count_chart = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig == "CHART"])
            count_canvas = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig == "CANVAS"])
            count_diagram = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig == "DIAGRAM"])
            count_line = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig == "LINE"])
            count_picture = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig == "PICTURE"])
            
            file_page_analytics_repo[page_number]["total_shapes"] = total_count_shapes
            file_page_analytics_repo[page_number]["count_auto_shape"] = count_auto_shape
            file_page_analytics_repo[page_number]["count_chart"] = count_chart
            file_page_analytics_repo[page_number]["count_canvas"] = count_canvas
            file_page_analytics_repo[page_number]["count_diagram"] = count_diagram
            file_page_analytics_repo[page_number]["count_line"] = count_line
            file_page_analytics_repo[page_number]["count_picture"] = count_picture
        
        if document_items.page_content == document_math_shape_analyzer_object_name:
            
            parser_identified_math_operators = list(equation_shapes.values())
            
            additive_operator = parser_identified_math_operators[0]
            substractive_operator = parser_identified_math_operators[1]
            multiplicative_operator = parser_identified_math_operators[2]
            divisor_operator = parser_identified_math_operators[3]
            equality_operator = parser_identified_math_operators[4]
            not_equal_operator = parser_identified_math_operators[5]
            
            count_shapes = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                            if fig != not_math_shape_label])
            
            count_pluses = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                             if fig == additive_operator])
            count_minuses = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                             if fig == substractive_operator])
            count_multiplicator = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                             if fig == multiplicative_operator])
            count_divisor = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                             if fig == divisor_operator])
            count_equals = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                             if fig == equality_operator])
            count_not_equals = len([fig for fig in document_items.metadata["emphasized_text_contents"]
                             if fig == not_equal_operator])
            
            file_page_analytics_repo[page_number]["math_shapes"] = count_shapes
            file_page_analytics_repo[page_number]["count_pluses"] = count_pluses
            file_page_analytics_repo[page_number]["count_minuses"] = count_minuses
            file_page_analytics_repo[page_number]["count_multiplicator"] = count_multiplicator
            file_page_analytics_repo[page_number]["count_divisor"] = count_divisor
            file_page_analytics_repo[page_number]["count_equals"] = count_equals
            file_page_analytics_repo[page_number]["count_not_equals"] = count_not_equals

    return file_page_analytics_repo


def triage_pages_for_base64_encoding(file_pages_analytics, 
                                     parsing_mode = "pptx"):
    """
    """
    def analysis_outcome(analytics,
                         parsing_mode = "pptx"):
        """
        """
        if parsing_mode == "pptx":
            triaging_rule = (analytics['count_auto_shape'] > BASE64_ENCODING_AUTO_SHAPE_TRIAGING or \
                            analytics['count_chart'] > BASE64_ENCODING_CHART_TRIAGING or \
                            analytics['count_canvas'] > BASE64_ENCODING_CANVAS_TRIAGING or \
                            analytics['count_diagram'] > BASE64_ENCODING_DIAGRAM_TRIAGING or \
                            analytics['math_shapes'] >= BASE64_ENCODING_MATH_SHAPE_TRIAGING or \
                            analytics['count_picture'] > BASE64_ENCODING_PICTURE_TRIAGING or \
                            analytics['count_line'] > BASE64_ENCODING_LINE_TRIAGING)

            complex_slide_identifier = (analytics['count_auto_shape'] > BASE64_ENCODING_AUTO_SHAPE_TRIAGING or \
                                        analytics['count_chart'] > BASE64_ENCODING_CHART_TRIAGING or \
                                        analytics['count_canvas'] > BASE64_ENCODING_CANVAS_TRIAGING or \
                                        analytics['count_diagram'] > BASE64_ENCODING_DIAGRAM_TRIAGING or \
                                        analytics['count_picture'] > BASE64_ENCODING_PICTURE_TRIAGING or \
                                        analytics['count_line'] > BASE64_ENCODING_LINE_TRIAGING)

            math_identifier = (analytics['math_shapes'] >= BASE64_ENCODING_MATH_SHAPE_TRIAGING)
            
            return triaging_rule, complex_slide_identifier, math_identifier
        
        if parsing_mode == "pdf":
            triaging_rule = (analytics['section_header_count'] >= BASE64_ENCODING_SECTION_HEADER_TRIAGING)
            
            complex_slide_identifier = (analytics['section_header_count'] >= BASE64_ENCODING_SECTION_HEADER_TRIAGING)
            
            return triaging_rule, complex_slide_identifier
        
    if parsing_mode == "pptx":
        
        # identify pages for base64 encoding
        repo_page_for_full_base64_encoding = {}
        for page_id, analytics in file_pages_analytics.items():
            
            triaging_rule, complex_slide_identifier,math_identifier = analysis_outcome(analytics,
                                                                                       parsing_mode = parsing_mode)
            
            if triaging_rule:
                repo_page_for_full_base64_encoding[page_id] = {}
                
                if complex_slide_identifier:
                    repo_page_for_full_base64_encoding[page_id]["complex_slide"] = True
                
                if math_identifier:
                    repo_page_for_full_base64_encoding[page_id]["math_equations"] = True
                    repo_page_for_full_base64_encoding[page_id]["count_pluses"] = analytics["count_pluses"]
                    repo_page_for_full_base64_encoding[page_id]["count_minuses"] = analytics["count_minuses"]
                    repo_page_for_full_base64_encoding[page_id]["count_multiplicator"] = analytics["count_multiplicator"]
                    repo_page_for_full_base64_encoding[page_id]["count_divisor"] = analytics["count_divisor"]
                    repo_page_for_full_base64_encoding[page_id]["count_equals"] = analytics["count_equals"]
                    repo_page_for_full_base64_encoding[page_id]["count_not_equals"] = analytics["count_not_equals"]
                        
        if bool(repo_page_for_full_base64_encoding):
            page_for_base64_conversion = ", ".join([str(item) for item in repo_page_for_full_base64_encoding])
            print(f"The page {page_for_base64_conversion} are planned for full base64 conversion")
        
        return repo_page_for_full_base64_encoding
    
    if parsing_mode == "pdf":
        
        # identify pages for base64 encoding
        repo_page_for_full_base64_encoding = {}
        for page_id, analytics in file_pages_analytics.items():
            
            triaging_rule, complex_slide_identifier = analysis_outcome(analytics,
                                                                       parsing_mode = parsing_mode)
            if triaging_rule:
                repo_page_for_full_base64_encoding[page_id] = {}

                if complex_slide_identifier:
                    repo_page_for_full_base64_encoding[page_id]["complex_slide"] = True
        
        if bool(repo_page_for_full_base64_encoding):
            page_for_base64_conversion = ", ".join([str(item) for item in repo_page_for_full_base64_encoding])
            print(f"The page {page_for_base64_conversion} are planned for full base64 conversion")

        return repo_page_for_full_base64_encoding