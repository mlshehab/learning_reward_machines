import xml.etree.ElementTree as ET


def indent(elem, level=0):
    """
    Indents the XML element tree to make it pretty-printed.

    Parameters:
    - elem: The root element of the XML tree.
    - level: Current indentation level (used for recursion).
    """
    i = "\n" + "  " * level  # Indentation per level
    if len(elem):  # If element has children
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)  # Recursively indent child elements
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def format_xml(input_filename, output_filename=None):
    """
    Formats the XML file to be more readable by adding proper indentation.

    Parameters:
    - input_filename: The input XML file to be formatted.
    - output_filename: The output XML file where the formatted XML will be saved.
                       If not provided, it will overwrite the input file.
    """
    # Parse the XML file
    tree = ET.parse(input_filename)
    root = tree.getroot()

    # Indent the XML tree
    indent(root)

    # Determine output filename
    if output_filename is None:
        output_filename = input_filename  # Overwrite the input file if no output filename provided

    # Write the formatted XML back to the file
    tree.write(output_filename, encoding="utf-8", xml_declaration=True)
    print(f"Formatted XML written to {output_filename}.")


# Example usage
input_file = "state_traces_blockworld.xml"  # Replace with your input XML file path
output_file = "state_traces_blockworld.xml"  # Replace with your desired output file path
format_xml(input_file, output_file)
