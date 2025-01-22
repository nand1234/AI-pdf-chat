import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with fitz.open(pdf_path) as pdf_document:
        text = ""
        # Iterate through each page of the PDF
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)  # Get page object
            text += page.get_text()  # Extract text from the page

    return text

if __name__ =='__main':
    # Example usage
    pdf_path = 'sample_pdf/Nandkumar_Ghatage_Latest_CV.pdf'  # Path to your PDF file
    extracted_text = extract_text_from_pdf(pdf_path)

    # Print the extracted text (you can also write it to a file if needed)
    print(extracted_text)
