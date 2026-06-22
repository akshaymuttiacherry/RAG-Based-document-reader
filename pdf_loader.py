from PyPDF2 import PdfReader

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        content = page.extract_text()

        if content:
            text += content

    return text