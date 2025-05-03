import fitz  # PyMuPDF

def parse_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])
