import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from fpdf import FPDF

# Load CV once (on app start)
@st.cache_data
def load_cv_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

cv_text = load_cv_text("cv/CV-EN-Kristine_2025_04.pdf")

# Define prompt template and chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that writes cover letters."),
    ("user", "Write a cover letter for the following job requirements:\n{job_requirements}\n\nUse the following CV information:\n{cv_text}")
])

llm = ChatBedrock(
    credentials_profile_name="default",
    region_name="eu-west-1",
    model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_kwargs={"max_tokens": 2048, "temperature": 0.1, "top_p": 1.0},
    # Add inference_profile param here if needed
)

chain = prompt | llm | StrOutputParser()

st.title("Cover Letter Generator")

job_requirements = st.text_area("Paste the job requirements here:", height=300)


if st.button("Generate Cover Letter") and job_requirements.strip():
    with st.spinner("Generating cover letter..."):
        cover_letter = chain.invoke({
            "job_requirements": job_requirements,
            "cv_text": cv_text
        })
    st.subheader("Generated Cover Letter")
    st.write(cover_letter)

    # Generate PDF with Unicode font
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Register and set DejaVuSans font (make sure path is correct)
    pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', size=12)

    # Write cover letter lines to PDF
    for line in cover_letter.split("\n"):
        pdf.cell(0, 10, line, ln=True)

    pdf_output_path = "cover_letter.pdf"
    pdf.output(pdf_output_path)

    # Provide download button
    with open(pdf_output_path, "rb") as f:
        st.download_button(
            label="Download Cover Letter as PDF",
            data=f,
            file_name="cover_letter.pdf",
            mime="application/pdf"
        )
