import os
from src.services.pdf_generator import generate_pdf_from_markdown

test_title = "Special Cabinet Committee Meeting"
test_markdown = """
# Special Cabinet Committee Meeting
> PIB UPSC Intelligence Note | *03 October 2026*

---

## 📌 Executive Summary
The Union Cabinet has restructured high-level advisory bodies to streamline policy implementation and monitor critical infrastructure projects.

## 💡 Why It Matters for UPSC
Directly impacts executive governance, accountability, and the operational division under the Government of India (Allocation of Business) Rules.

## 📝 Smart Notes (Key Exam Highlights)
* Cabinet Committees are extra-constitutional bodies not explicitly mentioned in the text of the Constitution.
* Established under the Rules of Business framed under Article 77(3).
* Headed usually by the Prime Minister; however, the Committee on Accommodation and Parliamentary Affairs have other senior ministers heading them.

## 🔗 Backward Linkages (Static Syllabus & Case Laws)
* Article 74 & 75 (Council of Ministers)
* Article 77 (Conduct of Government Business)
* Ram Jawaya Kapur v. State of Punjab (1955) on executive power scope.

## ✍️ Mains Analytical Practice Question
> *"Evaluate the significance of Cabinet Committees in enabling cabinet governance without diluting ministerial accountability."*
"""

def test_pdf():
    print("Testing PDF rendering from Markdown...")
    pdf_buffer = generate_pdf_from_markdown(title=test_title, markdown_content=test_markdown)
    
    output_path = "test_note_output.pdf"
    with open(output_path, "wb") as f:
        f.write(pdf_buffer.getvalue())
        
    print(f"✅ Success! PDF written to: {os.path.abspath(output_path)}")
    print(f"File size: {round(os.path.getsize(output_path) / 1024, 2)} KB")

if __name__ == "__main__":
    test_pdf()
