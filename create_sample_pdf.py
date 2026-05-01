"""Run once to generate a sample proposal PDF: python create_sample_pdf.py"""
from fpdf import FPDF, XPos, YPos


def heading(pdf: FPDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)


def body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(pdf.l_margin)
    effective_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(effective_w, 6, text)
    pdf.ln(2)


def bullet_list(pdf: FPDF, items: list) -> None:
    pdf.set_font("Helvetica", "", 11)
    effective_w = pdf.w - pdf.l_margin - pdf.r_margin
    for i, item in enumerate(items, 1):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(effective_w, 6, f"{i}. {item}")
    pdf.ln(2)


pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
effective_w = pdf.w - pdf.l_margin - pdf.r_margin

# Title block
pdf.set_font("Helvetica", "B", 16)
pdf.set_x(pdf.l_margin)
pdf.cell(effective_w, 10, "Research Proposal", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

pdf.set_font("Helvetica", "B", 13)
pdf.set_x(pdf.l_margin)
pdf.cell(effective_w, 8,
         "Federated Learning for Healthcare Privacy Preservation",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

pdf.set_font("Helvetica", "", 11)
pdf.set_x(pdf.l_margin)
pdf.cell(effective_w, 6, "Submitted by: Udula  |  SE4010 CTSE Assignment 2",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.ln(5)

# Keywords
pdf.set_x(pdf.l_margin)
pdf.set_font("Helvetica", "B", 11)
pdf.write(6, "Keywords: ")
pdf.set_font("Helvetica", "", 11)
pdf.write(6, "federated learning, privacy preservation, healthcare, "
             "differential privacy, deep learning, distributed systems")
pdf.ln(8)

heading(pdf, "1. Introduction")
body(pdf,
     "Healthcare institutions generate vast amounts of sensitive patient data that can be "
     "used to train powerful machine learning models. However, centralising this data raises "
     "serious privacy and regulatory concerns, particularly under HIPAA and GDPR. Federated "
     "learning (FL) offers a promising alternative by enabling model training across distributed "
     "hospital nodes without sharing raw patient data. This proposal investigates the design "
     "and evaluation of a privacy-preserving federated learning system tailored for medical "
     "imaging classification tasks.")

heading(pdf, "2. Objectives")
bullet_list(pdf, [
    "To design a federated learning architecture for multi-hospital medical imaging datasets.",
    "To implement and evaluate differential privacy mechanisms (Gaussian noise injection).",
    "To benchmark the privacy-utility trade-off against centralised baseline models.",
    "To develop an open-source framework and publish reproducible experimental results.",
])

heading(pdf, "3. Scope")
body(pdf,
     "The scope covers federated learning across a simulated network of at least five hospital "
     "nodes using de-identified chest X-ray datasets (CheXpert and MIMIC-CXR). The study "
     "focuses on binary and multi-label classification tasks. Out of scope: natural language "
     "processing, real-time inference systems, and non-medical imaging domains.")

heading(pdf, "4. Methodology")
body(pdf,
     "The research employs the FedAvg aggregation algorithm with differential privacy noise "
     "injection at the client level. Each participating node trains a ResNet-18 model locally "
     "on its data subset, then uploads only model gradients to a central aggregator. Privacy "
     "budgets are tracked using the Renyi Differential Privacy accountant. Experiments use "
     "the Flower (flwr) federated learning framework and PyTorch. Evaluation metrics include "
     "AUC-ROC, accuracy, and privacy budget epsilon (eps).")

heading(pdf, "5. Expected Contributions")
bullet_list(pdf, [
    "A modular, open-source federated learning pipeline for medical imaging.",
    "Empirical analysis of the privacy-utility trade-off at varying epsilon values.",
    "A reproducible benchmark comparing FedAvg, FedProx, and centralised training.",
    "Deployment guidelines for FL systems in real-world hospital networks.",
])

heading(pdf, "6. Timeline")
body(pdf,
     "Week 1-2: Literature review and development environment setup.\n"
     "Week 3-4: Dataset preparation and centralised baseline model training.\n"
     "Week 5-6: Federated pipeline implementation with differential privacy.\n"
     "Week 7-8: Experiments, evaluation, and final report writing.")

heading(pdf, "7. References")
pdf.set_font("Helvetica", "", 10)
refs = [
    "McMahan et al. (2017). Communication-Efficient Learning of Deep Networks. AISTATS.",
    "Dwork & Roth (2014). The Algorithmic Foundations of Differential Privacy. Now Publishers.",
    "Li et al. (2020). Federated Learning: Challenges, Methods, and Future Directions. IEEE SPM.",
    "Rieke et al. (2020). The Future of Digital Health with Federated Learning. npj Digital Medicine.",
    "Bonawitz et al. (2019). Towards Federated Learning at Scale: A System Design. MLSys.",
]
for ref in refs:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(effective_w, 5, f"  {ref}")
    pdf.ln(1)

pdf.output("sample_proposal.pdf")
print("sample_proposal.pdf created successfully.")
