# !pip install -q pymupdf
# !python -m spacy download es_core_news_sm

import pymupdf as p
import spacy
import re
from transformers import pipeline

pdf = "https://www.dropbox.com/scl/fi/xzmwmwp28dxe3hnghdsox/Informe-estudiantes-UNAHUR-2022-2023.pdf?rlkey=sursp0vijlkiyhunsgnbsf81a&st=uwgas41a&dl=1"
open_pdf = p.open(pdf)

data = []
for page in range(len(open_pdf)):
  page_loaded = open_pdf.load_page(page)
  data.append(page_loaded.get_text())

data = [m.replace("\n", ".") for m in data]
data = [m.replace(" .", ".") for m in data]
data = [m.replace("..", ".") for m in data]
data = " ".join(data)
nlp = spacy.load("es_core_news_sm")
doc = nlp(data)

sentences = []
for sent in doc.sents:
  if any(token.pos_ == "VERB"  for token in sent) and any(token.pos_ == "NOUN" for token in sent):
    sentences.append(sent)

for idx, sent in enumerate(sentences):
  if len(re.findall("[0-9]", str(sent))) == 0:
    sentences.pop(idx)

with open("UNAHUR-Students-data.txt", "w", encoding="utf-8") as st:
  for sent in sentences:
    st.write(f"{sent}\n")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

oraciones = [str(sent).strip() for sent in sentences if str(sent).strip()]
etiquetas = ["desercion estudiantil", "rendimiento academico", "otros"]
oraciones_de_desercion = []

for oracion in oraciones:
    resultado = classifier(oracion, etiquetas)
    if resultado['labels'][0] == "desercion estudiantil" and resultado['scores'][0] > 0.5:
        oraciones_de_desercion.append(oracion)

print("Oraciones sobre deserción:")
for o in oraciones_de_desercion:
    print("-", o)

"""Aclaracion: Luego de haber identificado todas las oraciones relevantes sobre la desercion estudiantil de UNAHUR 2021, le pedi a la IA que me arme el csv, para agilizar y automatizar tareas pesadas e ineficientes que demoran mucho tiempo"""