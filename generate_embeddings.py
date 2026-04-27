"""
Generate embeddings for all articles using bge-base-en-v1.5.
Run once: python generate_embeddings.py
Output: src/data/embeddings.json
"""

import json
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence_transformers not found. Activate the venv first.")
    sys.exit(1)

ARTICLES = [
    # AI
    {"id": 0, "title": "Neural Networks Achieve Superhuman Accuracy on Image Recognition", "text": "A new convolutional neural network model surpassed human-level accuracy on the ImageNet benchmark. The architecture uses attention mechanisms and self-supervised pre-training on unlabeled data."},
    {"id": 1, "title": "Deep Learning Model Outperforms Radiologists at Cancer Detection", "text": "Researchers trained a deep learning system on 100,000 chest X-rays. The model detected early-stage tumors with higher sensitivity than board-certified radiologists in a double-blind study."},
    {"id": 2, "title": "Artificial Minds Master Complex Strategic Reasoning", "text": "An intelligent system developed by a research lab defeated world champions at multi-player strategy games. The agent learned purely through self-play without any human demonstration data."},
    {"id": 3, "title": "Machine Learning Machine Learning Machine Learning Transforms Industries", "text": "Machine learning machine learning applications are growing. Machine learning is used in finance, healthcare, and retail. Machine learning machine learning tools are now accessible to small businesses."},
    {"id": 4, "title": "GPU Clusters Power the Next Generation of Intelligent Systems", "text": "Data centers equipped with thousands of graphics processing units now train models with hundreds of billions of parameters. Energy consumption has become the primary bottleneck for scaling intelligent systems."},
    # Space
    {"id": 5, "title": "SpaceX Rocket Successfully Lands on Mars Surface", "text": "A reusable rocket developed by SpaceX completed the first crewed Mars landing. Astronauts will spend 18 months on the Martian surface conducting geological surveys and atmospheric experiments."},
    {"id": 6, "title": "Interplanetary Vessel Touches Down on the Red Planet", "text": "An interplanetary vessel carrying six crew members successfully touched down after a seven-month transit. The crew will establish a base camp near a subsurface water ice deposit recently identified by orbital radar."},
    {"id": 7, "title": "Space Space Space Mission Breaks Distance Record", "text": "Space exploration space mission sets new space record. Space agency launches space rocket into deep space. Space space space astronauts travel through outer space."},
    {"id": 8, "title": "James Webb Telescope Captures Galaxy Formed 300 Million Years After Big Bang", "text": "The James Webb Space Telescope imaged a galaxy dating to just 300 million years after the Big Bang. Scientists were surprised by its unexpectedly high stellar mass, challenging current models of early universe formation."},
    {"id": 9, "title": "Astronomers Detect Liquid Ocean Beneath Europa Ice Shell", "text": "Radio telescope observations confirmed the presence of a saltwater ocean beneath the frozen crust of Jupiter's moon Europa. The ocean is estimated to contain twice the volume of all Earth's oceans combined."},
    # Health
    {"id": 10, "title": "New Drug Reduces Heart Attack Risk by 40% in Clinical Trial", "text": "A phase 3 clinical trial of a novel LDL-lowering drug showed a 40% reduction in major cardiovascular events. The drug works by inhibiting PCSK9, a protein that regulates cholesterol absorption in the liver."},
    {"id": 11, "title": "Cardiovascular Treatment Shows Breakthrough Results in High-Risk Patients", "text": "A new cardiovascular therapy demonstrated significant reduction in myocardial infarction rates among patients with pre-existing coronary artery disease. The treatment targets arterial plaque buildup directly."},
    {"id": 12, "title": "Heart Attack Heart Attack Heart Attack Prevention Study", "text": "Heart attack prevention is important. Heart attack risk factors include heart attack triggers. Heart attack heart attack rates have increased. Heart attack awareness saves lives from heart attacks."},
    {"id": 13, "title": "CRISPR Gene Editing Achieves Functional Cure for Sickle Cell Disease", "text": "A single CRISPR-based gene therapy treatment produced functional cures in 28 of 29 patients with sickle cell disease. All treated patients remained transfusion-independent at 24-month follow-up."},
    {"id": 14, "title": "Wearable Sensor Detects Atrial Fibrillation Before Stroke Occurs", "text": "A smartwatch algorithm identified undiagnosed atrial fibrillation in over 50,000 patients during a large population study. Early detection allowed anticoagulation therapy to prevent strokes in high-risk individuals."},
    # Finance
    {"id": 15, "title": "Bitcoin Surges 60% After Spot ETF Approval by SEC", "text": "Bitcoin's price surged 60% in the week following SEC approval of the first spot Bitcoin exchange-traded fund. Institutional investors poured over $10 billion into Bitcoin ETF products in the first month."},
    {"id": 16, "title": "Digital Asset Markets Rally as Regulatory Clarity Emerges", "text": "Cryptocurrency markets rallied broadly as regulatory frameworks clarified the legal status of digital tokens. Institutional capital allocation to blockchain-based assets reached record levels in Q3."},
    {"id": 17, "title": "Stock Market Stock Market Volatility Stock Market Crash Warning", "text": "Stock market volatility stock market indicators suggest stock market instability. Stock market stock market analysts warn of stock market correction. Stock market stock market stock market."},
    {"id": 18, "title": "Federal Reserve Holds Interest Rates Amid Inflation Concerns", "text": "The Federal Reserve kept its benchmark interest rate unchanged at 5.25% despite pressure from investors expecting cuts. Fed officials cited persistent inflation in services and shelter costs as reasons for caution."},
    {"id": 19, "title": "Ethereum Layer-2 Networks Process 10x More Transactions Than Mainnet", "text": "Ethereum scaling solutions including Arbitrum and Optimism collectively processed ten times more daily transactions than the Ethereum mainnet. Average fees on Layer-2 networks dropped below one cent per transaction."},
    # Climate
    {"id": 20, "title": "Solar Panel Efficiency Record Broken at 47.6%", "text": "A research team achieved 47.6% solar panel efficiency using a multi-junction concentrator design. Commercial panels currently average 22-24% efficiency, making this result a major milestone for photovoltaic technology."},
    {"id": 21, "title": "Photovoltaic Arrays Power Entire Cities During Summer Peak Demand", "text": "Regions with high photovoltaic installation density met 100% of electricity demand from solar generation on multiple summer days. Grid-scale battery storage allowed surplus energy to be discharged after sunset."},
    {"id": 22, "title": "Climate Climate Climate Change Threatens Climate Climate Ecosystems", "text": "Climate change climate impact on climate systems. Climate climate warming affects climate biodiversity. Climate change climate change climate solutions needed for climate crisis climate emergency."},
    {"id": 23, "title": "Offshore Wind Capacity Doubles as Turbine Technology Improves", "text": "Global offshore wind capacity doubled in three years as turbine blade length increased to over 100 meters. Floating offshore wind platforms now allow installation in deep water far from coastlines."},
    {"id": 24, "title": "Arctic Sea Ice Reaches Lowest Recorded Summer Extent", "text": "Arctic sea ice extent in September reached its lowest recorded level since satellite monitoring began in 1979. Scientists attribute the decline to accelerating warming at the poles, which is occurring three times faster than the global average."},
]

print("Loading bge-base-en-v1.5 model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

texts = [f"{a['title']} {a['text']}" for a in ARTICLES]
print(f"Encoding {len(texts)} articles...")
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

output = {str(a["id"]): emb for a, emb in zip(ARTICLES, embeddings)}

out_path = "src/data/embeddings.json"
with open(out_path, "w") as f:
    json.dump(output, f)

print(f"Done. Saved {len(output)} embeddings to {out_path}")
print(f"Embedding dimension: {len(embeddings[0])}")
