"""
Generate embeddings for corpus chunks and queries.
Run: python3 generate_corpus_embeddings.py
"""
import json
from sentence_transformers import SentenceTransformer

CHUNKS = [
    # AI
    {"id": 0,  "text": "Neural networks are the foundation of modern artificial intelligence. These deep learning models learn hierarchical representations from raw data. Training a neural network requires large labelled datasets and significant GPU compute. Backpropagation adjusts internal weights to minimise prediction error across millions of examples."},
    {"id": 1,  "text": "Deep learning has reshaped entire industries over the past decade. Convolutional neural networks revolutionised computer vision, enabling machines to classify images with superhuman accuracy. Recurrent and transformer-based architectures extended deep learning to natural language, powering translation systems, code generators, and conversational agents. Training these neural network models demands distributed clusters of specialised hardware running continuously for weeks. Researchers have developed techniques such as dropout, batch normalisation, and learning-rate scheduling to stabilise the artificial intelligence training process and improve generalisation. Despite the engineering complexity, the end result is a model capable of solving tasks that were considered impossible just a decade ago. The gap between human and machine performance continues to close across vision, speech, and language benchmarks."},
    {"id": 2,  "text": "Layered computational architectures that mimic cognitive processes have transformed automated perception. These systems develop internal representations by iteratively adjusting weighted connections during a supervised optimisation phase. Gradient-based methods allow the architecture to reduce a loss objective across millions of examples, producing models that surpass human accuracy on complex classification tasks. The resulting systems power autonomous vehicles, medical diagnostics, and real-time language translation."},
    {"id": 3,  "text": "The neural network of underground roads connects city districts. Researchers are deep learning about ancient Roman history through archaeological digs. The architectural model required six months of training to build by hand. Artificial intelligence tests student aptitude at the school admissions office."},
    # Finance
    {"id": 4,  "text": "Stock markets are the engine of modern economies. Investors allocate capital to publicly listed companies in exchange for equity shares, expecting financial returns through price appreciation and dividends. Portfolio diversification across asset classes reduces investment risk while maintaining exposure to long-term market growth."},
    {"id": 5,  "text": "Investment management is both a science and an art. Equity markets allow capital to flow from savers to productive enterprises, fuelling innovation and economic growth. Fixed-income instruments such as government and corporate bonds provide predictable returns at lower risk, forming the bedrock of conservative portfolios. Modern portfolio theory demonstrates that combining assets with low correlations reduces volatility without sacrificing expected returns. Passive index funds have democratised investing by delivering market returns at minimal cost, outperforming the majority of actively managed funds over long horizons. Central bank interest rate decisions exert powerful influence over asset valuations, as rising rates compress the present value of future cash flows and make bonds more attractive relative to stocks. Risk management frameworks such as Value at Risk and stress testing help institutions quantify their financial exposure to adverse market movements."},
    {"id": 6,  "text": "Capital allocation decisions shape the trajectory of enterprises and entire economies. Institutional asset managers deploy wealth across equities, fixed-income securities, and alternative instruments to achieve target risk-adjusted yields. Valuation models discount projected cash flows to determine whether a security is priced attractively relative to fundamentals. Macroeconomic conditions, including monetary policy and fiscal stimulus, exert significant influence over the relative attractiveness of different asset classes."},
    {"id": 7,  "text": "The stock market stall sold the freshest produce at the weekend fair. Investment in new kitchen appliances improved the restaurant's portfolio of dishes. Financial returns on the school bake sale exceeded expectations. Patient investors in the amateur theatre play waited for the curtain to rise."},
    # Health
    {"id": 8,  "text": "Modern medicine has transformed the treatment of infectious and chronic diseases. Vaccines have controlled illnesses that once killed millions, while antibiotics revolutionised the management of bacterial infections. Healthcare systems worldwide face growing pressure from ageing populations and the rising burden of non-communicable diseases such as diabetes, cardiovascular disease, and cancer."},
    {"id": 9,  "text": "The global burden of disease is shifting from infectious to chronic conditions as populations age and urbanise. Cardiovascular disease remains the leading cause of death worldwide, driving demand for statins, blood pressure medications, and surgical interventions. Cancer treatment has been transformed by targeted therapies and immunotherapies that exploit the immune system to destroy tumour cells. Diabetes management has advanced with continuous glucose monitors and automated insulin delivery systems. Mental health disorders, including depression and anxiety, account for a substantial share of disability-adjusted life years, yet treatment rates remain low in most countries. Digital health technologies, from wearable biosensors to AI-assisted diagnostic imaging, are extending the reach of healthcare delivery. Gene therapies offer the prospect of curing hereditary conditions by correcting faulty DNA sequences. Public health interventions such as vaccination programmes, tobacco control, and improved sanitation have historically delivered the largest gains in life expectancy."},
    {"id": 10, "text": "Advances in biomedical science have fundamentally altered how clinicians manage illness. Pharmacological agents targeting specific molecular pathways have improved therapeutic outcomes for previously untreatable conditions. Epidemiological research identifies risk factors that drive the incidence of non-communicable ailments, guiding preventive interventions at the population level. Surgical techniques refined over decades allow minimally invasive procedures that accelerate patient recovery and reduce complication rates."},
    {"id": 11, "text": "The patient gardener treated each disease-stricken plant with medicine from the garden centre. Healthcare workers dressed as doctors entertained children at the birthday party. The disease of procrastination spread quickly through the treatment group during the yoga retreat. Medicine cabinet doors were painted as a decoration in the hospital-themed kindergarten classroom."},
    # Climate
    {"id": 12, "text": "Climate change is accelerating at an unprecedented rate. Global warming driven by greenhouse gas emissions is raising average temperatures, melting polar ice caps, and intensifying extreme weather events. Renewable energy sources such as solar panels and wind turbines are the primary tools for reducing carbon emissions and limiting further temperature rise."},
    {"id": 13, "text": "The transition to renewable energy is one of the defining challenges of the twenty-first century. Solar power capacity has grown exponentially as the cost of photovoltaic panels has fallen by more than ninety percent over the last two decades. Wind energy now supplies a significant fraction of electricity in many European countries. Despite this progress, global carbon emissions from fossil fuels remain near record highs. Scientists warn that average global warming must be kept below 1.5 degrees Celsius to avoid the most catastrophic consequences of climate change, including widespread coastal flooding, agricultural collapse, and mass species extinction. Governments are implementing carbon pricing, renewable energy mandates, and phase-outs of petrol vehicles to accelerate the clean energy transition. Grid-scale battery storage is emerging as a critical technology for balancing the intermittency of solar and wind generation."},
    {"id": 14, "text": "Atmospheric heating caused by anthropogenic greenhouse gases is destabilising planetary ecosystems. The rapid expansion of photovoltaic arrays and wind turbines represents humanity's primary strategy for decarbonising electricity generation. Carbon neutrality targets set by governments around the world aim to halt the rise in average surface temperatures before irreversible tipping points are reached. Coastal communities face growing risks from sea-level rise driven by the melting of polar ice sheets."},
    {"id": 15, "text": "The political climate changed after the heated debate. Global warming friendships between athletes transcend national boundaries. Renewable energy and enthusiasm from fans drove the team to victory. Solar panels at the outdoor concert provided shade for the spectators."},
    # Space
    {"id": 16, "text": "The race to reach Mars is intensifying as both government agencies and private companies develop next-generation rockets. NASA and SpaceX have outlined crewed mission architectures targeting the Martian surface within this decade. Astronauts on a Mars mission would face extreme radiation exposure, a six-month transit, and the challenge of surviving on a planet with no breathable atmosphere."},
    {"id": 17, "text": "Space exploration entered a new era with the rise of reusable rocket technology. SpaceX demonstrated that orbital-class boosters could land vertically and fly again, dramatically reducing the cost of reaching orbit. NASA's Artemis programme aims to return astronauts to the Moon as a stepping stone toward Mars. The James Webb Space Telescope has opened an entirely new window on the early universe, imaging galaxies formed just hundreds of millions of years after the Big Bang. Satellite constellations in low Earth orbit are providing global broadband internet coverage and high-resolution Earth observation data. Scientists are also studying moons like Europa and Enceladus, which may harbour liquid water oceans beneath their icy surfaces and could potentially support microbial life. The Mars Perseverance rover continues to drill rock cores and cache samples for a future return mission."},
    {"id": 18, "text": "Interplanetary travel has moved from science fiction to engineering reality. Reusable propulsion vehicles capable of vertical landing have slashed the expense of reaching orbit. Crewed missions to the red planet are being planned by multiple aerospace organisations, with cosmonauts expected to traverse the Martian surface within the coming decades. Orbital telescopes are revealing the structure of distant galaxies and characterising the atmospheres of exoplanets that may harbour conditions suitable for life."},
    {"id": 19, "text": "The rocket launch party entertained neighbourhood children at the primary school. An astronaut costume was the most popular choice at the school play. The Mars chocolate bar outsold all competitors at the stadium kiosk. Mission accomplished signs decorated the graduation ceremony hall."},
]

QUERIES = [
    {"key": "ai",      "text": "neural network deep learning artificial intelligence model training"},
    {"key": "finance", "text": "stock market investment portfolio financial returns capital"},
    {"key": "health",  "text": "disease treatment medicine healthcare patient"},
    {"key": "climate", "text": "climate change global warming renewable energy solar wind carbon"},
    {"key": "space",   "text": "Mars mission rocket astronaut space exploration"},
]

print("Loading model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

chunk_texts = [c["text"] for c in CHUNKS]
query_texts = [q["text"] for q in QUERIES]

print(f"Encoding {len(chunk_texts)} chunks + {len(query_texts)} queries...")
chunk_embs = model.encode(chunk_texts, normalize_embeddings=True).tolist()
query_embs = model.encode(query_texts, normalize_embeddings=True).tolist()

chunk_out = {str(c["id"]): emb for c, emb in zip(CHUNKS, chunk_embs)}
query_out = {q["key"]: emb for q, emb in zip(QUERIES, query_embs)}

with open("src/data/corpus_embeddings.json", "w") as f:
    json.dump({"chunks": chunk_out, "queries": query_out}, f)

print(f"Done. {len(chunk_out)} chunk embeddings + {len(query_out)} query embeddings saved.")

# Also compute 2D PCA for scatter plot
import numpy as np
all_vecs = np.array(chunk_embs + query_embs)
cov = np.cov(all_vecs.T)
eigvals, eigvecs = np.linalg.eigh(cov)
top2 = eigvecs[:, np.argsort(eigvals)[::-1][:2]]
proj = all_vecs @ top2
mn, mx = proj.min(0), proj.max(0)
proj_n = (proj - mn) / (mx - mn) * 2 - 1

n = len(CHUNKS)
pos_chunks = {str(CHUNKS[i]["id"]): proj_n[i].tolist() for i in range(n)}
pos_queries = {QUERIES[i]["key"]: proj_n[n + i].tolist() for i in range(len(QUERIES))}

with open("src/data/corpus_positions_2d.json", "w") as f:
    json.dump({"chunks": pos_chunks, "queries": pos_queries}, f)

print("2D positions saved.")
