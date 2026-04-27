// Dataset designed to demonstrate retrieval method differences.
// Each topic has:
//   - "normal" articles (both BM25 and semantic find them)
//   - a "keyword spam" article (BM25 ranks it high, semantic ignores it)
//   - a "synonym" article (only semantic finds it — different words, same meaning)

export const ARTICLES = [

  // ── AI / MACHINE LEARNING ────────────────────────────────────────────────
  {
    id: 0, category: "ai",
    title: "Neural Networks Achieve Superhuman Accuracy on Image Recognition",
    text: "A new convolutional neural network model surpassed human-level accuracy on the ImageNet benchmark. The architecture uses attention mechanisms and self-supervised pre-training on unlabeled data.",
    note: "normal",
  },
  {
    id: 1, category: "ai",
    title: "Deep Learning Model Outperforms Radiologists at Cancer Detection",
    text: "Researchers trained a deep learning system on 100,000 chest X-rays. The model detected early-stage tumors with higher sensitivity than board-certified radiologists in a double-blind study.",
    note: "normal",
  },
  {
    id: 2, category: "ai",
    title: "Artificial Minds Master Complex Strategic Reasoning",
    text: "An intelligent system developed by a research lab defeated world champions at multi-player strategy games. The agent learned purely through self-play without any human demonstration data.",
    note: "synonym — uses 'artificial minds' and 'intelligent system' instead of AI or neural network. Semantic search finds it, BM25 misses it on query 'AI machine learning'.",
  },
  {
    id: 3, category: "ai",
    title: "Machine Learning Machine Learning Machine Learning Transforms Industries",
    text: "Machine learning machine learning applications are growing. Machine learning is used in finance, healthcare, and retail. Machine learning machine learning tools are now accessible to small businesses.",
    note: "keyword spam — BM25 ranks this very high on 'machine learning' query because of raw frequency. Semantic search ignores it because the content is meaningless.",
  },
  {
    id: 4, category: "ai",
    title: "GPU Clusters Power the Next Generation of Intelligent Systems",
    text: "Data centers equipped with thousands of graphics processing units now train models with hundreds of billions of parameters. Energy consumption has become the primary bottleneck for scaling intelligent systems.",
    note: "synonym — talks about AI infrastructure without using the word AI.",
  },

  // ── SPACE EXPLORATION ────────────────────────────────────────────────────
  {
    id: 5, category: "space",
    title: "SpaceX Rocket Successfully Lands on Mars Surface",
    text: "A reusable rocket developed by SpaceX completed the first crewed Mars landing. Astronauts will spend 18 months on the Martian surface conducting geological surveys and atmospheric experiments.",
    note: "normal",
  },
  {
    id: 6, category: "space",
    title: "Interplanetary Vessel Touches Down on the Red Planet",
    text: "An interplanetary vessel carrying six crew members successfully touched down after a seven-month transit. The crew will establish a base camp near a subsurface water ice deposit recently identified by orbital radar.",
    note: "synonym — 'interplanetary vessel' = rocket, 'red planet' = Mars. Semantic finds it, BM25 misses it on query 'Mars rocket landing'.",
  },
  {
    id: 7, category: "space",
    title: "Space Space Space Mission Breaks Distance Record",
    text: "Space exploration space mission sets new space record. Space agency launches space rocket into deep space. Space space space astronauts travel through outer space.",
    note: "keyword spam — BM25 ranks this top on any space query. Useless content.",
  },
  {
    id: 8, category: "space",
    title: "James Webb Telescope Captures Galaxy Formed 300 Million Years After Big Bang",
    text: "The James Webb Space Telescope imaged a galaxy dating to just 300 million years after the Big Bang. Scientists were surprised by its unexpectedly high stellar mass, challenging current models of early universe formation.",
    note: "normal",
  },
  {
    id: 9, category: "space",
    title: "Astronomers Detect Liquid Ocean Beneath Europa Ice Shell",
    text: "Radio telescope observations confirmed the presence of a saltwater ocean beneath the frozen crust of Jupiter's moon Europa. The ocean is estimated to contain twice the volume of all Earth's oceans combined.",
    note: "normal",
  },

  // ── HEALTH / MEDICINE ────────────────────────────────────────────────────
  {
    id: 10, category: "health",
    title: "New Drug Reduces Heart Attack Risk by 40% in Clinical Trial",
    text: "A phase 3 clinical trial of a novel LDL-lowering drug showed a 40% reduction in major cardiovascular events. The drug works by inhibiting PCSK9, a protein that regulates cholesterol absorption in the liver.",
    note: "normal",
  },
  {
    id: 11, category: "health",
    title: "Cardiovascular Treatment Shows Breakthrough Results in High-Risk Patients",
    text: "A new cardiovascular therapy demonstrated significant reduction in myocardial infarction rates among patients with pre-existing coronary artery disease. The treatment targets arterial plaque buildup directly.",
    note: "synonym — 'myocardial infarction' = heart attack, 'cardiovascular' relates to heart. Semantic finds it on 'heart attack' query, BM25 misses it.",
  },
  {
    id: 12, category: "health",
    title: "Heart Attack Heart Attack Heart Attack Prevention Study",
    text: "Heart attack prevention is important. Heart attack risk factors include heart attack triggers. Heart attack heart attack rates have increased. Heart attack awareness saves lives from heart attacks.",
    note: "keyword spam — BM25 ranks this first on 'heart attack' query. Zero medical value.",
  },
  {
    id: 13, category: "health",
    title: "CRISPR Gene Editing Achieves Functional Cure for Sickle Cell Disease",
    text: "A single CRISPR-based gene therapy treatment produced functional cures in 28 of 29 patients with sickle cell disease. All treated patients remained transfusion-independent at 24-month follow-up.",
    note: "normal",
  },
  {
    id: 14, category: "health",
    title: "Wearable Sensor Detects Atrial Fibrillation Before Stroke Occurs",
    text: "A smartwatch algorithm identified undiagnosed atrial fibrillation in over 50,000 patients during a large population study. Early detection allowed anticoagulation therapy to prevent strokes in high-risk individuals.",
    note: "normal",
  },

  // ── FINANCE / CRYPTO ─────────────────────────────────────────────────────
  {
    id: 15, category: "finance",
    title: "Bitcoin Surges 60% After Spot ETF Approval by SEC",
    text: "Bitcoin's price surged 60% in the week following SEC approval of the first spot Bitcoin exchange-traded fund. Institutional investors poured over $10 billion into Bitcoin ETF products in the first month.",
    note: "normal",
  },
  {
    id: 16, category: "finance",
    title: "Digital Asset Markets Rally as Regulatory Clarity Emerges",
    text: "Cryptocurrency markets rallied broadly as regulatory frameworks clarified the legal status of digital tokens. Institutional capital allocation to blockchain-based assets reached record levels in Q3.",
    note: "synonym — 'digital assets' = crypto, 'blockchain-based assets' = cryptocurrency. Semantic finds it, BM25 misses it on 'Bitcoin crypto' query.",
  },
  {
    id: 17, category: "finance",
    title: "Stock Market Stock Market Volatility Stock Market Crash Warning",
    text: "Stock market volatility stock market indicators suggest stock market instability. Stock market stock market analysts warn of stock market correction. Stock market stock market stock market.",
    note: "keyword spam.",
  },
  {
    id: 18, category: "finance",
    title: "Federal Reserve Holds Interest Rates Amid Inflation Concerns",
    text: "The Federal Reserve kept its benchmark interest rate unchanged at 5.25% despite pressure from investors expecting cuts. Fed officials cited persistent inflation in services and shelter costs as reasons for caution.",
    note: "normal",
  },
  {
    id: 19, category: "finance",
    title: "Ethereum Layer-2 Networks Process 10x More Transactions Than Mainnet",
    text: "Ethereum scaling solutions including Arbitrum and Optimism collectively processed ten times more daily transactions than the Ethereum mainnet. Average fees on Layer-2 networks dropped below one cent per transaction.",
    note: "normal",
  },

  // ── CLIMATE / ENERGY ─────────────────────────────────────────────────────
  {
    id: 20, category: "climate",
    title: "Solar Panel Efficiency Record Broken at 47.6%",
    text: "A research team achieved 47.6% solar panel efficiency using a multi-junction concentrator design. Commercial panels currently average 22-24% efficiency, making this result a major milestone for photovoltaic technology.",
    note: "normal",
  },
  {
    id: 21, category: "climate",
    title: "Photovoltaic Arrays Power Entire Cities During Summer Peak Demand",
    text: "Regions with high photovoltaic installation density met 100% of electricity demand from solar generation on multiple summer days. Grid-scale battery storage allowed surplus energy to be discharged after sunset.",
    note: "synonym — 'photovoltaic' = solar panel. Semantic finds it, BM25 misses it on 'solar energy' query.",
  },
  {
    id: 22, category: "climate",
    title: "Climate Climate Climate Change Threatens Climate Climate Ecosystems",
    text: "Climate change climate impact on climate systems. Climate climate warming affects climate biodiversity. Climate change climate change climate solutions needed for climate crisis climate emergency.",
    note: "keyword spam.",
  },
  {
    id: 23, category: "climate",
    title: "Offshore Wind Capacity Doubles as Turbine Technology Improves",
    text: "Global offshore wind capacity doubled in three years as turbine blade length increased to over 100 meters. Floating offshore wind platforms now allow installation in deep water far from coastlines.",
    note: "normal",
  },
  {
    id: 24, category: "climate",
    title: "Arctic Sea Ice Reaches Lowest Recorded Summer Extent",
    text: "Arctic sea ice extent in September reached its lowest recorded level since satellite monitoring began in 1979. Scientists attribute the decline to accelerating warming at the poles, which is occurring three times faster than the global average.",
    note: "normal",
  },
];

export const TOPICS = [
  { key: "ai",      label: "AI",      query: "machine learning neural network artificial intelligence" },
  { key: "space",   label: "Space",   query: "rocket Mars space exploration astronaut" },
  { key: "health",  label: "Health",  query: "heart attack drug clinical trial treatment" },
  { key: "finance", label: "Finance", query: "Bitcoin cryptocurrency stock market crypto" },
  { key: "climate", label: "Climate", query: "solar energy climate change renewable wind" },
];
