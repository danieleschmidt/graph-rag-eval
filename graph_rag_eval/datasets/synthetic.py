"""
SyntheticQADataset: generate Q&A pairs over a toy knowledge graph.

No downloads required. Generates a fictional company/person KG with
50 Q&A pairs covering single-hop and multi-hop questions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from graph_rag_eval.datasets.loader import DatasetLoader, QAPair
from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph
from graph_rag_eval.retrievers.base import Document


# ---------------------------------------------------------------------------
# Fictional entity data
# ---------------------------------------------------------------------------

COMPANIES = [
    {"id": "nexus_tech", "name": "Nexus Tech", "sector": "technology", "country": "USA"},
    {"id": "aurora_bio", "name": "Aurora Biotech", "sector": "biotechnology", "country": "Germany"},
    {"id": "peak_finance", "name": "Peak Finance", "sector": "finance", "country": "UK"},
    {"id": "solar_energy", "name": "Solar Energies", "sector": "energy", "country": "Spain"},
    {"id": "quantum_ai", "name": "Quantum AI", "sector": "technology", "country": "USA"},
    {"id": "blue_logistics", "name": "Blue Logistics", "sector": "logistics", "country": "Netherlands"},
    {"id": "green_agri", "name": "Green Agri", "sector": "agriculture", "country": "Brazil"},
    {"id": "nova_pharma", "name": "Nova Pharma", "sector": "biotechnology", "country": "Switzerland"},
    {"id": "apex_retail", "name": "Apex Retail", "sector": "retail", "country": "USA"},
    {"id": "delta_media", "name": "Delta Media", "sector": "media", "country": "France"},
    {"id": "iron_mining", "name": "Iron Mining Co", "sector": "mining", "country": "Australia"},
    {"id": "swift_cloud", "name": "Swift Cloud", "sector": "technology", "country": "Canada"},
    {"id": "ocean_shipping", "name": "Ocean Shipping Ltd", "sector": "logistics", "country": "Singapore"},
    {"id": "cyber_defense", "name": "Cyber Defense Corp", "sector": "technology", "country": "Israel"},
    {"id": "harvest_foods", "name": "Harvest Foods", "sector": "agriculture", "country": "USA"},
    {"id": "metro_bank", "name": "Metro Bank", "sector": "finance", "country": "UK"},
    {"id": "vortex_games", "name": "Vortex Games", "sector": "technology", "country": "Japan"},
    {"id": "amber_health", "name": "Amber Health", "sector": "biotechnology", "country": "Denmark"},
    {"id": "ridge_oil", "name": "Ridge Oil", "sector": "energy", "country": "Saudi Arabia"},
    {"id": "pixel_studios", "name": "Pixel Studios", "sector": "media", "country": "USA"},
]

PERSONS = [
    {"id": "alice_chen", "name": "Alice Chen", "role": "CEO", "employer": "nexus_tech"},
    {"id": "bob_mueller", "name": "Bob Mueller", "role": "CTO", "employer": "aurora_bio"},
    {"id": "carol_smith", "name": "Carol Smith", "role": "CFO", "employer": "peak_finance"},
    {"id": "david_torres", "name": "David Torres", "role": "Engineer", "employer": "quantum_ai"},
    {"id": "eva_lin", "name": "Eva Lin", "role": "CEO", "employer": "swift_cloud"},
    {"id": "frank_jones", "name": "Frank Jones", "role": "Director", "employer": "nexus_tech"},
    {"id": "grace_kim", "name": "Grace Kim", "role": "Researcher", "employer": "nova_pharma"},
    {"id": "henry_patel", "name": "Henry Patel", "role": "CFO", "employer": "apex_retail"},
    {"id": "iris_berg", "name": "Iris Berg", "role": "CEO", "employer": "aurora_bio"},
    {"id": "james_white", "name": "James White", "role": "Engineer", "employer": "cyber_defense"},
    {"id": "karen_nguyen", "name": "Karen Nguyen", "role": "Director", "employer": "delta_media"},
    {"id": "leo_russo", "name": "Leo Russo", "role": "CEO", "employer": "ridge_oil"},
    {"id": "mia_santos", "name": "Mia Santos", "role": "Researcher", "employer": "amber_health"},
    {"id": "nick_ford", "name": "Nick Ford", "role": "CTO", "employer": "vortex_games"},
    {"id": "olivia_park", "name": "Olivia Park", "role": "CFO", "employer": "metro_bank"},
]

EDGES = [
    # works_at
    ("alice_chen", "nexus_tech", "works_at"),
    ("frank_jones", "nexus_tech", "works_at"),
    ("bob_mueller", "aurora_bio", "works_at"),
    ("iris_berg", "aurora_bio", "works_at"),
    ("carol_smith", "peak_finance", "works_at"),
    ("olivia_park", "metro_bank", "works_at"),
    ("david_torres", "quantum_ai", "works_at"),
    ("eva_lin", "swift_cloud", "works_at"),
    ("grace_kim", "nova_pharma", "works_at"),
    ("henry_patel", "apex_retail", "works_at"),
    ("james_white", "cyber_defense", "works_at"),
    ("karen_nguyen", "delta_media", "works_at"),
    ("leo_russo", "ridge_oil", "works_at"),
    ("mia_santos", "amber_health", "works_at"),
    ("nick_ford", "vortex_games", "works_at"),
    # acquired
    ("nexus_tech", "swift_cloud", "acquired"),
    ("peak_finance", "metro_bank", "acquired"),
    ("quantum_ai", "cyber_defense", "acquired"),
    ("aurora_bio", "nova_pharma", "acquired"),
    # partnered_with
    ("nexus_tech", "quantum_ai", "partnered_with"),
    ("solar_energy", "green_agri", "partnered_with"),
    ("blue_logistics", "ocean_shipping", "partnered_with"),
    ("delta_media", "pixel_studios", "partnered_with"),
    # located_in
    ("nexus_tech", "usa_loc", "located_in"),
    ("quantum_ai", "usa_loc", "located_in"),
    ("apex_retail", "usa_loc", "located_in"),
    ("aurora_bio", "germany_loc", "located_in"),
    ("peak_finance", "uk_loc", "located_in"),
    ("metro_bank", "uk_loc", "located_in"),
    ("harvest_foods", "usa_loc", "located_in"),
]

# Virtual location nodes (lightweight)
LOCATIONS = [
    {"id": "usa_loc", "name": "USA", "type": "location"},
    {"id": "germany_loc", "name": "Germany", "type": "location"},
    {"id": "uk_loc", "name": "UK", "type": "location"},
]


def _make_company_description(c: dict) -> str:
    return (
        f"{c['name']} is a {c['sector']} company headquartered in {c['country']}."
    )


def _make_person_description(p: dict, employer_name: str) -> str:
    return (
        f"{p['name']} is a {p['role']} working at {employer_name}."
    )


class SyntheticQADataset(DatasetLoader):
    """
    Self-contained synthetic QA dataset based on a fictional company/person KG.

    Contains:
    - 20 company entities
    - 15 person entities
    - 30 directed edges
    - 50 Q&A pairs (mix of single-hop and multi-hop)

    Usage::

        dataset = SyntheticQADataset()
        kg = dataset.knowledge_graph
        pairs = dataset.load()
        documents = dataset.documents
    """

    def __init__(self) -> None:
        self._kg: Optional[KnowledgeGraph] = None
        self._documents: Optional[List[Document]] = None
        self._qa_pairs: Optional[List[QAPair]] = None
        self._employer_map: Dict[str, str] = {}  # person_id → company_name

    def _build(self) -> None:
        """Lazily build the KG, documents, and QA pairs."""
        if self._kg is not None:
            return

        kg = KnowledgeGraph()

        # Add company nodes
        for c in COMPANIES:
            desc = _make_company_description(c)
            kg.add_node(c["id"], name=c["name"], type="company",
                        sector=c["sector"], country=c["country"], description=desc)

        # Add location nodes
        for loc in LOCATIONS:
            kg.add_node(loc["id"], name=loc["name"], type="location",
                        description=f"{loc['name']} is a country/region.")

        # Build employer name map
        company_name_map = {c["id"]: c["name"] for c in COMPANIES}
        self._employer_map = {p["id"]: company_name_map.get(p["employer"], p["employer"])
                              for p in PERSONS}

        # Add person nodes
        for p in PERSONS:
            employer_name = company_name_map.get(p["employer"], p["employer"])
            desc = _make_person_description(p, employer_name)
            kg.add_node(p["id"], name=p["name"], type="person",
                        role=p["role"], employer=p["employer"], description=desc)

        # Add edges (bidirectional=False for directed semantics)
        for src, tgt, rel in EDGES:
            if kg.has_node(src) and kg.has_node(tgt):
                kg.add_edge(src, tgt, relation=rel, bidirectional=True)

        self._kg = kg

        # Build document list
        self._documents = [
            Document(id=node_id, content=attrs["description"], metadata=attrs)
            for node_id, attrs in kg.nodes.items()
            if "description" in attrs
        ]

        # Build QA pairs
        self._qa_pairs = self._generate_qa_pairs()

    def _generate_qa_pairs(self) -> List[QAPair]:
        pairs: List[QAPair] = []
        assert self._kg is not None

        # ── Single-hop: who works at <company>? ──────────────────────────────
        for c in COMPANIES[:8]:
            employees = [
                p["id"] for p in PERSONS if p["employer"] == c["id"]
            ]
            if not employees:
                continue
            employee_names = [
                p["name"] for p in PERSONS if p["employer"] == c["id"]
            ]
            pairs.append(QAPair(
                question=f"Who works at {c['name']}?",
                answer=", ".join(employee_names),
                relevant_ids=employees + [c["id"]],
                answer_entities=employees,
                answer_relations=[(e, c["id"], "works_at") for e in employees],
                metadata={"multihop": False, "type": "person_lookup"},
            ))

        # ── Single-hop: which companies are in <sector>? ─────────────────────
        for sector in ["technology", "biotechnology", "finance", "energy"]:
            sector_companies = [c["id"] for c in COMPANIES if c["sector"] == sector]
            sector_names = [c["name"] for c in COMPANIES if c["sector"] == sector]
            if sector_companies:
                pairs.append(QAPair(
                    question=f"Which companies are in the {sector} sector?",
                    answer=", ".join(sector_names),
                    relevant_ids=sector_companies,
                    answer_entities=sector_companies,
                    answer_relations=[],
                    metadata={"multihop": False, "type": "sector_lookup"},
                ))

        # ── Single-hop: what is <person>'s role? ─────────────────────────────
        for p in PERSONS[:8]:
            pairs.append(QAPair(
                question=f"What is {p['name']}'s role?",
                answer=p["role"],
                relevant_ids=[p["id"]],
                answer_entities=[p["id"]],
                answer_relations=[],
                metadata={"multihop": False, "type": "role_lookup"},
            ))

        # ── Single-hop: where is <company> headquartered? ────────────────────
        for c in COMPANIES[:6]:
            pairs.append(QAPair(
                question=f"Where is {c['name']} headquartered?",
                answer=c["country"],
                relevant_ids=[c["id"]],
                answer_entities=[c["id"]],
                answer_relations=[],
                metadata={"multihop": False, "type": "location_lookup"},
            ))

        # ── Multi-hop: who works at a company acquired by <acquirer>? ─────────
        acquisitions = [
            ("nexus_tech", "swift_cloud"),
            ("peak_finance", "metro_bank"),
            ("quantum_ai", "cyber_defense"),
            ("aurora_bio", "nova_pharma"),
        ]
        for acquirer_id, acquired_id in acquisitions:
            acquirer = next((c for c in COMPANIES if c["id"] == acquirer_id), None)
            acquired = next((c for c in COMPANIES if c["id"] == acquired_id), None)
            if not acquirer or not acquired:
                continue
            employees = [p["id"] for p in PERSONS if p["employer"] == acquired_id]
            employee_names = [p["name"] for p in PERSONS if p["employer"] == acquired_id]
            if employees:
                pairs.append(QAPair(
                    question=f"Who works at a company acquired by {acquirer['name']}?",
                    answer=", ".join(employee_names),
                    relevant_ids=[acquirer_id, acquired_id] + employees,
                    answer_entities=employees,
                    answer_relations=[
                        (acquirer_id, acquired_id, "acquired"),
                        *[(e, acquired_id, "works_at") for e in employees],
                    ],
                    metadata={"multihop": True, "type": "multi_hop_person"},
                ))

        # ── Multi-hop: who works at a company partnered with <partner>? ───────
        partnerships = [
            ("nexus_tech", "quantum_ai"),
            ("solar_energy", "green_agri"),
            ("delta_media", "pixel_studios"),
        ]
        for p1_id, p2_id in partnerships:
            p1 = next((c for c in COMPANIES if c["id"] == p1_id), None)
            p2 = next((c for c in COMPANIES if c["id"] == p2_id), None)
            if not p1 or not p2:
                continue
            employees = [p["id"] for p in PERSONS if p["employer"] == p2_id]
            employee_names = [p["name"] for p in PERSONS if p["employer"] == p2_id]
            if employees:
                pairs.append(QAPair(
                    question=f"Who works at a company partnered with {p1['name']}?",
                    answer=", ".join(employee_names),
                    relevant_ids=[p1_id, p2_id] + employees,
                    answer_entities=employees,
                    answer_relations=[
                        (p1_id, p2_id, "partnered_with"),
                        *[(e, p2_id, "works_at") for e in employees],
                    ],
                    metadata={"multihop": True, "type": "multi_hop_partner"},
                ))

        # ── Multi-hop: which companies share a country with <company>? ────────
        uk_companies = [c for c in COMPANIES if c["country"] == "UK"]
        if len(uk_companies) >= 2:
            pairs.append(QAPair(
                question="Which companies share their HQ country with Peak Finance?",
                answer="Metro Bank",
                relevant_ids=["peak_finance", "metro_bank"],
                answer_entities=["metro_bank"],
                answer_relations=[("peak_finance", "uk_loc", "located_in"),
                                  ("metro_bank", "uk_loc", "located_in")],
                metadata={"multihop": True, "type": "multi_hop_location"},
            ))

        usa_companies = [c for c in COMPANIES if c["country"] == "USA"]
        if len(usa_companies) >= 3:
            usa_ids = [c["id"] for c in usa_companies if c["id"] != "nexus_tech"]
            pairs.append(QAPair(
                question="Which other companies are headquartered in the same country as Nexus Tech?",
                answer=", ".join(c["name"] for c in COMPANIES
                                 if c["country"] == "USA" and c["id"] != "nexus_tech"),
                relevant_ids=["nexus_tech"] + usa_ids,
                answer_entities=usa_ids,
                answer_relations=[(cid, "usa_loc", "located_in") for cid in usa_ids],
                metadata={"multihop": True, "type": "multi_hop_location"},
            ))

        # Pad to 50 with extra single-hop questions about remaining companies
        while len(pairs) < 50:
            idx = len(pairs) % len(COMPANIES)
            c = COMPANIES[idx]
            pairs.append(QAPair(
                question=f"What sector does {c['name']} operate in?",
                answer=c["sector"],
                relevant_ids=[c["id"]],
                answer_entities=[c["id"]],
                answer_relations=[],
                metadata={"multihop": False, "type": "sector_attribute"},
            ))

        return pairs[:50]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """Return the underlying KnowledgeGraph."""
        self._build()
        return self._kg  # type: ignore[return-value]

    @property
    def documents(self) -> List[Document]:
        """Return all entity documents suitable for indexing."""
        self._build()
        return self._documents  # type: ignore[return-value]

    def load(self) -> List[QAPair]:
        """Return all 50 Q&A pairs."""
        self._build()
        return self._qa_pairs  # type: ignore[return-value]

    def multihop_pairs(self) -> List[QAPair]:
        """Return only multi-hop Q&A pairs."""
        return [p for p in self.load() if p.is_multihop()]

    def single_hop_pairs(self) -> List[QAPair]:
        """Return only single-hop Q&A pairs."""
        return [p for p in self.load() if not p.is_multihop()]
