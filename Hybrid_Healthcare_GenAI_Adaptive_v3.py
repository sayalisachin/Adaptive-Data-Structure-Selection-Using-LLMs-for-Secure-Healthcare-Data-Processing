# ==================================================
# HYBRID HEALTHCARE DIAGNOSIS SYSTEM WITH GEN AI
# ADAPTIVE DATA STRUCTURES VERSION
# ==================================================
#
# ADAPTIVE DATA STRUCTURES USED:
#
#   1. AdaptiveSymptomGraph   — weighted graph whose edges and weights
#      UPDATE dynamically when new patient cases are fed in.
#
#   2. AdaptiveDiseaseProfile — disease mean-vectors that RECALCULATE
#      INCREMENTALLY (Welford online algorithm) as new data arrives,
#      without re-scanning the whole dataset.
#
#   3. SymptomTrie            — prefix trie that GROWS as new symptoms
#      are discovered (from AI suggestions or new data), enabling fast
#      autocomplete on an ever-expanding symptom vocabulary.
#
#   4. LRUSymptomCache        — a fixed-capacity LRU cache for expansion
#      results that EVICTS the least-recently-used entry automatically,
#      keeping memory bounded while adapting to recent query patterns.
#
#   5. LLM-driven structure selector — the choose_data_structure()
#      function is now ACTUALLY USED to switch the graph's internal
#      representation between adjacency-list and adjacency-matrix
#      depending on density, rather than just printing text.
# ==================================================

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from itertools import combinations
from pathlib import Path
from openai import OpenAI

# --------------------------------------------------
# OPENAI
# --------------------------------------------------

client = OpenAI(
    api_key="INSERT_OPENAI_KEY"

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():
    path = Path(r"C:\Users\Sayali\Downloads\tuned_hospital_dataset.csv")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

df = load_dataset()

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------

def preprocess(df):
    symptom_col = [c for c in df.columns if "symptom" in c.lower()][0]
    symptom_lists = df[symptom_col].astype(str).apply(
        lambda x: [s.strip() for s in x.split(",")]
    )
    all_symptoms = sorted(
        set(s for row in symptom_lists for s in row)
    )
    matrix = []
    for row in symptom_lists:
        row_set = set(row)
        matrix.append([s in row_set for s in all_symptoms])
    return pd.DataFrame(matrix, columns=all_symptoms)

symptom_df = preprocess(df)

# --------------------------------------------------
# APRIORI
# --------------------------------------------------

def apriori(df, min_support=0.1, max_len=3):
    items = list(df.columns)
    total = len(df)
    freq = []
    for size in range(1, max_len + 1):
        for combo in combinations(items, size):
            support = df[list(combo)].all(axis=1).sum() / total
            if support >= min_support:
                freq.append({"itemsets": frozenset(combo), "support": support})
    return pd.DataFrame(freq)

freq = apriori(symptom_df)

# --------------------------------------------------
# RULES
# --------------------------------------------------

def generate_rules(freq_df, min_conf=0.5):
    support_map = {row["itemsets"]: row["support"] for _, row in freq_df.iterrows()}
    rules = []
    for itemset, support in support_map.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                support_A = support_map.get(antecedent, 0)
                support_B = support_map.get(consequent, 0)
                if support_A == 0 or support_B == 0:
                    continue
                confidence = support / support_A
                lift = confidence / support_B
                conviction = (
                    (1 - support_B) / (1 - confidence)
                    if confidence != 1
                    else float("inf")
                )
                cosine = support / math.sqrt(support_A * support_B)
                if confidence >= min_conf:
                    rules.append({
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "conviction": conviction,
                        "cosine": cosine,
                    })
    return pd.DataFrame(rules).sort_values(by="confidence", ascending=False)

rules = generate_rules(freq)

# --------------------------------------------------
# GEN AI HELPERS
# --------------------------------------------------

def llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {e}"


def explain_rule(rule):
    prompt = f"""
    Explain this healthcare association rule in plain language.
    Antecedent: {list(rule['antecedents'])}
    Consequent: {list(rule['consequents'])}
    Confidence: {rule['confidence']:.2f}
    Lift: {rule['lift']:.2f}
    Conviction: {rule['conviction']:.2f}
    Cosine: {rule['cosine']:.2f}
    """
    return llm(prompt)


def suggest_symptoms(symptoms):
    prompt = f"""
    Symptoms: {symptoms}
    Suggest 5 medically related symptoms.
    Return comma-separated values only, no extra text.
    """
    text = llm(prompt)
    return [x.strip() for x in text.split(",") if x.strip()]


def explain_prediction(symptoms, expanded, disease, score):
    prompt = f"""
    Patient Symptoms: {symptoms}
    Expanded Symptoms (after graph traversal): {expanded}
    Predicted Disease: {disease}
    Confidence Score: {score:.2f}
    Explain why this prediction was made in 3-4 sentences.
    """
    return llm(prompt)


# ==================================================
# ADAPTIVE DATA STRUCTURE 1: LLM-Driven Structure Selector
# ==================================================
# Previously choose_data_structure() only printed text — it was never
# used to make any real decision.  Now it returns a string "list" or
# "matrix" that the AdaptiveSymptomGraph actually uses to pick its
# internal representation.

def choose_data_structure(nodes: int, edges: int) -> str:
    """
    Ask the LLM whether to use adjacency-list or adjacency-matrix
    for the current graph density, then parse and RETURN that choice
    so the graph can adapt its internal representation accordingly.
    """
    density = edges / max(nodes * (nodes - 1), 1)
    prompt = f"""
    A symptom graph has:
      Nodes  = {nodes}
      Edges  = {edges}
      Density = {density:.4f}

    Should we use an adjacency LIST or adjacency MATRIX internally?
    Reply with exactly one word: LIST or MATRIX, then one sentence why.
    """
    response = llm(prompt)
    print("\n[LLM STRUCTURE DECISION]\n", response)
    # Parse the first word from the LLM response
    first_word = response.strip().split()[0].upper()
    return "matrix" if "MATRIX" in first_word else "list"


# ==================================================
# ADAPTIVE DATA STRUCTURE 2: AdaptiveSymptomGraph
# ==================================================
# What changed vs the original:
#   • Chooses adjacency-list OR adjacency-matrix based on LLM advice.
#   • add_case() lets you feed in a NEW patient record AFTER initial
#     build — edges and weights update immediately without rebuilding.
#   • remove_symptom() prunes a symptom node from the live graph,
#     so stale or erroneous symptoms can be removed at runtime.

class AdaptiveSymptomGraph:
    """
    A weighted undirected symptom graph whose internal representation
    (list vs matrix) is chosen by the LLM and whose edges adapt
    dynamically as new patient cases arrive.
    """

    def __init__(self, all_symptoms: list, structure: str = "list"):
        self.symptoms = all_symptoms
        self.structure = structure
        self.sym_index = {s: i for i, s in enumerate(all_symptoms)}
        n = len(all_symptoms)

        if structure == "matrix":
            # Adjacency matrix — O(1) edge lookup, high memory
            self._matrix = np.zeros((n, n), dtype=float)
            print(f"[Graph] Using ADJACENCY MATRIX ({n}×{n})")
        else:
            # Adjacency list — memory-efficient for sparse graphs
            self._adj = defaultdict(dict)
            print(f"[Graph] Using ADJACENCY LIST")

    # ── internal setters / getters ──────────────────────────────────

    def _set_edge(self, a: str, b: str, w: float):
        if a not in self.sym_index or b not in self.sym_index:
            return
        if self.structure == "matrix":
            i, j = self.sym_index[a], self.sym_index[b]
            self._matrix[i][j] = max(self._matrix[i][j], w)
            self._matrix[j][i] = max(self._matrix[j][i], w)
        else:
            self._adj[a][b] = max(self._adj[a].get(b, 0), w)
            self._adj[b][a] = max(self._adj[b].get(a, 0), w)

    def _get_neighbors(self, node: str) -> dict:
        if self.structure == "matrix":
            if node not in self.sym_index:
                return {}
            i = self.sym_index[node]
            return {
                self.symptoms[j]: self._matrix[i][j]
                for j in range(len(self.symptoms))
                if self._matrix[i][j] > 0
            }
        else:
            return dict(self._adj.get(node, {}))

    # ── public API ───────────────────────────────────────────────────

    def build_from_rules(self, rules_df: pd.DataFrame):
        """Initial build from association rules."""
        for _, r in rules_df.iterrows():
            for a in r["antecedents"]:
                for b in r["consequents"]:
                    self._set_edge(a, b, r["confidence"])
        print(f"[Graph] Built from {len(rules_df)} rules.")

    def add_case(self, symptom_list: list, confidence: float = 0.6):
        """
        ADAPTIVE: Feed in a new patient case.
        All co-occurring symptoms get their edge weight updated.
        This is the key adaptive behaviour — the graph evolves with data.
        """
        for a, b in combinations(symptom_list, 2):
            self._set_edge(a, b, confidence)

    def add_symptom_node(self, symptom: str):
        """
        ADAPTIVE: Register a brand-new symptom (e.g. discovered via AI)
        so it can participate in future edge additions.
        """
        if symptom not in self.sym_index:
            idx = len(self.symptoms)
            self.symptoms.append(symptom)
            self.sym_index[symptom] = idx
            if self.structure == "matrix":
                # Grow the matrix by one row and column
                n = len(self.symptoms)
                new_matrix = np.zeros((n, n), dtype=float)
                new_matrix[: n - 1, : n - 1] = self._matrix
                self._matrix = new_matrix
            print(f"[Graph] New symptom node added: '{symptom}'")

    def remove_symptom(self, symptom: str):
        """
        ADAPTIVE: Remove a symptom node (e.g. data quality correction).
        """
        if symptom not in self.sym_index:
            return
        if self.structure == "list":
            self._adj.pop(symptom, None)
            for neighbors in self._adj.values():
                neighbors.pop(symptom, None)
        else:
            i = self.sym_index[symptom]
            self._matrix[i, :] = 0
            self._matrix[:, i] = 0
        del self.sym_index[symptom]
        self.symptoms.remove(symptom)
        print(f"[Graph] Symptom node removed: '{symptom}'")

    def neighbors(self, node: str) -> dict:
        return self._get_neighbors(node)


# ==================================================
# ADAPTIVE DATA STRUCTURE 3: AdaptiveDiseaseProfile
# ==================================================
# Original: profiles were computed once as a static mean and never
# updated.  Now they use Welford's online algorithm to update the
# running mean and variance incrementally — O(1) per new case —
# so the profile adapts every time a new patient record is added.

class AdaptiveDiseaseProfile:
    """
    Maintains an online (incremental) mean symptom vector per disease
    using Welford's algorithm.  No need to store all past records.
    Adapts to new cases in O(1) time and O(features) space.
    """

    def __init__(self, symptoms: list):
        self.symptoms = symptoms
        self.n: dict[str, int] = defaultdict(int)          # case count
        self.mean: dict[str, np.ndarray] = {}              # running mean
        self.M2: dict[str, np.ndarray] = {}                # for variance

    def update(self, disease: str, symptom_vector: np.ndarray):
        """
        ADAPTIVE: Welford online update — called once per new patient.
        Updates the mean without re-scanning historical data.
        """
        self.n[disease] += 1
        if disease not in self.mean:
            self.mean[disease] = np.zeros(len(self.symptoms), dtype=float)
            self.M2[disease] = np.zeros(len(self.symptoms), dtype=float)
        delta = symptom_vector - self.mean[disease]
        self.mean[disease] += delta / self.n[disease]
        delta2 = symptom_vector - self.mean[disease]
        self.M2[disease] += delta * delta2

    def get_profile(self, disease: str) -> np.ndarray:
        return self.mean.get(disease, np.zeros(len(self.symptoms)))

    def variance(self, disease: str) -> np.ndarray:
        if self.n.get(disease, 0) < 2:
            return np.zeros(len(self.symptoms))
        return self.M2[disease] / (self.n[disease] - 1)

    def build_from_dataframe(self, df: pd.DataFrame, symptom_df: pd.DataFrame):
        """Seed the profiles from the initial dataset."""
        for _, row in df.iterrows():
            disease = row["Disease"]
            vec = symptom_df.loc[row.name].values.astype(float)
            self.update(disease, vec)
        print(f"[Profiles] Built for {len(self.mean)} diseases.")

    def add_new_case(self, disease: str, symptom_list: list):
        """
        ADAPTIVE: Called when a new patient case comes in.
        Updates the disease profile on-the-fly.
        """
        vec = np.array(
            [1.0 if s in symptom_list else 0.0 for s in self.symptoms]
        )
        self.update(disease, vec)
        print(f"[Profiles] Updated profile for '{disease}' "
              f"(n={self.n[disease]})")


# ==================================================
# ADAPTIVE DATA STRUCTURE 4: SymptomTrie
# ==================================================
# Brand new structure — not present in original.
# A prefix trie that grows dynamically as new symptoms are added
# (from AI suggestions, new dataset rows, or user corrections).
# Enables O(prefix_length) autocomplete on the live symptom vocabulary.

class TrieNode:
    __slots__ = ("children", "is_end", "word")
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.word: str = ""

class SymptomTrie:
    """
    A trie over symptom strings that GROWS at runtime.
    Used for autocomplete and for tracking the live symptom vocabulary.
    """

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def insert(self, symptom: str):
        """ADAPTIVE: Insert a new symptom into the trie."""
        node = self.root
        for ch in symptom.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_end:
            node.is_end = True
            node.word = symptom
            self._size += 1

    def search(self, prefix: str) -> list[str]:
        """Return all known symptoms that start with prefix."""
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]
        results = []
        self._dfs(node, results)
        return results

    def _dfs(self, node: TrieNode, results: list):
        if node.is_end:
            results.append(node.word)
        for child in node.children.values():
            self._dfs(child, results)

    def __len__(self):
        return self._size


# ==================================================
# ADAPTIVE DATA STRUCTURE 5: LRUSymptomCache
# ==================================================
# Original code re-expanded the same symptom sets every query.
# This LRU cache stores recent BFS expansion results.
# ADAPTIVE: automatically evicts the least-recently-used entry
# when capacity is hit, so it tracks recent query patterns without
# growing unboundedly.

class LRUSymptomCache:
    """
    Fixed-capacity LRU cache for BFS symptom expansion results.
    Adapts to recent usage — hot queries stay cached, cold ones evict.
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._cache: OrderedDict[str, list] = OrderedDict()

    def _key(self, symptoms: list) -> str:
        return ",".join(sorted(symptoms))

    def get(self, symptoms: list) -> list | None:
        k = self._key(symptoms)
        if k not in self._cache:
            return None
        self._cache.move_to_end(k)          # mark as recently used
        return self._cache[k]

    def put(self, symptoms: list, expanded: list):
        k = self._key(symptoms)
        if k in self._cache:
            self._cache.move_to_end(k)
        else:
            if len(self._cache) >= self.capacity:
                evicted = self._cache.popitem(last=False)
                print(f"[LRU Cache] Evicted: '{evicted[0][:40]}...'")
        self._cache[k] = expanded

    def __len__(self):
        return len(self._cache)


# --------------------------------------------------
# BUILD ADAPTIVE STRUCTURES
# --------------------------------------------------

all_symptoms = list(symptom_df.columns)

# 1. Ask LLM to decide list vs matrix, then use the decision
chosen_structure = choose_data_structure(
    nodes=len(all_symptoms),
    edges=len(rules)
)

# 2. Build adaptive graph with the chosen internal representation
adaptive_graph = AdaptiveSymptomGraph(all_symptoms, structure=chosen_structure)
adaptive_graph.build_from_rules(rules)

# 3. Build adaptive disease profiles (Welford incremental)
adaptive_profiles = AdaptiveDiseaseProfile(all_symptoms)
adaptive_profiles.build_from_dataframe(df, symptom_df)

# 4. Build symptom trie from known symptoms
symptom_trie = SymptomTrie()
for s in all_symptoms:
    symptom_trie.insert(s)
print(f"[Trie] Loaded {len(symptom_trie)} symptoms.")

# 5. LRU cache for BFS expansions
expansion_cache = LRUSymptomCache(capacity=128)

# --------------------------------------------------
# BFS EXPANSION (now cache-aware)
# --------------------------------------------------

def expand(graph: AdaptiveSymptomGraph, symptoms: list, depth: int = 2) -> list:
    # Check cache first
    cached = expansion_cache.get(symptoms)
    if cached is not None:
        print("[LRU Cache] Hit!")
        return cached

    visited = set(symptoms)
    queue = [(s, 0) for s in symptoms]
    while queue:
        node, d = queue.pop(0)
        if d < depth:
            for neigh in graph.neighbors(node):
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, d + 1))

    result = list(visited)
    expansion_cache.put(symptoms, result)       # store in LRU cache
    return result

# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_manual(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if (na == 0 or nb == 0) else dot / (na * nb)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_disease(symptoms: list):
    # 1. AI suggests related symptoms
    ai_symptoms = suggest_symptoms(symptoms)

    # 2. Register AI-suggested symptoms in trie and graph (ADAPTIVE)
    for s in ai_symptoms:
        symptom_trie.insert(s)            # trie grows
        adaptive_graph.add_symptom_node(s)  # graph grows if new

    merged = list(set(symptoms + ai_symptoms))

    # 3. BFS expansion (LRU-cached)
    expanded = expand(adaptive_graph, merged)

    # 4. Build query vector over ALL known symptoms (including new ones)
    all_syms = adaptive_graph.symptoms
    vec = np.array([1.0 if s in expanded else 0.0 for s in all_syms])

    # 5. Score against adaptive profiles
    scores = {}
    for disease in adaptive_profiles.mean:
        profile_vec = adaptive_profiles.get_profile(disease)
        # Align lengths if new symptoms were added
        if len(vec) != len(profile_vec):
            min_len = min(len(vec), len(profile_vec))
            scores[disease] = cosine_manual(vec[:min_len], profile_vec[:min_len])
        else:
            scores[disease] = cosine_manual(vec, profile_vec)

    return (
        sorted(scores.items(), key=lambda x: x[1], reverse=True),
        expanded,
        ai_symptoms,
    )

# --------------------------------------------------
# ADD NEW PATIENT CASE (demonstrates live adaptation)
# --------------------------------------------------

def add_new_patient(disease: str, symptom_list: list):
    """
    Feed a confirmed new patient case into the system.
    Both the graph and the disease profiles adapt immediately.
    No retraining required.
    """
    print(f"\n[Adaptation] Adding new case: {disease} | {symptom_list}")
    adaptive_graph.add_case(symptom_list, confidence=0.65)
    adaptive_profiles.add_new_case(disease, symptom_list)
    for s in symptom_list:
        symptom_trie.insert(s)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

def confidence_level(score: float) -> str:
    if score >= 0.85:
        return "VERY HIGH"
    elif score >= 0.70:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    return "LOW"


def plot_confidence(preds):
    diseases = [x[0] for x in preds[:5]]
    scores = [x[1] * 100 for x in preds[:5]]
    plt.figure(figsize=(8, 5))
    plt.bar(diseases, scores)
    plt.title("Disease Prediction Confidence")
    plt.ylabel("Confidence (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# DIAGNOSE
# --------------------------------------------------

def diagnose(symptoms: list):
    preds, expanded, ai_symptoms = predict_disease(symptoms)

    print("\n" + "=" * 70)
    print("PATIENT DIAGNOSIS REPORT")
    print("=" * 70)

    print("\nINPUT SYMPTOMS")
    print(symptoms)

    print("\nAI SUGGESTED SYMPTOMS")
    print(ai_symptoms)

    print("\nEXPANDED SYMPTOMS (via adaptive graph BFS)")
    print(expanded)

    print(f"\n[Trie] Symptom vocabulary size: {len(symptom_trie)}")
    print(f"[LRU Cache] Cached expansions: {len(expansion_cache)}")

    print("\nTOP PREDICTIONS")
    for i, (d, s) in enumerate(preds[:3], 1):
        print(f"  {i}. {d} : {s:.2%} ({confidence_level(s)})")

    print("\nCONFIDENCE BARS")
    for d, s in preds[:3]:
        print(f"  {d:<25} {s:.2%} " + "█" * int(s * 30))

    best_disease, best_score = preds[0]

    print("\nFINAL RECOMMENDATION")
    print(f"  {best_disease}  (Confidence: {best_score:.2%})")

    print("\nGEN AI EXPLANATION\n")
    print(explain_prediction(symptoms, expanded, best_disease, best_score))

    plot_confidence(preds)

    # --- Demo: adapt the system with this diagnosis (if confirmed) ---
    # In a real system, a doctor would confirm the disease first.
    # Uncomment the next line to feed confirmed cases back in:
    # add_new_patient(best_disease, symptoms)

# --------------------------------------------------
# RULE DEMO
# --------------------------------------------------

if len(rules):
    print("\nTOP RULE EXPLANATION\n")
    print(explain_rule(rules.iloc[0]))

# --------------------------------------------------
# SYMPTOM AUTOCOMPLETE DEMO (Trie)
# --------------------------------------------------

print("\n[Trie Autocomplete Demo]")
prefix = "fe"   # e.g. "fe" → fever, feeling nauseous, etc.
matches = symptom_trie.search(prefix)
print(f"  Symptoms starting with '{prefix}': {matches[:10]}")

# --------------------------------------------------
# USER INPUT LOOP
# --------------------------------------------------

print("\nAvailable Symptoms (Trie autocomplete):\n")
print(", ".join(symptom_trie.search("")[:50]), "...")   # first 50

while True:
    user_input = input(
        "\nEnter symptoms separated by commas (or type exit): "
    )

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Trie autocomplete: show completions for each entered term
    symptoms = [s.strip() for s in user_input.split(",") if s.strip()]
    for s in symptoms:
        completions = symptom_trie.search(s[:4])
        if completions and s not in completions:
            print(f"  [Trie] Did you mean: {completions[:5]}")

    diagnose(symptoms)
