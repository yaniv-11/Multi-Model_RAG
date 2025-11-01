import json, networkx as nx
from typing import List

GRAPH_FILE = "city_graph.gpickle"

def build_graph(triplets_path="triplets.json"):
    with open(triplets_path,"r",encoding="utf-8") as f:
        triplets = json.load(f)
    G = nx.DiGraph()
    for t in triplets:
        if isinstance(t, dict):
            s = t.get("subject"); r = t.get("relation"); o = t.get("object")
        else:
            s,r,o = t
        if s and o:
            G.add_node(s); G.add_node(o)
            G.add_edge(s,o,relation=r)
    nx.write_gpickle(G, GRAPH_FILE)
    return G

def load_graph():
    return nx.read_gpickle(GRAPH_FILE)

def query_graph_for_entities(entities: List[str], hops=1):
    G = load_graph()
    facts=[]
    for ent in entities:
        if ent in G:
            for nbr in G[ent]:
                facts.append({"fact": f"{ent} -[{G[ent][nbr].get('relation')}]→ {nbr}", "node": ent})
    return facts
