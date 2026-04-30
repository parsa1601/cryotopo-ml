import requests
from typing import Dict, List

BASE_URL = "https://data.rcsb.org/rest/v1/core/entry"


def get_protein_lengths(pdb_id: str) -> Dict:
    """
    Returns key length metrics for a PDB entry:
    - full sequence length
    - modeled (observed) residues
    - missing residues
    """
    url = f"{BASE_URL}/{pdb_id}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] {pdb_id}: {e}")
        return None

    info = data.get("rcsb_entry_info", {})

    full_len = info.get("deposited_polymer_monomer_count")
    modeled_len = info.get("deposited_modeled_polymer_monomer_count")
    missing_len = info.get("deposited_unmodeled_polymer_monomer_count")

    return {
        "pdb_id": pdb_id,
        "full_length": full_len,
        "modeled_length": modeled_len,
        "missing_length": missing_len,
    }


def batch_lengths(pdb_ids: List[str]) -> List[Dict]:
    results = []
    for pdb_id in pdb_ids:
        res = get_protein_lengths(pdb_id)
        if res:
            results.append(res)
    return results


# Example usage
pdb_list = [
    "1A7D",
    "1BJ7",
    "1BZ4",
    "1FLP",
    "1HG5",
    "1HBE",
    "1HZ4",
    "1ICX",
    "1LWB",
    "1NG6",
    "1OZ9",
    "1P5X",
    "1XQO",
    "1YD0",
    "1Z1L",
    "2OVJ",
    "2XB5",
    "2XVV",
    "2Y4Z",
    "3ACW",
    "3C91",
    "3FIN",
    "3HJL",
    "3IEE",
    "3IXV",
    "3LTJ",
    "3ODS",
    "4CHV",
    "4OXW",
    "4R9A",
    "4UE4",
    "4YOK",
    "5I1M",
    "5KBU",
    "5M50",
    "5O8O",
    "5UZB",
    "6EM3",
    "6F36",
    "6UXW",
]

results = batch_lengths(pdb_list)

for r in results:
    print(
        f"{r['pdb_id']}: "
        f"full={r['full_length']}, "
        f"modeled={r['modeled_length']}, "
        f"missing={r['missing_length']}"
    )
