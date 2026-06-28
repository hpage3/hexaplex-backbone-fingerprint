#!/usr/bin/env python3
import random, string, time, sys
import requests

RCSB_CORE = "https://data.rcsb.org/rest/v1/core/entry/{}"

def random_pdb_id():
    # PDB IDs are 4 chars: [0-9A-Za-z] but commonly start with a digit
    # We'll bias first char to digits for faster hits, but allow letters too.
    digits = string.digits
    letters = string.ascii_lowercase
    first = random.choice(digits + letters)
    rest = "".join(random.choice(digits + letters) for _ in range(3))
    return (first + rest).upper()

def is_protein_entry(pdbid, session, timeout=10):
    try:
        r = session.get(RCSB_CORE.format(pdbid), timeout=timeout)
        if r.status_code != 200:
            return False
        j = r.json()
        # Quick, permissive checks:
        info = j.get("rcsb_entry_info", {}) or {}
        if info.get("polymer_entity_count_protein", 0) <= 0:
            return False  # no proteins
        # Optional: screen out very low-res X-ray (>3.5Å); keep EM too
        # If present, resolution_combined is a list of floats
        res = info.get("resolution_combined")
        if isinstance(res, list) and len(res) > 0:
            try:
                best = min(float(x) for x in res if x is not None)
                if best > 3.5:
                    # still acceptable for geometry, but skip if you want stricter set:
                    # return False
                    pass
            except Exception:
                pass
        return True
    except Exception:
        return False

def main(n=250, seed=42, out_path="ids.txt"):
    random.seed(seed)
    found = []
    seen = set()
    with requests.Session() as s:
        while len(found) < n:
            pid = random_pdb_id()
            if pid in seen:
                continue
            seen.add(pid)
            if is_protein_entry(pid, s):
                found.append(pid)
                if len(found) % 25 == 0:
                    print(f"[ok] {len(found)} / {n}: last={pid}")
            # be a good citizen
            if len(seen) % 50 == 0:
                time.sleep(0.2)

    with open(out_path, "w") as fh:
        for pid in sorted(found):
            fh.write(pid.upper() + "\n")
    print(f"[done] wrote {len(found)} IDs to {out_path}")

if __name__ == "__main__":
    # Usage: python make_ids.py  (defaults to 250)
    n = 250
    if len(sys.argv) >= 2:
        try:
            n = int(sys.argv[1])
        except Exception:
            pass
    main(n=n)
