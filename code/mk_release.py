r"""Supplementary data for the paper: one OFF file per member in canonical form + a CSV summary.
    python3 mk_release.py [nproc]   -> ../release/{off/<class>/<id>.off, members.csv, README.md}  (resumable)
classes: proper / trivial / degenerate
"""
import json, os, sys, csv
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.expanduser("~/nmj"))
from robust import canonical_robust
from mapcheck import orient

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(HERE, "..", "release")
SQ = os.path.expanduser("~/nmj/sesqpaper/ext2")
fam = {x["id"]: x for x in json.load(open(SQ + "/family_ids.json"))}
st = {r["id"]: r for r in json.load(open(SQ + "/stats.json"))}
deg = set(json.load(open(SQ + "/degenerate_ids.json")))


def klass(i):
    return "degenerate" if i in deg else ("trivial" if st[i]["V4"] == 0 else "proper")


def path(i):
    return os.path.join(REL, "off", klass(i), i + ".off")


def work(i):
    x = fam[i]
    F = orient([list(f) for f in x["child"]])
    res = canonical_robust(F, x["cnv"])
    if res is None:
        return i, None
    V, d = res
    V = np.asarray(V, float)
    r = st[i]
    E = sum(len(f) for f in F) // 2
    lines = ["OFF",
             "# %s  sesquitruncation sigma(P), %s member, canonical (midsphere, barycentred) realization" % (i, klass(i)),
             "# group %s  edge ratio %.6f  big faces %s" % (r["group"], d["ratio"], json.dumps(r["big"], sort_keys=True)),
             "# parent P*: %d vertices, faces %s" % (x["pnv"], json.dumps(x["parent"])),
             "%d %d %d" % (len(V), len(F), E)]
    lines += ["%.12f %.12f %.12f" % tuple(p) for p in V]
    lines += ["%d %s" % (len(f), " ".join(map(str, f))) for f in F]
    os.makedirs(os.path.dirname(path(i)), exist_ok=True)
    open(path(i), "w").write("\n".join(lines) + "\n")
    return i, float(d["ratio"])


if __name__ == "__main__":
    ids = sorted(fam)
    todo = [i for i in ids if not os.path.exists(path(i))]
    print("todo", len(todo), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(work, todo, chunksize=8)):
            if r is None:
                print("FAILED", i, flush=True)
            if k % 500 == 0:
                print("..%d" % k, flush=True)
    with open(os.path.join(REL, "members.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "class", "V", "F", "faces_of_P", "big_faces", "group", "group_order", "canonical_ratio"])
        for i in ids:
            r = st[i]
            w.writerow([i, klass(i), r["V"], r["F"], r["inF"],
                        " ".join("%sx%s" % (v, k) for k, v in sorted(r["big"].items(), key=lambda kv: int(kv[0]))),
                        r["group"].replace("{", "").replace("}", ""), r["order"], "%.6f" % r["canon"]])
    print("done")
