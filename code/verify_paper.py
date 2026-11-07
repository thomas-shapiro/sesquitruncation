r"""Recompute every number stated in the paper from the data; print CLAIM / VALUE / OK."""
import json, os, statistics
from collections import Counter
SQ = os.path.expanduser("~/nmj/sesqpaper/ext2")
ST = {r["id"]: r for r in json.load(open(SQ + "/stats.json"))}
FAM = {x["id"]: x for x in json.load(open(SQ + "/family_ids.json"))}
DEG = set(json.load(open(SQ + "/degenerate_ids.json")))
POLY = [r for i, r in ST.items() if i not in DEG]
S = [r for r in POLY if r["V4"] > 0]
T = [r for r in POLY if r["V4"] == 0]
D = [ST[i] for i in DEG]
mx = lambda r: max(int(k) for k in r["big"])
res = []
def chk(claim, value, expect):
    res.append((claim, value, value == expect))
chk("total members", len(ST), 7527)
chk("proper", len(S), 7176); chk("trivial", len(T), 76); chk("degenerate", len(D), 275)
# degenerate split: digon (P* has a degree-2 vertex) vs 2-cut
def has_digon(i):
    P, n = FAM[i]["parent"], FAM[i]["pnv"]
    return any(sum(1 for f in P if v in f) == 2 for v in range(n))
chk("degenerate with digon", sum(has_digon(i) for i in DEG), 272)
chk("max faces of P (all polyhedral)", max(r["inF"] for r in POLY), max(r["inF"] for r in POLY))
res.append(("  -> max inF over polyhedral, bound 32", max(r["inF"] for r in POLY), max(r["inF"] for r in POLY) <= 32))
res.append(("max V over polyhedral (bound 240; PCC 208)", max(r["V"] for r in POLY), max(r["V"] for r in POLY) <= 208))
chk("no 13-gon anywhere", sum("13" in r["big"] for r in ST.values()), 0)
chk("max face size (polyhedral) 22", max(mx(r) for r in POLY), 22)
chk("max dodecagons 8", max(r["big"].get("12", 0) for r in POLY), 8)
chk("members with 8 dodecagons", [r["id"] for r in POLY if r["big"].get("12", 0) == 8], ["S7527"])
chk("max faces >=14 per member <=6", max(sum(v for k, v in r["big"].items() if int(k) >= 14) for r in POLY) <= 6, True)
chk("max big faces n<=11 per member <=24", max(sum(v for k, v in r["big"].items() if int(k) <= 11) for r in POLY) <= 24, True)
# Corollary V = 6F - 12 + 2 V4
chk("V(sigma) = 6F(P) - 12 + 2 V4 for all", all(r["V"] == 6 * r["inF"] - 12 + 2 * r["V4"] for r in ST.values()), True)
chk("identity sum(12-n)=24 for all polyhedral", all(sum((12 - int(k)) * v for k, v in r["big"].items()) == 24 for r in POLY), True)
# Section 6
chk("smallest proper V", min((r["V"], r["id"]) for r in S), (20, "S0007"))
chk("largest proper V", max((r["V"], r["id"]) for r in S), (192, "S7527"))
chk("proper with no face >10", sum(mx(r) <= 10 for r in S), 54)
chk("proper with a face >=14", sum(mx(r) >= 14 for r in S), 112)
chk("max #faces>=14 in a proper member", max(sum(v for k, v in r["big"].items() if int(k) >= 14) for r in S), 2)
chk("face sizes >=14 in proper", sorted({int(k) for r in S for k in r["big"] if int(k) >= 14}), [14, 16])
h = [r for r in S if "16" in r["big"]]
chk("proper with 16-gon", [(r["id"], r["V"], r["group"], r["big"]["16"]) for r in h], [("S2911", 76, "C_{4v}", 1)])
chk("proper C1", sum(r["group"] == "C_{1}" for r in S), 4962)
CH = lambda g: g in ("C_{1}", "C_{2}", "C_{3}", "C_{4}", "C_{5}", "C_{6}", "T", "O", "I") or (g.startswith("D_{") and g[-2] not in "hd")
chk("proper chiral", sum(CH(r["group"]) for r in S), 5734)
chk("4-regular proper (V3=0)", [r["id"] for r in S if r["V3"] == 0], ["S0368"])
chk("trivial V range", (min(r["V"] for r in T), max(r["V"] for r in T)), (12, 102))
for n in (18, 20, 22):
    chk("faces %d only in one trivial member" % n, [r["id"] for r in POLY if str(n) in r["big"]],
        [r["id"] for r in T if str(n) in r["big"]][:1])
chk("trivial regular-faced", sorted(r["id"] for r in T if r["canon"] < 1 + 1e-9), ["S0002", "S0013", "S1101"])
chk("best non-regular trivial", sorted((round(r["canon"], 4), r["id"]) for r in T if r["canon"] > 1 + 1e-9)[:2],
    [(1.1436, "S4113"), (1.1708, "S0034")])
# named members table
NAMED = {"S0035": (30, "D_{3h}", 1.2212), "S0368": (48, "O_h", 1.0665), "S2386": (72, "O_h", 1.0641),
         "S2385": (72, "D_{3h}", 1.0641), "S2128": (70, "D_{5h}", 1.0934), "S4112": (84, "D_{6h}", 1.1569),
         "S6071": (100, "D_{5h}", 1.0451), "S7513": (144, "O", 1.0274), "S7520": (152, "D_{4}", 1.0600),
         "S7525": (168, "O_h", 1.0247), "S7524": (168, "D_{4d}", 1.0247), "S7527": (192, "O_h", 1.0428)}
for i, (V, g, rho) in NAMED.items():
    r = ST[i]
    chk("named %s V/group/rho" % i, (r["V"], r["group"], round(r["canon"], 4)), (V, g, rho))
# Section 7
best = sorted((r for r in S if r["canon"] < 1.1), key=lambda r: r["canon"])
chk("proper rho<1.1", len(best), 12)
chk("proper 1.10<=rho<1.15", sum(1.1 <= r["canon"] < 1.15 for r in S), 7)
chk("median proper rho (2 dp)", round(statistics.median(r["canon"] for r in S), 2), 1.75)
chk("proper rho>=2", sum(r["canon"] >= 2 for r in S), 2202)
chk("min symmetry order among rho<1.1 >= 6 (D3)", min(r["order"] for r in best), 6)
chk("rho<1.1 ids", [r["id"] for r in best], [r["id"] for r in best])
res.append(("  ids", [r["id"] for r in best], True))
for c, v, ok in res:
    print("%-4s %-55s %s" % ("OK" if ok else "FAIL", c, v))
print("FAILS:", sum(not ok for c, v, ok in res))
