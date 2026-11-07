# Sesquitruncation

Code and data for the paper *Sesquitruncation: a new polytope operation* by Thomas Shapiro.

Sesquitruncation truncates every vertex of order 3 of a polyhedron and replaces every vertex of order k ≥ 4 by a k-gon surrounded by k triangles. The paper classifies the locally spherical sesquitruncates of polyhedra whose vertices have order 3 or 4. There are 7176 proper ones and 76 trivial ones (truncations). With a vertex of order ≥ 5, only the pentagonal pyramid qualifies.

## Data

- `members.csv`: one row per member. The columns are:
  - `id`: S0001–S7527.
  - `class`: one of
    - `proper`: P has an order-4 vertex.
    - `trivial`: P is simple, so σ(P) is the truncation.
    - `degenerate`: P has a digon or a 2-vertex cut.
  - `V`, `F`: the numbers of vertices and faces of σ(P).
  - `faces_of_P`.
  - `big_faces`: given as count×size.
  - `group`: the combinatorial symmetry group, which equals the point group of the canonical form.
  - `group_order`.
  - `canonical_ratio`: the edge-length ratio of the canonical realization.
- `off/<class>/<id>.off`: the canonical (midsphere) realization of every member, in OFF format.
  - All edges are tangent to the unit sphere, and the barycentre of the tangency points is the origin.
  - The header comments give the group, the ratio, the big faces and the dual P\* of the input.
- `off/beyond/`: the canonical forms of the examples in Section 6.4 and Proposition 3.10:
  - the sesquitruncated cuboctahedron,
  - the sesquitruncated icosidodecahedron,
  - the sesquitruncated order-5 truncated deltoidal hexecontahedron,
  - the sesquitruncated pentagonal pyramid.

| class | members |
|---|---|
| proper | 7176 |
| trivial | 76 |
| degenerate inputs | 275 |

## Code (`code/`)

**Enumeration**
- Generator 1 (typed maps): `ext_gen2.py` with `tmap3x.py`.
- Generator 2 (independent): `gen2_run.sh`, which runs `plantri -pc3f4 n | gen2_filter | gen2_plantri.py`. plantri is by Brinkmann and McKay.
- `gen2_compare.py`: compares the two generators.
- `gen3_run.sh` with `gen3_filter.c`: the check with order-5 vertices allowed (`plantri -pc3f5`).
- `gsesq.py`: the general operation for any vertex orders.
- `pp_off.py`: the canonical sesquitruncated pentagonal pyramid.

**Statistics and tables**
- `sesq_stats.py`: symmetry groups and statistics.
- `sesq_tables.py`: the tables in the paper.
- `verify_paper.py`, `verify_named.py`: check the numbers quoted in the paper.

**Geometry**
- `mobius.py`: the best edge ratio over all midscribed realizations.
- `hart.py`: canonicalization by iteration, used for the examples with flat or hyperbolic vertices.
- `ana_degen.py`: vertex types and sphericity of an OFF file.

**Higher dimensions**
- `sesq4d.py`: the sesquitruncated 24-cell and joined pentachoron.
- `jp_fit.py`: fits the joined pentachoron.
- `sesq_honey.py`: the rhombic dodecahedral honeycomb.
- `sesq_honey2.py`: Voronoi honeycombs, including the trapezo-rhombic one.
- `trd_fit.py`: the per-class fit for the trapezo-rhombic honeycomb.
- The `*.npy` / `*.npz` files hold their parameters and coordinates.

**Figures**
- `render_poly.py`: polyhedra.
- `render_hd2.py`: honeycombs and 4-polytopes.
- `render_hd.py`: helpers.
- Output is TikZ, compiled with `tectonic`.

**Helpers (`lib/`)**
- Maps, the operation, and the canonical (midsphere) solver.

## Requirements and paths

- Python 3, numpy, scipy and networkx; plantri for the plantri-based checks.
- The scripts expect `lib/` on `PYTHONPATH`.
- Several scripts read or write a data directory `~/nmj/sesqpaper/ext2`, set near the top of each script.
