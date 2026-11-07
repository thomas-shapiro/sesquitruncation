/* gen3_filter: generalized sesquitruncation (order-k vertices of P, k >= 4, become a k-gon ringed by
   triangles).  Reads plantri planar_code of P* (faces of size 3..MAXF = vertex orders of P), keeps maps
   whose generalized sesquitruncate is strictly locally spherical AND that have at least one face of
   size 5 (an order-5 vertex of P); writes them as planar_code.  Big face of a P*-vertex:
   sum over incident faces of 2 (triangle) or 3 (size >= 4).
   gcc -O2 -o gen3_filter gen3_filter.c -lm */
#include <stdio.h>
#include <math.h>
#define MAXN 64
#define MAXD 64
#define MAXF 6
static int n, deg[MAXN], adj[MAXN][MAXD];
static double ang(int k) { return M_PI * (k - 2) / k; }
int main(void) {
  char hdr[15];
  if (fread(hdr, 1, 15, stdin) != 15) return 1;
  fwrite(">>planar_code<<", 1, 15, stdout);
  long nin = 0, nout = 0, any = 0;
  int c;
  while ((c = getchar()) != EOF) {
    n = c; nin++;
    for (int v = 0; v < n; v++) {
      deg[v] = 0;
      while ((c = getchar()) != 0) adj[v][deg[v]++] = c - 1;
    }
    int big[MAXN] = {0}, ok = 1, npent = 0;
    static char seen[MAXN][MAXD];
    for (int v = 0; v < n; v++) for (int j = 0; j < deg[v]; j++) seen[v][j] = 0;
    static int fv[MAXF + 2], fl[MAXN * MAXF], fk[MAXN * MAXF];
    int nfl = 0;
    for (int u = 0; u < n && ok; u++) for (int j = 0; j < deg[u] && ok; j++) {
      if (seen[u][j]) continue;
      int a = u, ja = j, len = 0;
      while (!seen[a][ja]) {
        seen[a][ja] = 1; if (len < MAXF + 2) fv[len] = a; len++;
        int b = adj[a][ja], i;
        for (i = 0; adj[b][i] != a; i++);
        int jb = (i - 1 + deg[b]) % deg[b];
        a = b; ja = jb;
      }
      if (len < 3 || len > MAXF) { ok = 0; break; }
      if (len == 5) npent++;
      for (int t = 0; t < len; t++) big[fv[t]] += (len == 3 ? 2 : 3);
      if (len >= 4) for (int t = 0; t < len; t++) { fl[nfl] = fv[t]; fk[nfl++] = len; }
    }
    if (ok) for (int u = 0; u < n && ok; u++) for (int j = 0; j < deg[u]; j++) {
      int v = adj[u][j];
      if (u < v && ang(3) + ang(big[u]) + ang(big[v]) >= 2 * M_PI - 1e-12) { ok = 0; break; }
    }
    if (ok) for (int t = 0; t < nfl; t++)
      if (2 * ang(3) + ang(fk[t]) + ang(big[fl[t]]) >= 2 * M_PI - 1e-12) { ok = 0; break; }
    if (ok) any++;
    if (ok && npent > 0) {
      nout++; putchar(n);
      for (int v = 0; v < n; v++) { for (int j = 0; j < deg[v]; j++) putchar(adj[v][j] + 1); putchar(0); }
    }
  }
  fprintf(stderr, "# read %ld spherical %ld with-order5 %ld\n", nin, any, nout);
  return 0;
}
