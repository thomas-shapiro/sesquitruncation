/* gen2_filter: read plantri planar_code on stdin, keep maps whose sesquitruncate is strictly
   locally spherical, write them as planar_code on stdout. Same test as gen2_plantri.py:sesq_ok.
   gcc -O2 -o gen2_filter gen2_filter.c -lm */
#include <stdio.h>
#include <math.h>
#define MAXN 64
#define MAXD 64
static int n, deg[MAXN], adj[MAXN][MAXD];
static double ang(int k) { return M_PI * (k - 2) / k; }
int main(void) {
  char hdr[15];
  if (fread(hdr, 1, 15, stdin) != 15) return 1;
  fwrite(">>planar_code<<", 1, 15, stdout);
  long nin = 0, nout = 0;
  int c;
  while ((c = getchar()) != EOF) {
    n = c; nin++;
    for (int v = 0; v < n; v++) {
      deg[v] = 0;
      while ((c = getchar()) != 0) adj[v][deg[v]++] = c - 1;
    }
    int big[MAXN] = {0}, ok = 1;
    static char seen[MAXN][MAXD];
    for (int v = 0; v < n; v++) for (int j = 0; j < deg[v]; j++) seen[v][j] = 0;
    /* faces: same walk as faces_of: from dart (a,b), next = (b, adj[b][pos(a)-1]) */
    static int fv[4 * MAXN], q4[MAXN];
    int nq4 = 0;
    for (int u = 0; u < n && ok; u++) for (int j = 0; j < deg[u] && ok; j++) {
      if (seen[u][j]) continue;
      int a = u, ja = j, len = 0;
      while (!seen[a][ja]) {
        seen[a][ja] = 1; if (len < 8) fv[len] = a; len++;
        int b = adj[a][ja], i;
        for (i = 0; adj[b][i] != a; i++);
        int jb = (i - 1 + deg[b]) % deg[b];
        a = b; ja = jb;
      }
      if (len != 3 && len != 4) { ok = 0; break; }
      for (int t = 0; t < len; t++) big[fv[t]] += (len == 3 ? 2 : 3);
      if (len == 4) for (int t = 0; t < 4; t++) q4[nq4++] = fv[t];
    }
    if (ok) for (int u = 0; u < n && ok; u++) for (int j = 0; j < deg[u]; j++) {
      int v = adj[u][j];
      if (u < v && ang(3) + ang(big[u]) + ang(big[v]) >= 2 * M_PI - 1e-12) { ok = 0; break; }
    }
    if (ok) for (int t = 0; t < nq4; t++)
      if (2 * ang(3) + ang(4) + ang(big[q4[t]]) >= 2 * M_PI - 1e-12) { ok = 0; break; }
    if (ok) {
      nout++; putchar(n);
      for (int v = 0; v < n; v++) { for (int j = 0; j < deg[v]; j++) putchar(adj[v][j] + 1); putchar(0); }
    }
  }
  fprintf(stderr, "# read %ld kept %ld\n", nin, nout);
  return 0;
}
