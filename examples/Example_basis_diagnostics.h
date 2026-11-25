// Example_basis_diagnostics.h
//
// Diagnostics of a raw near-null basis V, for the question why deeper inverse
// iteration makes the MG worse at a fixed field (research hub,
// projects/multigrid, notes/qhmc-48cubed-basis-updates.md section 4.3).  The
// over-convergence hypothesis predicts better nulliness, growing principal
// angles to a fresh basis, and a worse block-local representation of fresh
// near-null vectors than the fresh basis gives.
//
// The block-level measures (BasisDefect, BasisBlockGram) are sections 3.1 and
// 3.2 of Peter Boyle's hand-off, notes/mg-hmc-peter/
// BasisColinearityDiagnostic_Handoff.md: completeness of fresh near-null
// vectors in the block span, and the local colinearity of the raw basis.
//
// Built from SpanProjector (Example_mg_hmc.h), RefineBasisGCR
// (Example_mg_refine.h) and the coarsening's own block projector; include
// both of those headers first.
#pragma once

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

/** mean, min and max of x */
inline std::string BasisStat(const std::string &name, const std::vector<RealD> &x)
{
  RealD mean = 0.0, lo = INFINITY, hi = -INFINITY;
  for (auto v : x) { mean += v / x.size(); lo = std::min(lo,v); hi = std::max(hi,v); }
  std::ostringstream s;
  s.precision(6);
  s << " " << name << "_mean " << mean << " " << name << "_min " << lo << " " << name << "_max " << hi;
  return s.str();
}

/** x as one comma-separated token */
inline std::string BasisList(const std::vector<RealD> &x)
{
  std::ostringstream s;
  s.precision(6);
  for (size_t i = 0; i < x.size(); i++) s << (i ? "," : "") << x[i];
  return s.str();
}

/** Copy between fine fields of the same or of different precision. */
template<class Field>
void BasisFieldCopy(const Field &in, Field &out) { out = in; }
template<class FieldIn, class Field>
void BasisFieldCopy(const FieldIn &in, Field &out) { precisionChange(out,in); }

/** ||A v|| / ||v|| for one vector; Av is left holding A v. */
template<class Field>
RealD BasisVectorNulliness(const Field &v, LinearOperatorBase<Field> &A, Field &Av)
{
  A.Op(v,Av);
  return std::sqrt(norm2(Av) / norm2(v));
}

/** |<w|v>| / (|w| |v|) */
template<class Field>
RealD BasisCosine(const Field &w, const Field &v)
{
  return abs(innerProduct(w,v)) / std::sqrt(norm2(w)*norm2(v));   // ComplexD is thrust::complex on GPU builds
}

/**
 * Nulliness of each basis vector: nul_b = ||A v_b|| / ||v_b|| and
 * rq_b = |<v_b|A v_b>| / ||v_b||^2.
 */
template<class Field>
void BasisNulliness(std::vector<Field> &V, LinearOperatorBase<Field> &A,
                    std::vector<RealD> &nul, std::vector<RealD> &rq)
{
  Field Av(V[0].Grid());
  nul.resize(V.size());
  rq.resize(V.size());
  for (size_t b = 0; b < V.size(); b++) {
    nul[b] = BasisVectorNulliness(V[b],A,Av);
    rq[b]  = abs(innerProduct(V[b],Av)) / norm2(V[b]);   // ComplexD is thrust::complex on GPU builds
  }
}

/**
 * G(i,j) = <W_i|V_j>.  W may be stored at lower precision; each W_i is
 * converted into one temporary of V's type.
 */
template<class FieldW, class Field>
Eigen::MatrixXcd BasisCrossGram(std::vector<FieldW> &W, std::vector<Field> &V)
{
  std::vector<const Field *> Vp;
  for (auto &v : V) Vp.push_back(&v);
  Field w(V[0].Grid());
  Eigen::MatrixXcd G(W.size(),V.size());
  std::vector<ComplexD> ip;
  for (size_t i = 0; i < W.size(); i++) {
    BasisFieldCopy(W[i],w);
    innerProductMulti(ip,Vp,w);                     // ip[j] = <V_j|W_i>
    for (size_t j = 0; j < V.size(); j++) G(i,j) = conjugate(ip[j]);
  }
  return G;
}

/**
 * Cosines of the principal angles between span(W) and span(V), largest first,
 * from the Gram matrices GW = W^dag W, GV = V^dag V and C = W^dag V:
 *     cos theta_i = sigma_i( L_W^{-1} C L_V^{-dag} ),  G = L L^dag .
 * Neither basis needs to be orthonormal.
 */
inline std::vector<RealD> PrincipalCosines(const Eigen::MatrixXcd &GW, const Eigen::MatrixXcd &GV,
                                           const Eigen::MatrixXcd &C)
{
  Eigen::LLT<Eigen::MatrixXcd> LW(GW), LV(GV);
  GRID_ASSERT( LW.info()==Eigen::Success && LV.info()==Eigen::Success );
  Eigen::MatrixXcd X = LW.matrixL().solve(C);
  Eigen::MatrixXcd M = LV.matrixL().solve(X.adjoint()).adjoint();
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M);
  std::vector<RealD> c(svd.singularValues().size());
  for (size_t i = 0; i < c.size(); i++) c[i] = std::min(1.0,(RealD)svd.singularValues()(i));
  return c;
}

/**
 * How well a basis represents vectors T outside it, locally and globally:
 *     loc_t  = ||(1 - P P^dag) t|| / ||t||,  P the block-orthonormal
 *              prolongator of the current coarsening (Proj),
 *     glob_t = ||Q t|| / ||t||,               Q off span(V).
 * T may be stored at lower precision; it is processed in chunks of nchunk
 * vectors of the fine type on fgrid, so the projector's scratch stays at
 * nchunk fields.
 */
template<class Field, class CoarseVector, class Projector, class FieldT>
void BasisRepresentation(Projector &Proj, SpanProjector<Field> &Q, GridBase *fgrid, GridBase *cgrid,
                         std::vector<FieldT> &T, int nchunk,
                         std::vector<RealD> &loc, std::vector<RealD> &glob)
{
  int nt = T.size();
  loc.resize(nt);
  glob.resize(nt);
  for (int t0 = 0; t0 < nt; t0 += nchunk) {
    int nc = std::min(nchunk,nt-t0);
    std::vector<Field> f(nc,fgrid), back(nc,fgrid);
    std::vector<CoarseVector> c(nc,cgrid);
    for (int r = 0; r < nc; r++) BasisFieldCopy(T[t0+r],f[r]);
    Proj.blockProject(f,c);
    Proj.blockPromote(back,c);
    for (int r = 0; r < nc; r++) {
      RealD n2 = norm2(f[r]);
      back[r] = f[r] - back[r];
      loc[t0+r] = std::sqrt(norm2(back[r]) / n2);
      Q.Complement(f[r],back[r]);
      glob[t0+r] = std::sqrt(norm2(back[r]) / n2);
    }
  }
  Proj.ReleaseScratch();
}

/**
 * Test vectors made the way the setup makes the basis (CreateSubspaceGCR):
 * normalised Gaussian noise, then rounds of unpreconditioned GCR(10,10)
 * inverse iteration to 5e-4, at most 30 restarts.  Rounds is ascending; with
 * n = T.size()/Rounds.size(), the n vectors after Rounds[d] rounds are stored
 * in T[d*n .. (d+1)*n).  One draw serves every depth: depth Rounds[d+1]
 * continues depth Rounds[d].  The work is n fields of the operator's type on
 * fgrid; T may be stored at lower precision.  Rounds {3} is the setup's depth.
 * The rounds' own output is bracketed by QHMC-TESTGEN lines so the compact
 * parser does not count them as refresh passes.
 */
template<class Field, class FieldT>
void BasisTestVectors(std::vector<FieldT> &T, const std::vector<int> &Rounds,
                      LinearOperatorBase<Field> &A, GridBase *fgrid, GridParallelRNG &RNG)
{
  const int nd = Rounds.size();
  const int n  = T.size() / nd;
  GRID_ASSERT( nd > 0 && n*nd == (int)T.size() );
  RealD t = -usecond();
  std::cout << GridLogMessage << "QHMC-TESTGEN begin n " << n << " rounds";
  for (auto r : Rounds) std::cout << " " << r;
  std::cout << std::endl;
  std::vector<Field> W(n,fgrid);
  for (auto &v : W) {
    gaussian(RNG,v);
    v = v*std::pow(norm2(v),-0.5);
  }
  int done = 0;
  for (int d = 0; d < nd; d++) {
    for (; done < Rounds[d]; done++) RefineBasisGCR(W,A,n,5.0e-4,30);
    for (int i = 0; i < n; i++) BasisFieldCopy(W[i],T[d*n+i]);
  }
  t = (t+usecond())/1.0e6;
  std::cout << GridLogMessage << "QHMC-TESTGEN end n " << n << " seconds " << t << std::endl;
}

//////////////////////////////////////////////////////////////////////
// Block-level measures.  bgrid is the block grid of a coarsening: one site
// per aggregate, with the fine grid's SIMD layout so that blockInnerProductD
// maps fine fields onto it (BasisBlockGrid).  Per-block values come to the
// host in local lexicographic order and are gathered, so every rank holds
// all N_b of them in global-index order for quantiles and worst blocks.
//////////////////////////////////////////////////////////////////////

/** The block grid of the coarse lattice cgrid: its sites, fgrid's SIMD layout and processor grid. */
inline GridCartesian *BasisBlockGrid(GridBase *fgrid, GridBase *cgrid)
{
  Coordinate dims = cgrid->FullDimensions();
  Coordinate simd = fgrid->_simd_layout;
  Coordinate mpi  = fgrid->_processors;
  GRID_ASSERT( dims.size() == simd.size() );
  for (int d = 0; d < (int)dims.size(); d++) {
    GRID_ASSERT( mpi[d] == cgrid->_processors[d] );
    GRID_ASSERT( (dims[d]/mpi[d]) % simd[d] == 0 );   // the fine SIMD layout fits the local block lattice
  }
  return new GridCartesian(dims,simd,mpi);
}

/** A block-level field's sites on this rank, in local lexicographic order. */
inline void BasisBlockLocal(const LatticeComplexD &x, std::vector<ComplexD> &loc)
{
  std::vector<TComplexD> s;
  unvectorizeToLexOrdArray(s,x);
  loc.resize(s.size());
  for (size_t i = 0; i < s.size(); i++) loc[i] = TensorRemove(s[i]);
}

/** Per-block values loc (local lexicographic order) from every rank, by global block index. */
inline void BasisBlockGather(GridBase *bgrid, const std::vector<RealD> &loc, std::vector<RealD> &all)
{
  const int nd = bgrid->Nd();
  Coordinate lcoor(nd), gcoor(nd);
  all.assign(bgrid->gSites(),0.0);
  for (int64_t l = 0; l < (int64_t)loc.size(); l++) {
    Lexicographic::CoorFromIndex(lcoor,l,bgrid->_ldimensions);
    for (int d = 0; d < nd; d++) gcoor[d] = bgrid->_processor_coor[d]*bgrid->_ldimensions[d] + lcoor[d];
    int64_t g;
    bgrid->GlobalCoorToGlobalIndex(gcoor,g);
    all[g] = loc[l];
  }
  bgrid->GlobalSumVector(all.data(),all.size());
}

/** Nearest-rank quantiles qs (each in [0,1]) of x. */
inline std::vector<RealD> BasisQuantiles(std::vector<RealD> x, const std::vector<RealD> &qs)
{
  std::sort(x.begin(),x.end());
  std::vector<RealD> q;
  for (auto f : qs) q.push_back(x[(size_t)std::llround(f*(x.size()-1))]);
  return q;
}

/**
 * The k blocks with the largest (largest = true) or smallest values of all, as
 * one token "c0.c1.c2.c3.c4:value,..." with the block coordinates on bgrid
 * (c0 is the fifth dimension, extent 1 when the blocks span all of L_s).
 */
inline std::string BasisBlockWorst(GridBase *bgrid, const std::vector<RealD> &all, int k, bool largest)
{
  std::vector<int64_t> idx(all.size());
  for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
  k = std::min<int64_t>(k,idx.size());
  std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),
                    [&](int64_t a,int64_t b){ return largest ? all[a] > all[b] : all[a] < all[b]; });
  std::ostringstream s;
  s.precision(4);
  Coordinate c(bgrid->Nd());
  for (int i = 0; i < k; i++) {
    Lexicographic::CoorFromIndex(c,idx[i],bgrid->FullDimensions());
    s << (i ? "," : "");
    for (int d = 0; d < bgrid->Nd(); d++) s << (d ? "." : "") << c[d];
    s << ":" << all[idx[i]];
  }
  return s.str();
}

/**
 * Completeness of the vectors T[t0 .. t0+nt) in the block span of the current
 * coarsening (hand-off section 3.1).  With P the block-orthonormal prolongator
 * of Proj and r = (1 - P P^dag) t:
 *   defect  = ||r||^2 / ||t||^2
 *   adefect = ||A r||^2 / ||A t||^2   small: the missed part is smooth, a gap in
 *                                     the coarse space; near 1 or above: rough,
 *                                     left to the smoother
 *   glob    = ||Q t||^2 / ||t||^2     Q off span(V); only when Q is given
 *   nul     = ||A t|| / ||t||
 *   blocks  = ||r||_b^2 / ||t||_b^2 for every block b, gathered (one list per t)
 * Squared, as in the hand-off: defect = loc^2 and glob = glob^2 of
 * BasisRepresentation.  T may be stored at lower precision; it is processed in
 * chunks of nchunk vectors of the fine type on fgrid, as BasisRepresentation.
 */
template<class Field, class CoarseVector, class Projector, class FieldT>
void BasisDefect(Projector &Proj, LinearOperatorBase<Field> &A, SpanProjector<Field> *Q,
                 GridBase *fgrid, GridBase *cgrid, GridBase *bgrid,
                 std::vector<FieldT> &T, int t0, int nt, int nchunk,
                 std::vector<RealD> &defect, std::vector<RealD> &adefect, std::vector<RealD> &glob,
                 std::vector<RealD> &nul, std::vector<std::vector<RealD> > &blocks)
{
  defect.resize(nt);
  adefect.resize(nt);
  glob.assign(nt,0.0);
  nul.resize(nt);
  blocks.resize(nt);
  LatticeComplexD rb(bgrid), tb(bgrid);
  std::vector<ComplexD> rl, tl;
  std::vector<RealD> ratio;
  Field Ax(fgrid);
  for (int c0 = 0; c0 < nt; c0 += nchunk) {
    int nc = std::min(nchunk,nt-c0);
    std::vector<Field> f(nc,fgrid), r(nc,fgrid);
    std::vector<CoarseVector> c(nc,cgrid);
    for (int i = 0; i < nc; i++) BasisFieldCopy(T[t0+c0+i],f[i]);
    Proj.blockProject(f,c);
    Proj.blockPromote(r,c);
    for (int i = 0; i < nc; i++) {
      const int t = c0+i;
      r[i] = f[i] - r[i];
      RealD n2 = norm2(f[i]);
      defect[t] = norm2(r[i]) / n2;
      A.Op(f[i],Ax);
      RealD an2 = norm2(Ax);
      nul[t] = std::sqrt(an2 / n2);
      A.Op(r[i],Ax);
      adefect[t] = norm2(Ax) / an2;
      if ( Q ) {
        Q->Complement(f[i],Ax);
        glob[t] = norm2(Ax) / n2;
      }
      blockInnerProductD(rb,r[i],r[i]);
      blockInnerProductD(tb,f[i],f[i]);
      BasisBlockLocal(rb,rl);
      BasisBlockLocal(tb,tl);
      ratio.resize(rl.size());
      for (size_t b = 0; b < rl.size(); b++) ratio[b] = real(rl[b]) / real(tl[b]);
      BasisBlockGather(bgrid,ratio,blocks[t]);
    }
  }
  Proj.ReleaseScratch();
}

/**
 * Local colinearity of the raw basis V (hand-off section 3.2).  For every
 * block b, G_b = V_b^dag V_b (N x N, N = V.size()) and its column-normalised
 * form C_b = D_b^{-1/2} G_b D_b^{-1/2}, D_b = diag(G_b).  Per block, gathered:
 *   lmin  = lambda_min(C_b)
 *   kappa = sqrt(lambda_max(C_b) / lambda_min(C_b)), the condition number of V_b
 *   deff  = (sum lambda)^2 / (sum lambda^2), between 1 and N: how many of the
 *           N vectors are independent on b
 *   pres  = min_i N_b ||v_i||_b^2 / ||v_i||^2, 1 when every vector is spread
 *           uniformly; small when some vector is nearly absent from b, which
 *           the column normalisation hides
 *   gmean = (det C_b)^{1/N} = exp(mean_k log lambda_k), the geometric-mean
 *           eigenvalue, 1 for orthonormal columns.  det C_b is the product over
 *           k of the fraction of v_k on b that is new against v_1..v_{k-1}, so
 *           it falls with every direction that goes to 0, not only the smallest
 *   rank[t] = #{k : lambda_k > taus[t]}, the numerical rank of C_b at floor taus[t]
 * logspec[k]: mean over all blocks of log lambda_k(C_b), eigenvalues ascending;
 * exp(logspec) is the typical sorted spectrum, and sum_k logspec[k] the mean
 * log det C_b.  Eigenvalues below 1e-16 count as 1e-16 in the logs.
 * glspec: the eigenvalues, ascending, of the column-normalised global Gram
 * C = D^{-1/2} G D^{-1/2}, G = sum_b G_b (the unblocked basis).  N(N+1)/2 calls of blockInnerProductD; host storage N^2
 * complex numbers per local block.  V must be raw, never block-orthonormalised.
 * keep, if given, receives the per-block G_b (local blocks, local lexicographic
 * order) for BasisBlockAngles; gram, if given, the global Gram G.
 * V may be stored at lower precision (FieldW): each pair is then converted
 * into two temporaries of the working type Field on fgrid, whose layout the
 * block grid matches.
 */
template<class Field, class FieldW>
void BasisBlockGram(std::vector<FieldW> &V, GridBase *fgrid, GridBase *bgrid,
                    std::vector<RealD> &lmin, std::vector<RealD> &kappa,
                    std::vector<RealD> &deff, std::vector<RealD> &pres,
                    std::vector<RealD> &gmean, const std::vector<RealD> &taus,
                    std::vector<std::vector<RealD> > &rank, std::vector<RealD> &logspec,
                    std::vector<RealD> &glspec, std::vector<Eigen::MatrixXcd> *keep = nullptr,
                    Eigen::MatrixXcd *gram = nullptr)
{
  typedef std::complex<double> cplx;
  constexpr bool same = std::is_same<Field,FieldW>::value;
  const int n    = V.size();
  const int nloc = bgrid->lSites();
  const RealD nb = bgrid->gSites();
  std::vector<Eigen::MatrixXcd> Gb(nloc,Eigen::MatrixXcd(n,n));
  LatticeComplexD ip(bgrid);
  std::vector<ComplexD> loc;
  std::vector<Field> tmp;                            // the converted pair, only for FieldW != Field
  if ( !same ) tmp.assign(2,Field(fgrid));
  auto Load = [&](int k, int slot) -> const Field & {
    if constexpr (same) return V[k];
    else { BasisFieldCopy(V[k],tmp[slot]); return tmp[slot]; }
  };
  for (int i = 0; i < n; i++) {
    const Field &vi = Load(i,0);
    for (int j = i; j < n; j++) {
      blockInnerProductD(ip,vi,Load(j,1));           // <v_i|v_j> on each block
      BasisBlockLocal(ip,loc);
      for (int b = 0; b < nloc; b++) {
        Gb[b](i,j) = loc[b];
        Gb[b](j,i) = conjugate(loc[b]);
      }
    }
  }
  Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(n,n);
  for (auto &g : Gb) G += g;
  bgrid->GlobalSumVector((RealD *)G.data(),2*n*n);

  // C = D^{-1/2} M D^{-1/2}: its eigenvalues, ascending
  auto NormalisedEigenvalues = [&](const Eigen::MatrixXcd &M) -> Eigen::VectorXd {
    Eigen::VectorXcd s = M.diagonal().real().cwiseSqrt().cwiseInverse().cast<cplx>();
    Eigen::MatrixXcd C = s.asDiagonal() * M * s.asDiagonal();
    C = 0.5*(C + C.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(C,Eigen::EigenvaluesOnly);
    return es.eigenvalues();
  };
  Eigen::VectorXd gl = NormalisedEigenvalues(G);
  glspec.assign(gl.data(),gl.data()+n);

  const int nt = taus.size();
  std::vector<RealD> l(nloc), k(nloc), e(nloc), p(nloc), gm(nloc);
  std::vector<std::vector<RealD> > r(nt,std::vector<RealD>(nloc));
  logspec.assign(n,0.0);
  for (int b = 0; b < nloc; b++) {
    Eigen::VectorXd ev = NormalisedEigenvalues(Gb[b]);
    l[b] = ev(0);
    k[b] = ev(0) > 0.0 ? std::sqrt(ev(n-1)/ev(0)) : INFINITY;
    e[b] = ev.sum()*ev.sum() / ev.squaredNorm();
    RealD pm = INFINITY;
    for (int i = 0; i < n; i++) pm = std::min(pm, nb*Gb[b](i,i).real() / G(i,i).real());
    p[b] = pm;
    RealD ld = 0.0;
    for (int i = 0; i < n; i++) {
      RealD li = std::log(std::max(ev(i),1.0e-16));
      ld += li;
      logspec[i] += li;
    }
    gm[b] = std::exp(ld/n);
    for (int t = 0; t < nt; t++) {
      int c = 0;
      for (int i = 0; i < n; i++) c += (ev(i) > taus[t]);
      r[t][b] = c;
    }
  }
  bgrid->GlobalSumVector(logspec.data(),n);
  for (auto &x : logspec) x /= nb;
  BasisBlockGather(bgrid,l,lmin);
  BasisBlockGather(bgrid,k,kappa);
  BasisBlockGather(bgrid,e,deff);
  BasisBlockGather(bgrid,p,pres);
  BasisBlockGather(bgrid,gm,gmean);
  rank.resize(nt);
  for (int t = 0; t < nt; t++) BasisBlockGather(bgrid,r[t],rank[t]);
  if ( keep ) *keep = std::move(Gb);
  if ( gram ) *gram = G;
}

/** BasisBlockGram for a basis stored at its working precision. */
template<class Field>
void BasisBlockGram(std::vector<Field> &V, GridBase *bgrid,
                    std::vector<RealD> &lmin, std::vector<RealD> &kappa,
                    std::vector<RealD> &deff, std::vector<RealD> &pres,
                    std::vector<RealD> &gmean, const std::vector<RealD> &taus,
                    std::vector<std::vector<RealD> > &rank, std::vector<RealD> &logspec,
                    std::vector<RealD> &glspec, std::vector<Eigen::MatrixXcd> *keep = nullptr,
                    Eigen::MatrixXcd *gram = nullptr)
{
  BasisBlockGram<Field,Field>(V,V[0].Grid(),bgrid,lmin,kappa,deff,pres,gmean,taus,rank,logspec,glspec,keep,gram);
}

/**
 * |<w_i|v_i>| / (|w_i| |v_i|) for each i: how much of each old vector w_i
 * survives in the new v_i.  W may be stored at lower precision.
 */
template<class FieldW, class Field>
std::vector<RealD> BasisPairCosines(std::vector<FieldW> &W, std::vector<Field> &V)
{
  Field w(V[0].Grid());
  std::vector<RealD> c(V.size());
  for (size_t i = 0; i < V.size(); i++) {
    BasisFieldCopy(W[i],w);
    c[i] = BasisCosine(w,V[i]);
  }
  return c;
}

/**
 * Principal angles between the block spans of two bases (hand-off section
 * 3.3), per block b.  GW, GV are the per-block Grams of W and V as kept by
 * BasisBlockGram (local blocks, local lexicographic order) and
 * C_b(i,j) = <w_i|v_j>_b.  On each side the Gram is column-normalised and
 * restricted to its eigenvectors above rtol times its largest eigenvalue (the
 * numerically independent directions, r_W and r_V of them); the cosines are
 * the singular values of Lambda_W^{-1/2} U_W^dag C' U_V Lambda_V^{-1/2},
 * min(r_W, r_V) of them.  Per block, gathered:
 *   sin2 = mean of 1 - cos^2 over the cosines: the fraction of the block span that turned
 *   cmin = the smallest cosine: the direction that turned most
 * W may be stored at lower precision: C then carries its rounding, which the
 * nearly dependent directions amplify by up to (lambda_max/lambda)^{1/2}.
 * N_W x N_V calls of blockInnerProductD.
 */
template<class FieldW, class Field>
void BasisBlockAngles(std::vector<FieldW> &W, const std::vector<Eigen::MatrixXcd> &GW,
                      std::vector<Field> &V, const std::vector<Eigen::MatrixXcd> &GV,
                      GridBase *bgrid, RealD rtol,
                      std::vector<RealD> &sin2, std::vector<RealD> &cmin)
{
  typedef std::complex<double> cplx;
  const int nw = W.size(), nv = V.size();
  const int nloc = bgrid->lSites();
  GRID_ASSERT( (int)GW.size() == nloc && (int)GV.size() == nloc );
  std::vector<Eigen::MatrixXcd> Cb(nloc,Eigen::MatrixXcd(nw,nv));
  LatticeComplexD ip(bgrid);
  std::vector<ComplexD> loc;
  Field w(V[0].Grid());
  for (int i = 0; i < nw; i++) {
    BasisFieldCopy(W[i],w);
    for (int j = 0; j < nv; j++) {
      blockInnerProductD(ip,w,V[j]);                  // <w_i|v_j> on each block
      BasisBlockLocal(ip,loc);
      for (int b = 0; b < nloc; b++) Cb[b](i,j) = loc[b];
    }
  }
  // D^{-1/2} M D^{-1/2} = U Lambda U^dag; returns D^{-1/2} U_r Lambda_r^{-1/2}, the r directions above rtol
  auto Whiten = [&](const Eigen::MatrixXcd &M) -> Eigen::MatrixXcd {
    Eigen::VectorXcd s = M.diagonal().real().cwiseSqrt().cwiseInverse().cast<cplx>();
    Eigen::MatrixXcd X = s.asDiagonal() * M * s.asDiagonal();
    X = 0.5*(X + X.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(X);
    const Eigen::VectorXd &ev = es.eigenvalues();
    const int n = ev.size();
    int r = 0;
    while (r < n && ev(n-1-r) > rtol*ev(n-1)) r++;
    Eigen::MatrixXcd U = es.eigenvectors().rightCols(r);
    Eigen::VectorXcd isq = ev.tail(r).cwiseSqrt().cwiseInverse().cast<cplx>();
    return s.asDiagonal() * U * isq.asDiagonal();
  };
  std::vector<RealD> s2(nloc), cm(nloc);
  for (int b = 0; b < nloc; b++) {
    Eigen::MatrixXcd M = Whiten(GW[b]).adjoint() * Cb[b] * Whiten(GV[b]);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M);
    const Eigen::VectorXd &c = svd.singularValues();
    RealD acc = 0.0, mn = 1.0;
    for (int i = 0; i < c.size(); i++) {
      RealD ci = std::min(1.0,(RealD)c(i));
      acc += 1.0 - ci*ci;
      mn = std::min(mn,ci);
    }
    s2[b] = acc / c.size();
    cm[b] = mn;
  }
  BasisBlockGather(bgrid,s2,sin2);
  BasisBlockGather(bgrid,cm,cmin);
}

/**
 * Block-resolved nulliness of the basis vectors: A applied to the whole
 * vector v_i, measured on each block,
 *     n_ib = ( ||A v_i||_b^2 / ||v_i||_b^2 )^{1/2},
 * so that sum_b ||A v_i||_b^2 = ||A v_i||^2.  Per block, over the N vectors,
 * gathered:
 *   bmin      = min_i n_ib, the most null basis vector on b
 *   bmed      = the median over i of n_ib
 *   bmax      = max_i n_ib, the roughest basis vector on b
 *   count[t]  = #{i : n_ib < taus[t]}, the basis vectors that are near-null on
 *               b at that scale
 * logspec[k]: mean over all blocks of log n_(k)b, the k-th smallest on each
 * block; exp(logspec) is the typical sorted local spectrum.  N applications
 * of A and 2N calls of blockInnerProductD; host storage N reals per local block.
 */
template<class Field>
void BasisBlockNulliness(std::vector<Field> &V, LinearOperatorBase<Field> &A, GridBase *bgrid,
                         const std::vector<RealD> &taus,
                         std::vector<RealD> &bmin, std::vector<RealD> &bmed, std::vector<RealD> &bmax,
                         std::vector<std::vector<RealD> > &count, std::vector<RealD> &logspec)
{
  const int n    = V.size();
  const int nloc = bgrid->lSites();
  const int nt   = taus.size();
  std::vector<std::vector<RealD> > nb(nloc,std::vector<RealD>(n));
  LatticeComplexD av(bgrid), vv(bgrid);
  std::vector<ComplexD> al, vl;
  Field Av(V[0].Grid());
  for (int i = 0; i < n; i++) {
    A.Op(V[i],Av);
    blockInnerProductD(av,Av,Av);
    blockInnerProductD(vv,V[i],V[i]);
    BasisBlockLocal(av,al);
    BasisBlockLocal(vv,vl);
    for (int b = 0; b < nloc; b++) nb[b][i] = std::sqrt(real(al[b]) / real(vl[b]));
  }
  std::vector<RealD> lo(nloc), md(nloc), hi(nloc);
  std::vector<std::vector<RealD> > c(nt,std::vector<RealD>(nloc));
  logspec.assign(n,0.0);
  for (int b = 0; b < nloc; b++) {
    std::vector<RealD> &x = nb[b];
    std::sort(x.begin(),x.end());
    lo[b] = x[0];
    md[b] = x[n/2];
    hi[b] = x[n-1];
    for (int k = 0; k < n; k++) logspec[k] += std::log(std::max(x[k],1.0e-300));
    for (int t = 0; t < nt; t++) {
      int m = 0;
      for (int k = 0; k < n; k++) m += (x[k] < taus[t]);
      c[t][b] = m;
    }
  }
  bgrid->GlobalSumVector(logspec.data(),n);
  for (auto &x : logspec) x /= bgrid->gSites();
  BasisBlockGather(bgrid,lo,bmin);
  BasisBlockGather(bgrid,md,bmed);
  BasisBlockGather(bgrid,hi,bmax);
  count.resize(nt);
  for (int t = 0; t < nt; t++) BasisBlockGather(bgrid,c[t],count[t]);
}

/**
 * Compatible relaxation: the smoother S run on the error equation A e = 0 with
 * the coarse space projected out after every sweep,
 *     e_{k+1} = (1 - P P^dag) (e_k - S(A e_k)),
 * from nvec Gaussian vectors with e_0 = (1 - P P^dag) eta.  The coarse
 * correction removes range(P) exactly, so the decay of e_k is what the
 * smoother alone does to the complement of the coarse space: slow decay means
 * slow modes that the coarse space does not hold.  Per sweep k = 1..nsweep,
 * averaged over the vectors (geometric mean):
 *   err[k-1] = ||e_k|| / ||e_{k-1}||
 *   res[k-1] = ||A e_k|| / ||A e_{k-1}||
 * S is the solver's own fine smoother; for a Krylov smoother this is an
 * operational rate, S not being a fixed linear map.  nsweep x (one smoother
 * call + 2 applications of A) per vector.
 */
template<class Field, class CoarseVector, class Projector>
void BasisCompatibleRelaxation(Projector &Proj, LinearOperatorBase<Field> &A, LinearFunction<Field> &S,
                               GridBase *fgrid, GridBase *cgrid, GridParallelRNG &RNG,
                               int nvec, int nsweep, std::vector<RealD> &err, std::vector<RealD> &res)
{
  err.assign(nsweep,0.0);
  res.assign(nsweep,0.0);
  std::vector<Field> e(1,fgrid), pe(1,fgrid);
  std::vector<CoarseVector> c(1,cgrid);
  Field Ae(fgrid), z(fgrid);
  auto Complement = [&]() {           // e <- (1 - P P^dag) e
    Proj.blockProject(e,c);
    Proj.blockPromote(pe,c);
    e[0] = e[0] - pe[0];
  };
  for (int v = 0; v < nvec; v++) {
    gaussian(RNG,e[0]);
    Complement();
    e[0] = e[0]*std::pow(norm2(e[0]),-0.5);
    A.Op(e[0],Ae);
    RealD ne = 1.0, na = std::sqrt(norm2(Ae));
    for (int k = 0; k < nsweep; k++) {
      z = Zero();
      S(Ae,z);
      e[0] = e[0] - z;
      Complement();
      A.Op(e[0],Ae);
      RealD ne1 = std::sqrt(norm2(e[0])), na1 = std::sqrt(norm2(Ae));
      err[k] += std::log(ne1/ne) / nvec;
      res[k] += std::log(na1/na) / nvec;
      ne = ne1; na = na1;
    }
  }
  for (auto &x : err) x = std::exp(x);
  for (auto &x : res) x = std::exp(x);
  Proj.ReleaseScratch();
}

NAMESPACE_END(Grid);
