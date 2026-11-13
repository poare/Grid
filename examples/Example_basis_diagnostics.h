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
    A.Op(V[b],Av);
    RealD n2 = norm2(V[b]);
    nul[b] = std::sqrt(norm2(Av) / n2);
    rq[b]  = abs(innerProduct(V[b],Av)) / n2;   // ComplexD is thrust::complex on GPU builds
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
 * glmin, glmax: the extreme eigenvalues of the column-normalised global Gram
 * G = sum_b G_b.  N(N+1)/2 calls of blockInnerProductD; host storage N^2
 * complex numbers per local block.  V must be raw, never block-orthonormalised.
 */
template<class Field>
void BasisBlockGram(std::vector<Field> &V, GridBase *bgrid,
                    std::vector<RealD> &lmin, std::vector<RealD> &kappa,
                    std::vector<RealD> &deff, std::vector<RealD> &pres,
                    RealD &glmin, RealD &glmax)
{
  typedef std::complex<double> cplx;
  const int n    = V.size();
  const int nloc = bgrid->lSites();
  const RealD nb = bgrid->gSites();
  std::vector<Eigen::MatrixXcd> Gb(nloc,Eigen::MatrixXcd(n,n));
  LatticeComplexD ip(bgrid);
  std::vector<ComplexD> loc;
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      blockInnerProductD(ip,V[i],V[j]);              // <v_i|v_j> on each block
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
  glmin = gl(0);
  glmax = gl(n-1);

  std::vector<RealD> l(nloc), k(nloc), e(nloc), p(nloc);
  for (int b = 0; b < nloc; b++) {
    Eigen::VectorXd ev = NormalisedEigenvalues(Gb[b]);
    l[b] = ev(0);
    k[b] = ev(0) > 0.0 ? std::sqrt(ev(n-1)/ev(0)) : INFINITY;
    e[b] = ev.sum()*ev.sum() / ev.squaredNorm();
    RealD pm = INFINITY;
    for (int i = 0; i < n; i++) pm = std::min(pm, nb*Gb[b](i,i).real() / G(i,i).real());
    p[b] = pm;
  }
  BasisBlockGather(bgrid,l,lmin);
  BasisBlockGather(bgrid,k,kappa);
  BasisBlockGather(bgrid,e,deff);
  BasisBlockGather(bgrid,p,pres);
}

NAMESPACE_END(Grid);
