// Example_basis_diagnostics.h
//
// Diagnostics of a raw near-null basis V, for the question why deeper inverse
// iteration makes the MG worse at a fixed field (research hub,
// projects/multigrid, notes/qhmc-48cubed-basis-updates.md section 4.3).  The
// over-convergence hypothesis predicts better nulliness, growing principal
// angles to a fresh basis, and a worse block-local representation of fresh
// near-null vectors than the fresh basis gives.
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
 * normalised Gaussian noise, then three rounds of unpreconditioned GCR(10,10)
 * inverse iteration to 5e-4, at most 30 restarts.  The rounds' own output is
 * bracketed by QHMC-TESTGEN lines so the compact parser does not count them
 * as refresh passes.
 */
template<class Field>
void BasisTestVectors(std::vector<Field> &T, LinearOperatorBase<Field> &A, GridParallelRNG &RNG)
{
  RealD t = -usecond();
  std::cout << GridLogMessage << "QHMC-TESTGEN begin n " << T.size() << std::endl;
  for (auto &v : T) {
    gaussian(RNG,v);
    v = v*std::pow(norm2(v),-0.5);
  }
  for (int round = 0; round < 3; round++) RefineBasisGCR(T,A,T.size(),5.0e-4,30);
  t = (t+usecond())/1.0e6;
  std::cout << GridLogMessage << "QHMC-TESTGEN end n " << T.size() << " seconds " << t << std::endl;
}

NAMESPACE_END(Grid);
