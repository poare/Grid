/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_hmc.h

    Copyright (C) 2026

Author: Patrick Oare

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
//
// Example_mg_hmc.h
//
// Pieces of the MG-in-HMC test (Example_mg_hmc.cc): fine-matvec accounting,
// the projector onto the complement of a near-null span, and the projected
// operator and preconditioner used by the perturbative basis update.
// Design: research hub, projects/multigrid/notes/deflation-basis-hmc.md.
//
#pragma once

#include <map>

NAMESPACE_BEGIN(Grid);

/**
 * Counts applications of the fine operator, attributed to the category that
 * is current when the application happens (setup, coarsen, force, correct,
 * measure, ...). A HermOp is two applications. Applications of the coarse
 * operator are kept apart, under the same categories, and never enter Total().
 */
struct MatvecCounter {
  std::string category = "none";
  std::map<std::string, uint64_t> counts;
  std::map<std::string, uint64_t> coarse;

  void Set(const std::string &c) { category = c; }
  void Add(uint64_t n) { counts[category] += n; }
  void AddCoarse(uint64_t n) { coarse[category] += n; }
  uint64_t Get(const std::string &c) const {
    auto it = counts.find(c);
    return it == counts.end() ? 0 : it->second;
  }
  uint64_t GetCoarse(const std::string &c) const {
    auto it = coarse.find(c);
    return it == coarse.end() ? 0 : it->second;
  }
  uint64_t Total() const {
    uint64_t t = 0;
    for (auto &kv : counts) t += kv.second;
    return t;
  }
  uint64_t TotalCoarse() const {
    uint64_t t = 0;
    for (auto &kv : coarse) t += kv.second;
    return t;
  }
  /** One grep-able line: <tag> <category> <count> ... total <n> coarse_<category> <count> ... coarse_total <n> */
  void Report(const std::string &tag) const {
    std::cout << GridLogMessage << tag;
    for (auto &kv : counts) std::cout << " " << kv.first << " " << kv.second;
    std::cout << " total " << Total();
    for (auto &kv : coarse) std::cout << " coarse_" << kv.first << " " << kv.second;
    std::cout << " coarse_total " << TotalCoarse() << std::endl;
  }
};

/**
 * Wraps a LinearOperatorBase and counts every application in a shared
 * MatvecCounter. Op and AdjOp count 1, HermOp and HermOpAndNorm count 2.
 * With coarse = true the applications go to the counter's coarse tally.
 */
template<class Field>
class CountingLinearOperator : public LinearOperatorBase<Field> {
  LinearOperatorBase<Field> &_Op;
  MatvecCounter             &_Counter;
  bool                       _Coarse;
  void Count(uint64_t n) { if (_Coarse) _Counter.AddCoarse(n); else _Counter.Add(n); }
public:
  CountingLinearOperator(LinearOperatorBase<Field> &op, MatvecCounter &c, bool coarse=false)
    : _Op(op), _Counter(c), _Coarse(coarse) {};

  void OpDiag   (const Field &in, Field &out)                 { _Op.OpDiag(in,out); }
  void OpDir    (const Field &in, Field &out,int dir,int disp) { _Op.OpDir(in,out,dir,disp); }
  void OpDirAll (const Field &in, std::vector<Field> &out)    { _Op.OpDirAll(in,out); }
  void Op       (const Field &in, Field &out)                 { Count(1); _Op.Op(in,out); }
  void AdjOp    (const Field &in, Field &out)                 { Count(1); _Op.AdjOp(in,out); }
  void HermOp   (const Field &in, Field &out)                 { Count(2); _Op.HermOp(in,out); }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2) { Count(2); _Op.HermOpAndNorm(in,out,n1,n2); }
};

/**
 * Orthogonal projector onto the complement of span(V),
 *     Q = 1 - V (V^dag V)^{-1} V^dag ,
 * for a basis V that need not be orthonormal. The Gram matrix is Cholesky
 * factored once at construction; applying Q costs N inner products and one
 * linear combination. V is held by reference and must outlive the projector.
 */
template<class Field>
class SpanProjector {
  std::vector<Field>          &_V;
  Eigen::LLT<Eigen::MatrixXcd> _LLT;
  RealD                        _Cond;
  bool                         _Ok;
public:
  SpanProjector(std::vector<Field> &V) : _V(V) {
    int nn = _V.size();
    Eigen::MatrixXcd G(nn,nn);
    for (int i = 0; i < nn; i++) {
      for (int j = i; j < nn; j++) {
        G(i,j) = innerProduct(_V[i],_V[j]);
        G(j,i) = std::conj(G(i,j));
      }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(G,Eigen::EigenvaluesOnly);
    RealD lmin = es.eigenvalues().minCoeff(), lmax = es.eigenvalues().maxCoeff();
    _Cond = (lmin > 0.0) ? lmax / lmin : INFINITY;
    _LLT.compute(G);
    _Ok = (_LLT.info() == Eigen::Success) && std::isfinite(_Cond);
  }

  bool  Ok()           const { return _Ok; }
  RealD GramCondition() const { return _Cond; }

  /** out = Q in */
  void Complement(const Field &in, Field &out) const {
    assert(_Ok);
    int nn = _V.size();
    Eigen::VectorXcd c(nn);
    for (int i = 0; i < nn; i++) c(i) = innerProduct(_V[i],in);
    Eigen::VectorXcd a = _LLT.solve(c);
    out = in;
    for (int i = 0; i < nn; i++) out = out - ComplexD(a(i)) * _V[i];
  }
};

/** Q A Q, the operator of the subspace correction equation. */
template<class Field>
class ProjectedLinearOperator : public LinearOperatorBase<Field> {
  SpanProjector<Field>      &_Q;
  LinearOperatorBase<Field> &_A;
public:
  ProjectedLinearOperator(SpanProjector<Field> &Q, LinearOperatorBase<Field> &A) : _Q(Q), _A(A) {};

  void OpDiag   (const Field &in, Field &out)                 { assert(0); }
  void OpDir    (const Field &in, Field &out,int dir,int disp) { assert(0); }
  void OpDirAll (const Field &in, std::vector<Field> &out)    { assert(0); }
  void Op(const Field &in, Field &out) {
    Field tmp(in.Grid());
    _Q.Complement(in,tmp);
    _A.Op(tmp,out);
    _Q.Complement(out,out);
  }
  void AdjOp(const Field &in, Field &out) {
    Field tmp(in.Grid());
    _Q.Complement(in,tmp);
    _A.AdjOp(tmp,out);
    _Q.Complement(out,out);
  }
  void HermOp(const Field &in, Field &out) {
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2) { assert(0); }
};

/** Q M Q for a preconditioner M, matching ProjectedLinearOperator. */
template<class Field>
class ProjectedPreconditioner : public LinearFunction<Field> {
  SpanProjector<Field>  &_Q;
  LinearFunction<Field> &_M;
public:
  using LinearFunction<Field>::operator();
  ProjectedPreconditioner(SpanProjector<Field> &Q, LinearFunction<Field> &M) : _Q(Q), _M(M) {};
  void operator()(const Field &in, Field &out) {
    Field tmp(in.Grid());
    _Q.Complement(in,tmp);
    _M(tmp,out);
    _Q.Complement(out,out);
  }
};

/**
 * How far span(V) is from being invariant under A:
 *     ( sum_b |Q A v_b|^2 / sum_b |v_b|^2 )^{1/2} .
 * Zero for an exactly invariant subspace.
 */
template<class Field>
RealD InvarianceResidual(std::vector<Field> &V, LinearOperatorBase<Field> &A)
{
  SpanProjector<Field> Q(V);
  if ( !Q.Ok() ) return NAN;
  Field Av(V[0].Grid());
  RealD num = 0.0, den = 0.0;
  for (auto &v : V) {
    A.Op(v,Av);
    Q.Complement(Av,Av);
    num += norm2(Av);
    den += norm2(v);
  }
  return std::sqrt(num / den);
}

/**
 * psi(x, s) -> g(x) psi(x, s) for every s of a 5d domain wall field (s is
 * dimension 0 of the 5d grid). With U -> U^g, D(U^g) = G D(U) G^dag for this G.
 */
inline void GaugeTransformFermion5d(LatticeFermion &psi, LatticeColourMatrix &g)
{
  GridBase *FGrid = psi.Grid();
  int Ls = FGrid->GlobalDimensions()[0];
  LatticeFermion psi4(g.Grid());
  for (int s = 0; s < Ls; s++) {
    ExtractSlice(psi4, psi, s, 0);
    psi4 = g * psi4;
    InsertSlice(psi4, psi, s, 0);
  }
}

/**
 * Form of the first-order correction w in v -> v - w:
 *   Projected : (Q A_new Q) w = Q dA v , preconditioned by Q M Q
 *               (the subspace correction equation; exact to first order
 *               for a pure gauge change of an invariant span)
 *   Plain     : A_new w = Q dA v , preconditioned by M
 */
enum class CorrectionForm { Projected, Plain };

/** Cost and diagnostics of one perturbative update of a basis. */
struct UpdateStats {
  std::vector<int>   iters;         // outer GCR steps per vector
  std::vector<RealD> wratio;        // |w| / |v| per vector
  RealD    resid_before = 0.0;      // InvarianceResidual with A_new, before
  RealD    resid_after  = 0.0;      // ... and after the update
  uint64_t matvecs      = 0;        // fine matvecs under category "correct"
  RealD    seconds      = 0.0;
  int      capped       = 0;        // correction solves that hit maxsteps
  bool     skipped      = false;    // Gram matrix not positive definite
};

/**
 * First-order update of a near-null basis V from A_old to A_new:
 *     v -> (v - w) / |v - w| ,   w from CorrectionForm with dA = A_new - A_old ,
 * Q = 1 - V (V^dag V)^{-1} V^dag built from the OLD V and held fixed for all
 * vectors. Each solve is flexible GCR(16,16) to relative tolerance tol,
 * capped at maxsteps outer steps; a capped w is still applied. V must be the
 * basis as updated, never the block-orthonormalised copy in an Aggregation.
 * Matvecs are attributed to counter categories "correct" and "measure"; the
 * operators passed in must be CountingLinearOperators for them to register.
 * With nupdate >= 0 only the first nupdate vectors are corrected (Q still
 * spans all of V); the rest are returned unchanged.
 */
template<class Field>
UpdateStats PerturbativeUpdate(std::vector<Field> &V,
                               LinearOperatorBase<Field> &Anew,
                               LinearOperatorBase<Field> &Aold,
                               LinearFunction<Field> &Prec,
                               RealD tol, int maxsteps,
                               CorrectionForm form,
                               MatvecCounter &counter,
                               int nupdate = -1)
{
  UpdateStats st;
  RealD t0 = usecond();
  int nn = V.size();
  GridBase *grid = V[0].Grid();

  std::vector<Field> Vold = V;          // Q must stay fixed while V changes
  SpanProjector<Field> Q(Vold);
  if ( !Q.Ok() ) {
    st.skipped = true;
    std::cout << GridLogMessage << "MGHMC update skipped: Gram condition " << Q.GramCondition() << std::endl;
    return st;
  }

  counter.Set("measure");
  st.resid_before = InvarianceResidual(Vold, Anew);

  ProjectedLinearOperator<Field> QAQ(Q, Anew);
  ProjectedPreconditioner<Field> QMQ(Q, Prec);
  LinearOperatorBase<Field> &Op = (form == CorrectionForm::Projected) ? (LinearOperatorBase<Field> &)QAQ : Anew;
  LinearFunction<Field>     &M  = (form == CorrectionForm::Projected) ? (LinearFunction<Field> &)QMQ  : Prec;

  int nrestart = (maxsteps + 15) / 16;
  PrecGeneralisedConjugateResidualNonHermitian<Field> GCR(tol, nrestart, Op, M, 16, 16);

  Field Anv(grid), Aov(grid), rhs(grid), w(grid), r(grid);
  uint64_t c0 = counter.Get("correct");
  int nup = (nupdate < 0) ? nn : std::min(nupdate, nn);
  for (int b = 0; b < nup; b++) {
    counter.Set("correct");
    Anew.Op(Vold[b], Anv);
    Aold.Op(Vold[b], Aov);
    rhs = Anv - Aov;
    Q.Complement(rhs, rhs);

    w = Zero();
    GCR(rhs, w);
    st.iters.push_back(GCR.Steps());

    counter.Set("measure");
    Op.Op(w, r);
    r = rhs - r;
    if ( std::sqrt(norm2(r) / norm2(rhs)) > tol ) st.capped++;

    st.wratio.push_back(std::sqrt(norm2(w) / norm2(Vold[b])));
    V[b] = Vold[b] - w;
    V[b] = V[b] * std::pow(norm2(V[b]), -0.5);
  }
  st.matvecs = counter.Get("correct") - c0;

  counter.Set("measure");
  st.resid_after = InvarianceResidual(V, Anew);
  st.seconds = (usecond() - t0) / 1.0e6;
  return st;
}

/**
 * One MD trajectory, identical to Integrator::integrate, but calling
 * hook(s) after MD step s = 1 .. MDsteps so that work (a force-solve
 * stand-in, a basis update) can be done at the field of each step.
 */
template<class IntegratorType>
void IntegrateWithHook(IntegratorType &MD, LatticeGaugeField &U, std::function<void(int)> hook)
{
  MD.t_U = 0;
  for (int level = 0; level < MD.as.size(); ++level) MD.t_P[level] = 0;

  for (int stp = 0; stp < MD.Params.MDsteps; ++stp) {
    int first_step = (stp == 0);
    int last_step  = (stp == MD.Params.MDsteps - 1);
    MD.step(U, 0, first_step, last_step);
    hook(stp + 1);
  }

  for (int level = 0; level < MD.as.size(); ++level) {
    assert(fabs(MD.t_U - MD.t_P[level]) < 1.0e-6);
  }
  IntegratorType::FieldImplementation::Project(U);
  assert(fabs(MD.t_U - MD.Params.trajL) < 1.0e-6);
}

/** Metropolis test, as in HybridMonteCarlo::metropolis_test. */
inline bool MetropolisAccept(RealD dH, GridSerialRNG &sRNG)
{
  RealD rn_test;
  random(sRNG, rn_test);
  RealD prob = std::exp(-dH);
  std::cout << GridLogMessage << "MGHMC Metropolis dH " << dH << " exp(-dH) " << prob << " random " << rn_test << std::endl;
  return (prob > 1.0) || (rn_test <= prob);
}

NAMESPACE_END(Grid);
