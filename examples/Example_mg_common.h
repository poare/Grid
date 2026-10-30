/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_common.h

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
// Example_mg_common.h
//
// Helpers shared by the MG near-null basis examples (Example_mg_evolution.cc,
// Example_mg_hmc.cc): the PV^dag M operators, the two-level MG preconditioner,
// the sloppy subspace refresh, and the basis-geometry measurements.
//
#pragma once

#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>

#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidual.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include "Example_mg_refine.h"

NAMESPACE_BEGIN(Grid);

//////////////////////////////////////////////////////////////////////////////
// PV^dag M operator and shifted variant (copied from Example_pvdagm_svd.cc).
// PV^dag M is the Pauli-Villars preconditioned operator: its low modes are the
// physical near-null space of the domain wall operator with the bulk (heavy)
// modes cancelled, and it is nearest-neighbour enough to coarsen well.
//////////////////////////////////////////////////////////////////////////////
template<class Matrix,class Field>
class PVdagMLinearOperator : public LinearOperatorBase<Field> {
  Matrix &_Mat;
  Matrix &_PV;
public:
  PVdagMLinearOperator(Matrix &Mat,Matrix &PV): _Mat(Mat),_PV(PV){};

  void OpDiag (const Field &in, Field &out) {    assert(0);  }
  void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
  void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
  void Op     (const Field &in, Field &out){
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
  }
  void AdjOp     (const Field &in, Field &out){
    Field tmp(in.Grid());
    _PV.M(in,tmp);
    _Mat.Mdag(tmp,out);
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){
    HermOp(in,out);
    ComplexD dot = innerProduct(in,out);
    n1=real(dot);
    n2=norm2(out);
  }
  void HermOp(const Field &in, Field &out){
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
};

template<class Matrix,class Field>
class ShiftedPVdagMLinearOperator : public LinearOperatorBase<Field> {
  Matrix &_Mat;
  Matrix &_PV;
  RealD shift;
public:
  ShiftedPVdagMLinearOperator(RealD _shift,Matrix &Mat,Matrix &PV): shift(_shift),_Mat(Mat),_PV(PV){};

  void OpDiag (const Field &in, Field &out) {    assert(0);  }
  void OpDir  (const Field &in, Field &out,int dir,int disp) {    assert(0);  }
  void OpDirAll  (const Field &in, std::vector<Field> &out){    assert(0);  };
  void Op     (const Field &in, Field &out){
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    _PV.Mdag(tmp,out);
    out = out + shift * in;
  }
  void AdjOp     (const Field &in, Field &out){
    Field tmp(in.Grid());
    _PV.M(tmp,out);
    _Mat.Mdag(in,tmp);
    out = out + shift * in;
  }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){    assert(0);  }
  void HermOp(const Field &in, Field &out){
    Field tmp(in.Grid());
    Op(in,tmp);
    AdjOp(tmp,out);
  }
};

//////////////////////////////////////////////////////////////////////////////
// Two level (fine + coarse) V-cycle preconditioner
// (copied from Example_pvdagm_svd.cc): pre-smooth, coarse-grid correction
// through the aggregated subspace, post-smooth.
//////////////////////////////////////////////////////////////////////////////
template<class Fobj,class CComplex,int nbasis>
class MGPreconditioner : public LinearFunction< Lattice<Fobj> > {
public:
  using LinearFunction<Lattice<Fobj> >::operator();

  typedef Aggregation<Fobj,CComplex,nbasis> Aggregates;
  typedef typename Aggregation<Fobj,CComplex,nbasis>::FineField    FineField;
  typedef typename Aggregation<Fobj,CComplex,nbasis>::CoarseVector CoarseVector;
  typedef typename Aggregation<Fobj,CComplex,nbasis>::CoarseMatrix CoarseMatrix;
  typedef LinearOperatorBase<FineField>                            FineOperator;
  typedef LinearFunction    <FineField>                            FineSmoother;
  typedef LinearOperatorBase<CoarseVector>                         CoarseOperator;
  typedef LinearFunction    <CoarseVector>                         CoarseSolver;
  Aggregates     & _Aggregates;
  FineOperator   & _FineOperator;
  FineSmoother   & _PreSmoother;
  FineSmoother   & _PostSmoother;
  CoarseOperator & _CoarseOperator;
  CoarseSolver   & _CoarseSolve;

  int    level;  void Level(int lv) {level = lv; };

  // Accumulated wall time (us) per phase, across every application
  RealD tsmooth=0, tcoarse=0, tproj=0, tfine=0;
  int   calls=0;

  MGPreconditioner(Aggregates &Agg,
		   FineOperator &Fine,
		   FineSmoother &PreSmoother,
		   FineSmoother &PostSmoother,
		   CoarseOperator &CoarseOperator_,
		   CoarseSolver &CoarseSolve_)
    : _Aggregates(Agg),
      _FineOperator(Fine),
      _PreSmoother(PreSmoother),
      _PostSmoother(PostSmoother),
      _CoarseOperator(CoarseOperator_),
      _CoarseSolve(CoarseSolve_),
      level(1)  {  }

  virtual void operator()(const FineField &in, FineField & out)
  {
    GridBase *CoarseGrid = _Aggregates.CoarseGrid;
    CoarseVector Csrc(CoarseGrid);
    CoarseVector Csol(CoarseGrid);
    FineField vec1(in.Grid());
    FineField vec2(in.Grid());

    calls++;

    // Pre-smooth on the fine grid
    out = Zero();
    tsmooth-=usecond();
    _PreSmoother(in,out);
    tsmooth+=usecond();

    // Residual after smoothing
    tfine-=usecond();
    _FineOperator.Op(out,vec1);  sub(vec1, in ,vec1);
    tfine+=usecond();

    // Coarse grid correction: project residual to the coarse space,
    // solve the little Dirac equation, prolongate back and add.
    tproj-=usecond();
    _Aggregates.ProjectToSubspace  (Csrc,vec1);
    tproj+=usecond();
    Csol = Zero();
    tcoarse-=usecond();
    _CoarseSolve(Csrc,Csol);
    tcoarse+=usecond();
    tproj-=usecond();
    _Aggregates.PromoteFromSubspace(Csol,vec1);
    tproj+=usecond();
    add(out,out,vec1);

    // Residual after coarse correction, then post-smooth
    tfine-=usecond();
    _FineOperator.Op(out,vec1);  sub(vec1 ,in , vec1);
    tfine+=usecond();
    vec2=Zero();
    tsmooth-=usecond();
    _PostSmoother(vec1,vec2);
    tsmooth+=usecond();
    add( out,out,vec2);
  }
};

// Subspace refresh by inverse iteration (strategy (c)): RefineBasisGCR on the
// vectors of an Aggregation.
template<class Aggregates, class Field>
std::vector<int> RefineSubspaceGCR(Aggregates &Agg,
		       LinearOperatorBase<Field> &DiracOp,
		       int nn,
		       RealD tol, int maxouter)
{
  return RefineBasisGCR(Agg.subspace,DiracOp,nn,tol,maxouter);
}

//////////////////////////////////////////////////////////////////////////////
// "Nulliness" of the subspace S = span{v_b} against A = PV^dag M (U_t):
//
//     eta(S) = tr(V^dag A^dag A V)/N ,   V^dag V = I ,
//
// a property of S alone (GL(N)-invariant), bounded below by
// lambda_min(A^dag A) by interlacing. No orthonormal V is formed: with
// G = <v_i|v_j> and M = <A v_i|A v_j> for the stored vectors, the spectrum
// of V^dag A^dag A V is that of the generalised problem M x = mu G x. The
// subspace itself is not modified -- it is the live MG basis.
//
// Per-vector lines are instruments only (basis-dependent): the Rayleigh
// quotient rho = <v|A|v>, |Av|^2, and the pseudo-eigen residual
// eps^2 = |Av|^2 - |rho|^2. Small |rho| is NOT a target; see
// notes/deflation-basis-evolution.md section 1.
//
// Pass the basis as bred, NOT an Aggregation after CoarsenOperator: that
// block-orthonormalises in place, which changes the global span. Returns eta(S).
//////////////////////////////////////////////////////////////////////////////
template<class Field>
RealD ReportNulliness(const std::string &tag,
		      std::vector<Field> &V,
		      LinearOperatorBase<Field> &DiracOp,
		      int nn)
{
  std::vector<Field> Av(nn,V[0].Grid());
  Eigen::MatrixXcd G(nn,nn);
  Eigen::MatrixXcd M(nn,nn);

  for(int b=0;b<nn;b++){
    DiracOp.Op(V[b],Av[b]);
  }
  for(int i=0;i<nn;i++){
    for(int j=i;j<nn;j++){
      G(i,j) = innerProduct(V[i],V[j]);
      M(i,j) = innerProduct(Av[i],Av[j]);
      G(j,i) = std::conj(G(i,j));
      M(j,i) = std::conj(M(i,j));
    }
  }

  // Per-vector instruments, normalised by |v|^2 = G(b,b)
  for(int b=0;b<nn;b++){
    RealD    vv   = real(G(b,b));
    ComplexD rho  = innerProduct(V[b],Av[b])/vv;             // <v|A|v>/<v|v>
    RealD    aa   = real(M(b,b))/vv;                         // |Av|^2/|v|^2
    RealD    eps2 = aa - norm(rho);                          // pseudo-eigen residual^2

    std::cout<<GridLogMessage<<"MGEVO-NULL "<<tag<<" ["<<b<<"] rho "<<rho
	     <<" |Av|^2 "<<aa<<" eps^2 "<<eps2
	     <<" |Av|/|rho| "<<std::sqrt(aa)/abs(rho)<<std::endl;
  }

  // Conditioning of the stored basis, so a near-dependent set is visible
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> gs(G,Eigen::EigenvaluesOnly);
  std::cout<<GridLogMessage<<"MGEVO-NULL "<<tag<<" Gram eigenvalues ["
	   <<gs.eigenvalues().minCoeff()<<", "<<gs.eigenvalues().maxCoeff()<<"]"<<std::endl;

  // Spectrum of V^dag A^dag A V on an orthonormal basis of S
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> ges(M,G,Eigen::EigenvaluesOnly);
  Eigen::VectorXd mu = ges.eigenvalues();
  for(int b=0;b<nn;b++){
    std::cout<<GridLogMessage<<"MGEVO-ETA "<<tag<<" mu["<<b<<"] "<<mu(b)<<std::endl;
  }
  RealD eta = mu.sum()/nn;
  std::cout<<GridLogMessage<<"MGEVO-ETA "<<tag<<" eta(S) = tr(V^dag A^dag A V)/N = "<<eta<<std::endl;
  return eta;
}

//////////////////////////////////////////////////////////////////////////////
// Principal angles between S_A = span{a_i} and S_B = span{b_j}, N each.
//
// With G_A = <a_i|a_j> = L_A L_A^dag (Cholesky), Q_A = A L_A^{-dag} is an
// orthonormal basis of S_A, and Q_A^dag Q_B = L_A^{-1} C L_B^{-dag} with
// C_ij = <a_i|b_j>. Its singular values are cos(theta_k). Neither basis needs
// to be orthonormal, and neither is modified.
//
// Prints every angle (degrees) and the squared projection distance
// dist2 = sum_k sin^2(theta_k) = N - |Q_A^dag Q_B|_F^2, which keeps grading
// the separation after theta_max has saturated near 90. Returns theta_max.
//////////////////////////////////////////////////////////////////////////////
template<class Field>
RealD ReportAngles(const std::string &tag,
		   std::vector<Field> &A,
		   std::vector<Field> &B,
		   int nn,
		   RealD &dist2)
{
  Eigen::MatrixXcd GA(nn,nn);
  Eigen::MatrixXcd GB(nn,nn);
  Eigen::MatrixXcd C (nn,nn);
  for(int i=0;i<nn;i++){
    for(int j=i;j<nn;j++){
      GA(i,j) = innerProduct(A[i],A[j]);  GA(j,i) = std::conj(GA(i,j));
      GB(i,j) = innerProduct(B[i],B[j]);  GB(j,i) = std::conj(GB(i,j));
    }
    for(int j=0;j<nn;j++){
      C(i,j) = innerProduct(A[i],B[j]);
    }
  }

  Eigen::LLT<Eigen::MatrixXcd> LA(GA);
  Eigen::LLT<Eigen::MatrixXcd> LB(GB);
  if ( LA.info()!=Eigen::Success || LB.info()!=Eigen::Success ) {
    std::cout<<GridLogMessage<<"MGEVO-ANGLE "<<tag<<" Gram matrix not positive definite; angles skipped"<<std::endl;
    dist2 = NAN;
    return NAN;
  }
  Eigen::MatrixXcd X = LA.matrixL().solve(C);                       // L_A^{-1} C
  Eigen::MatrixXcd W = LB.matrixL().solve(X.adjoint()).adjoint();   // L_A^{-1} C L_B^{-dag}

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(W);
  Eigen::VectorXd cs = svd.singularValues();   // descending, so angles ascend

  RealD thmax = 0.0;
  dist2 = 0.0;
  for(int k=0;k<nn;k++){
    RealD c  = std::min(cs(k),1.0);
    RealD th = std::acos(c)*180.0/M_PI;
    dist2 += 1.0-c*c;
    thmax  = std::max(thmax,th);
    std::cout<<GridLogMessage<<"MGEVO-ANGLE "<<tag<<" theta["<<k<<"] "<<th<<" deg"<<std::endl;
  }
  std::cout<<GridLogMessage<<"MGEVO-ANGLE "<<tag<<" theta_max "<<thmax<<" deg  dist2 "<<dist2<<std::endl;
  return thmax;
}

//////////////////////////////////////////////////////////////////////////////
// Captured fraction of each w in W by the block span of Agg:
//     f(w) = |P w|^2 / |w|^2 ,  P = PromoteFromSubspace o ProjectToSubspace,
// an orthogonal projection only if Agg.subspace is block-orthonormal (true
// after CoarsenOperator or Agg.Orthogonalise()). This is the coarse space
// the MG actually uses, unlike the global span seen by eta and the angles.
// Returns the mean over W.
//////////////////////////////////////////////////////////////////////////////
template<class Aggregates, class Field>
RealD ReportCapture(const std::string &tag, Aggregates &Agg, std::vector<Field> &W, int nn)
{
  typename Aggregates::CoarseVector c(Agg.CoarseGrid);
  Field wp(W[0].Grid());
  RealD mean=0.0, fmin=1.0;
  for(int b=0;b<nn;b++){
    Agg.ProjectToSubspace(c,W[b]);
    Agg.PromoteFromSubspace(c,wp);
    RealD f = norm2(wp)/norm2(W[b]);
    std::cout<<GridLogMessage<<"MGEVO-CAPTURE "<<tag<<" ["<<b<<"] "<<f<<std::endl;
    mean += f/nn;
    fmin  = std::min(fmin,f);
  }
  std::cout<<GridLogMessage<<"MGEVO-CAPTURE "<<tag<<" mean "<<mean<<" min "<<fmin<<std::endl;
  return mean;
}

//////////////////////////////////////////////////////////////////////////////
// Gauge distance from the starting field, averaged over links:
//     d_U = 1 - < Re tr(U_0^dag U_t) > / Nc ,
// zero for U_t = U_0. A common axis for runs with different trajectory
// lengths, since short trajectories with a momentum refresh between them
// random-walk where one long trajectory moves ballistically.
//////////////////////////////////////////////////////////////////////////////
RealD LinkDistance(LatticeGaugeField &U0, LatticeGaugeField &Ut)
{
  RealD s = 0.0;
  for(int mu=0;mu<Nd;mu++){
    LatticeColourMatrix u0 = PeekIndex<LorentzIndex>(U0,mu);
    LatticeColourMatrix ut = PeekIndex<LorentzIndex>(Ut,mu);
    s += real(TensorRemove(sum(trace(adj(u0)*ut))));
  }
  return 1.0 - s/(Nd*Nc*(RealD)U0.Grid()->gSites());
}

//////////////////////////////////////////////////////////////////////////////
// Read one field written in SciDAC format (as the Krylov-Schur drivers write
// their eigenvectors, evec0 ... evecN). Needs LIME.
//////////////////////////////////////////////////////////////////////////////
template<class Field>
void ReadScidacField(Field &out, const std::string &fname)
{
#ifdef HAVE_LIME
  std::cout<<GridLogMessage<<"Reads at: "<<fname<<std::endl;
  emptyUserRecord record;
  ScidacReader SR;
  SR.open(fname);
  SR.readScidacFieldRecord(out,record);
  SR.close();
#else
  std::cout<<GridLogError<<"ReadScidacField: Grid built without LIME, cannot read "<<fname<<std::endl;
  assert(0);
#endif
}

NAMESPACE_END(Grid);
