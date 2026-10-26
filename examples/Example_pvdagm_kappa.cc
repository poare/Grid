/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvdagm_kappa.cc

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
// Example_pvdagm_kappa.cc
//
// Eigenvalue condition numbers of the low modes of A = PV^dag M, and how well
// first-order perturbation theory moves those modes under a small MD step.
//
// 1. Read N right eigenvectors psi_i of A(U_0) (evec0 ... evec{N-1}, SciDAC,
//    as written by the Krylov-Schur drivers) and measure lambda_i = <psi|A|psi>
//    and the residual |A psi - lambda psi|.
// 2. Left eigenvectors without a second eigensolve. With Gamma = gamma_5 R_5,
//    M^dag = Gamma M Gamma and PV^dag = Gamma PV Gamma, so
//        A^dag = Gamma (M PV^dag) Gamma = Gamma PV^{dag,-1} A PV^dag Gamma ,
//    and if A psi_j = mu psi_j then phi = Gamma PV^{dag,-1} psi_j obeys
//    A^dag phi = mu phi. The left eigenvector of lambda_i (A^dag phi = lambda_i^* phi)
//    is therefore built from the partner psi_j with lambda_j = lambda_i^*; the
//    spectrum comes in conjugate pairs. The left residual checks the relation.
//    kappa_i = |phi_i| |psi_i| / |phi_i^dag psi_i|.
// 3. Principal angles between the right and left low-mode spaces: whether
//    Galerkin coarsening (R = P^dag) is adequate or Petrov-Galerkin is needed.
// 4. For each step size eps: one MinimumNorm2 step of the quenched Iwasaki MD
//    from U_0 (same momentum every time), dA = A(U') - A(U_0), and for the
//    lowest nmodes modes:
//      - the predicted eigenvalue shift  dlambda = phi^dag dA psi / phi^dag psi ;
//      - the residual under A(U') of psi itself (stale), and of psi corrected by
//        the per-vector update  A w = Q dA psi  (plain) and by the projected
//        update  (QAQ) w = Q dA psi  (projected), psi -> psi - w, with Q the
//        complement of the span of all N eigenvectors and the solves
//        MG-preconditioned (PerturbativeUpdate).
//    First-order exact updates should leave a residual O(eps^2) where the stale
//    vector has O(eps); halving eps tests the order.
//
// Design: research hub, projects/multigrid/notes/deflation-basis-evolution.md
// (section 2, the per-vector formulation, and the kappa(lambda) measurement).
//
#include "Example_mg_common.h"
#include "Example_mg_hmc.h"

using namespace std;
using namespace Grid;

template<class FAction>
FAction *MakeAction(LatticeGaugeField &U, GridCartesian *FGrid, GridRedBlackCartesian *FrbGrid,
		    GridCartesian *UGrid, GridRedBlackCartesian *UrbGrid, RealD mass, RealD M5, RealD b, RealD c)
{
  if constexpr (std::is_same<FAction,MobiusFermionD>::value) {
    return new MobiusFermionD(U,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
  } else {
    return new DomainWallFermionD(U,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  }
}

template<class FAction>
int Run(int argc, char **argv)
{
  auto opt  = [&](const char *name){ return GridCmdOptionExists(argv,argv+argc,name); };
  auto optS = [&](const char *name){ return GridCmdOptionPayload(argv,argv+argc,name); };

  //////////////////////////////////////////////////////////////////////////
  // Parameters
  //////////////////////////////////////////////////////////////////////////
  const int Ls     = 16;
  const int nbasis = 40;     // MG basis size, as in the MG-HMC runs
  const int cb     = 0;
  RealD mass = 0.001, M5 = 1.8, mob_b = 1.5, mob_c = 0.5;

  std::string config("/sdcc/u/poare/PETSc-Grid/ckpoint_lat.4000");
  if (opt("--config")) config = optS("--config");
  std::string evecDir("/hpcgpfs01/work/lqcd/staging/RBC/ckpoint_lat.4000/ks_evecs/PVdagM_Nm80_Nk40_Niter5000_m0p001_339143");
  if (opt("--evec-dir")) evecDir = optS("--evec-dir");
  int   Nev     = opt("--nev")    ? std::stoi(optS("--nev"))    : 40;   // eigenvectors read
  int   nmodes  = opt("--nmodes") ? std::stoi(optS("--nmodes")) : 8;    // modes perturbed
  std::vector<int> lat_size {16, 16, 16, 32};
  if (opt("--lat")) GridCmdOptionIntVector(optS("--lat"),lat_size);

  // MD step sizes; each is one MinimumNorm2 step from U_0 with the same momentum
  std::vector<RealD> epsList {0.025, 0.0125, 0.00625};
  if (opt("--eps")) {
    epsList.clear();
    std::stringstream ss(optS("--eps"));
    std::string item;
    while (std::getline(ss,item,',')) epsList.push_back(std::stod(item));
  }
  RealD beta       = opt("--beta")       ? std::stod(optS("--beta"))       : 2.13;
  RealD pairTol    = opt("--pair-tol")   ? std::stod(optS("--pair-tol"))   : 1.0e-6;  // |lambda_j - lambda_i^*| / |lambda_i|
  RealD pvTol      = 1.0e-10;
  RealD updateTol  = opt("--update-tol") ? std::stod(optS("--update-tol")) : 1.0e-6;
  int   updateSteps= opt("--update-maxsteps") ? std::stoi(optS("--update-maxsteps")) : 400;
  RealD coarseTol  = opt("--coarse-tol") ? std::stod(optS("--coarse-tol")) : 1.0e-1;
  int   coarseSteps= opt("--coarse-maxsteps") ? std::stoi(optS("--coarse-maxsteps")) : 200;
  bool  mgFromEvecs= opt("--mg-basis") && optS("--mg-basis") == "evecs";
  bool  skipPerturb= opt("--skip-perturb");
  nmodes = std::min(nmodes,Nev);

  std::cout<<GridLogMessage<<"KAPPA-PARAMS action "<<(std::is_same<FAction,MobiusFermionD>::value ? "mobius" : "dwf")
	   <<" mass "<<mass<<" M5 "<<M5<<" Ls "<<Ls<<" Nev "<<Nev<<" nmodes "<<nmodes<<" evec_dir "<<evecDir
	   <<" pair_tol "<<pairTol<<" update_tol "<<updateTol<<" update_maxsteps "<<updateSteps
	   <<" coarse_tol "<<coarseTol<<" coarse_maxsteps "<<10*((coarseSteps+9)/10)
	   <<" mg_basis "<<(mgFromEvecs ? "evecs" : "bred")<<" eps";
  for (auto e : epsList) std::cout<<" "<<e;
  std::cout<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // Grids, gauge field, operators at U_0
  //////////////////////////////////////////////////////////////////////////
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(lat_size, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  Coordinate clatt = lat_size;
  for(int d=0;d<clatt.size();d++) clatt[d] = clatt[d]/2;
  GridCartesian *Coarse4d = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridCartesian *Coarse5d = SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  LatticeGaugeField U0(UGrid);
  {
    FieldMetaData header;
    NerscIO::readConfiguration(U0,header,config);
  }
  LatticeGaugeField U1 = U0;

  typedef PVdagMLinearOperator       <FAction,LatticeFermionD> PVdagM_t;
  typedef ShiftedPVdagMLinearOperator<FAction,LatticeFermionD> ShiftedPVdagM_t;

  MatvecCounter counter;
  std::unique_ptr<FAction> D0 (MakeAction<FAction>(U0,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,mob_b,mob_c));
  std::unique_ptr<FAction> PV0(MakeAction<FAction>(U0,FGrid,FrbGrid,UGrid,UrbGrid,1.0 ,M5,mob_b,mob_c));
  std::unique_ptr<FAction> D1 (MakeAction<FAction>(U1,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,mob_b,mob_c));
  std::unique_ptr<FAction> PV1(MakeAction<FAction>(U1,FGrid,FrbGrid,UGrid,UrbGrid,1.0 ,M5,mob_b,mob_c));
  PVdagM_t        PVdagM0(*D0,*PV0);
  PVdagM_t        PVdagM1(*D1,*PV1);
  ShiftedPVdagM_t ShiftedPVdagM0(0.01,*D0,*PV0);
  CountingLinearOperator<LatticeFermion> A0    (PVdagM0,counter);
  CountingLinearOperator<LatticeFermion> A1    (PVdagM1,counter);
  CountingLinearOperator<LatticeFermion> Ashift(ShiftedPVdagM0,counter);

  //////////////////////////////////////////////////////////////////////////
  // 1. Right eigenvectors and their eigenvalues
  //////////////////////////////////////////////////////////////////////////
  counter.Set("measure");
  std::vector<LatticeFermion> Psi(Nev,FGrid);
  std::vector<ComplexD> lam(Nev);
  LatticeFermion tmp(FGrid), Av(FGrid);
  for (int i = 0; i < Nev; i++) {
    ReadScidacField(Psi[i],evecDir+"/evec"+std::to_string(i));
    Psi[i] = Psi[i] * std::pow(norm2(Psi[i]),-0.5);
    A0.Op(Psi[i],Av);
    lam[i] = innerProduct(Psi[i],Av);
    tmp = Av - lam[i]*Psi[i];
    std::cout<<GridLogMessage<<"KAPPA-RIGHT "<<i<<" "<<real(lam[i])<<" "<<imag(lam[i])
	     <<" "<<std::sqrt(norm2(tmp))<<" "<<std::sqrt(norm2(tmp))/std::abs(lam[i])<<std::endl;
  }

  //////////////////////////////////////////////////////////////////////////
  // 2. Left eigenvectors phi_i = Gamma PV^{dag,-1} psi_partner(i), and kappa
  //////////////////////////////////////////////////////////////////////////
  // PV^{dag,-1} x = PV (PV^dag PV)^{-1} x, by CG on the well-conditioned PV^dag PV
  MdagMLinearOperator<FAction,LatticeFermionD> PVdagPV(*PV0);
  ConjugateGradient<LatticeFermionD> CGpv(pvTol,10000);

  std::vector<LatticeFermion> Phi(Nev,FGrid);
  std::vector<int>   partner(Nev,-1);
  std::vector<RealD> kappa(Nev,NAN);
  LatticeFermion y(FGrid), Adv(FGrid);
  for (int i = 0; i < Nev; i++) {
    RealD best = 1.0e300;
    for (int j = 0; j < Nev; j++) {
      RealD d = std::abs(lam[j] - std::conj(lam[i])) / std::abs(lam[i]);
      if (d < best) { best = d; partner[i] = j; }
    }
    if (best > pairTol) partner[i] = -1;
    if (partner[i] < 0) {
      std::cout<<GridLogMessage<<"KAPPA-LEFT "<<i<<" unpaired "<<best<<std::endl;
      continue;
    }
    y = Zero();
    CGpv(PVdagPV,Psi[partner[i]],y);
    PV0->M(y,tmp);                 // PV^{dag,-1} psi_j
    G5R5(Phi[i],tmp);
    Phi[i] = Phi[i] * std::pow(norm2(Phi[i]),-0.5);

    A0.AdjOp(Phi[i],Adv);
    tmp = Adv - std::conj(lam[i])*Phi[i];
    RealD lres = std::sqrt(norm2(tmp));
    kappa[i] = 1.0 / std::abs(innerProduct(Phi[i],Psi[i]));   // both unit norm
    std::cout<<GridLogMessage<<"KAPPA-LEFT "<<i<<" "<<partner[i]<<" "<<best
	     <<" "<<lres<<" "<<lres/std::abs(lam[i])<<" "<<kappa[i]<<std::endl;
  }

  //////////////////////////////////////////////////////////////////////////
  // 3. Right vs left low-mode spaces, over the paired modes
  //////////////////////////////////////////////////////////////////////////
  {
    std::vector<LatticeFermion> R, L;
    for (int i = 0; i < Nev; i++) if (partner[i] >= 0) { R.push_back(Psi[i]); L.push_back(Phi[i]); }
    RealD dist2 = 0.0;
    RealD thmax = ReportAngles("right vs left",R,L,(int)R.size(),dist2);
    std::cout<<GridLogMessage<<"KAPPA-SPACES npaired "<<R.size()<<" theta_max "<<thmax<<" dist2 "<<dist2<<std::endl;
  }

  if (skipPerturb) {
    counter.Report("KAPPA-CUM");
    return 0;
  }

  //////////////////////////////////////////////////////////////////////////
  // MG at U_0 for the correction solves (as in Example_mg_hmc.cc)
  //////////////////////////////////////////////////////////////////////////
  typedef Aggregation<vSpinColourVector,vTComplex,nbasis>             Subspace;
  typedef DeprecatedGeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis>  LittleDiracOperator;
  typedef typename LittleDiracOperator::CoarseVector                  CoarseVector;
  typedef MGPreconditioner<vSpinColourVector,vTComplex,nbasis>        TwoLevelMG;
  typedef NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> CoarseOp;
  typedef PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>  CoarseGCR;
  typedef PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion> FineGCR;

  NextToNearestStencilGeometry5D geom(Coarse5d);
  TrivialPrecon<LatticeFermion>  simpleFine;
  TrivialPrecon<CoarseVector>    simpleCoarse;
  Subspace Agg(Coarse5d,FGrid,cb);
  RealD t_setup = -usecond();
  counter.Set("setup");
  if (mgFromEvecs) {
    for (int i = 0; i < nbasis; i++) Agg.subspace[i] = Psi[i % Nev];
  } else {
    GridParallelRNG RNGB(FGrid); RNGB.SeedFixedIntegers(std::vector<int>({13,14,15,16}));
    Agg.CreateSubspaceGCR(RNGB,A0,nbasis);
  }
  counter.Set("coarsen");
  LittleDiracOperator LDOp(geom,FGrid,Coarse5d);
  LDOp.CoarsenOperator(A0,Agg);
  CoarseOp LinOpCoarse(LDOp);
  CountingLinearOperator<CoarseVector> CountedCoarse(LinOpCoarse,counter,true);
  CoarseGCR L2PGCR(coarseTol,(coarseSteps+9)/10,CountedCoarse,simpleCoarse,10,10);
  L2PGCR.Level(3);
  FineGCR SmootherGCR(0.01,1,Ashift,simpleFine,16,16);
  SmootherGCR.Level(2);
  TwoLevelMG MG(Agg,A0,simpleFine,SmootherGCR,CountedCoarse,L2PGCR);
  t_setup = (t_setup+usecond())/1.0e6;
  std::cout<<GridLogMessage<<"KAPPA-SETUP seconds "<<t_setup<<" setup_matvecs "<<counter.Get("setup")
	   <<" coarsen_matvecs "<<counter.Get("coarsen")<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // 4. One MD step of size eps from U_0, and first-order updates per mode
  //////////////////////////////////////////////////////////////////////////
  typedef PeriodicGimplR Gimpl;
  typedef NoSmearing<Gimpl> SmearingPolicy;
  typedef MinimumNorm2<Gimpl,SmearingPolicy> IntegratorType;
  IwasakiGaugeActionR GaugeAction(beta);
  ActionLevel<LatticeGaugeField> Level1(1);
  Level1.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField,NoHirep> FullSet;
  FullSet.push_back(Level1);
  SmearingPolicy Smearing;

  // Residual of x under A(U') with its Rayleigh quotient
  auto Residual = [&](const LatticeFermion &x, ComplexD &theta) -> RealD {
    LatticeFermion xn(FGrid), Ax(FGrid), r(FGrid);
    xn = x * std::pow(norm2(x),-0.5);
    A1.Op(xn,Ax);
    theta = innerProduct(xn,Ax);
    r = Ax - theta*xn;
    return std::sqrt(norm2(r));
  };

  for (auto eps : epsList) {
    // Same momentum for every eps: reseed before each refresh
    GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(std::vector<int>({1,2,3,4}));
    GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers(std::vector<int>({9,10,11,12}));
    IntegratorParameters MDpar(1,eps);
    IntegratorType MD(UGrid,MDpar,FullSet,Smearing);
    U1 = U0;
    MD.refresh(U1,sRNG,RNG4);
    IntegrateWithHook(MD,U1,[](int){});
    D1->ImportGauge(U1);
    PV1->ImportGauge(U1);
    std::cout<<GridLogMessage<<"KAPPA-EPS "<<eps<<" dU "<<LinkDistance(U0,U1)<<std::endl;

    // Per-vector and projected updates of the first nmodes eigenvectors, Q over all Nev
    std::vector<LatticeFermion> Vplain = Psi, Vproj = Psi;
    UpdateStats sp = PerturbativeUpdate(Vplain,A1,A0,MG,updateTol,updateSteps,CorrectionForm::Plain,    counter,nmodes);
    UpdateStats sq = PerturbativeUpdate(Vproj, A1,A0,MG,updateTol,updateSteps,CorrectionForm::Projected,counter,nmodes);
    for (auto &pr : std::vector<std::pair<std::string,UpdateStats*> >{{"plain",&sp},{"projected",&sq}}) {
      std::cout<<GridLogMessage<<"KAPPA-UPDATE "<<eps<<" "<<pr.first<<" seconds "<<pr.second->seconds
	       <<" matvecs "<<pr.second->matvecs<<" capped "<<pr.second->capped<<" iters";
      for (auto it : pr.second->iters) std::cout<<" "<<it;
      std::cout<<" wratio";
      for (auto w : pr.second->wratio) std::cout<<" "<<w;
      std::cout<<std::endl;
    }

    counter.Set("measure");
    for (int i = 0; i < nmodes; i++) {
      // Predicted shift: phi^dag dA psi / phi^dag psi
      ComplexD dlam_pred(NAN,NAN);
      if (partner[i] >= 0) {
	LatticeFermion a1(FGrid), a0(FGrid);
	A1.Op(Psi[i],a1);
	A0.Op(Psi[i],a0);
	a1 = a1 - a0;
	dlam_pred = innerProduct(Phi[i],a1) / innerProduct(Phi[i],Psi[i]);
      }
      ComplexD th_stale, th_plain, th_proj;
      RealD r_stale = Residual(Psi[i],  th_stale);
      RealD r_plain = Residual(Vplain[i],th_plain);
      RealD r_proj  = Residual(Vproj[i], th_proj);
      std::cout<<GridLogMessage<<"KAPPA-PERT "<<eps<<" "<<i<<" "<<kappa[i]
	       <<" "<<real(dlam_pred)<<" "<<imag(dlam_pred)
	       <<" "<<real(th_stale-lam[i])<<" "<<imag(th_stale-lam[i])
	       <<" "<<real(th_plain-lam[i])<<" "<<imag(th_plain-lam[i])
	       <<" "<<real(th_proj -lam[i])<<" "<<imag(th_proj -lam[i])
	       <<" "<<r_stale<<" "<<r_plain<<" "<<r_proj<<std::endl;
    }
  }

  counter.Report("KAPPA-CUM");
  return 0;
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);
  bool mobius = GridCmdOptionExists(argv,argv+argc,"--action") &&
                GridCmdOptionPayload(argv,argv+argc,"--action") == "mobius";
  int rc = mobius ? Run<MobiusFermionD>(argc,argv) : Run<DomainWallFermionD>(argc,argv);
  Grid_finalize();
  return rc;
}
