/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_hmc_gcrupdate.cc

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
// Example_mg_hmc_gcrupdate.cc
//
// Can a near-null basis be kept current through an MD trajectory by one cheap
// pass of inverse iteration per vector at every step? The basis S_0 is bred at
// U_0 and the gauge field follows trajectory 1 of Example_mg_hmc.cc (same seeds,
// same dtau). At every MD step (every --refine-every steps) the CURRENT basis is
// refreshed in place,
//     v -> A(U)^{-1} v / |A(U)^{-1} v| ,
// by one unpreconditioned GCR(10,10) solve per vector to tolerance tol with at
// most maxsteps steps (RefineSubspaceGCR); the coarse operator is then rebuilt
// at U and one MG-preconditioned force solve is done with the source of
// Example_mg_hmc.cc at that step. One (tol, maxsteps) setting per run.
//
// Every step logs the refresh's cost (wall time, fine matvecs, GCR steps per
// vector), the basis's health (nulliness, invariance residual before and after
// the refresh, Gram condition number) and the force solve. At the --fresh-at
// steps a fresh basis is also bred at U as ground truth: its block-span capture
// by the current basis, the principal angles between the two, the current
// basis's drift from S_0, and one force solve with the fresh basis for the best
// achievable cost.
//
// Design: research hub, projects/multigrid (deflation-basis-hmc).
//
#include "Example_mg_common.h"
#include "Example_mg_hmc.h"

using namespace std;
using namespace Grid;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  auto opt  = [&](const char *name){ return GridCmdOptionExists(argv,argv+argc,name); };
  auto optS = [&](const char *name){ return GridCmdOptionPayload(argv,argv+argc,name); };

  //////////////////////////////////////////////////////////////////////////
  // Parameters
  //////////////////////////////////////////////////////////////////////////
  const int Ls     = 16;
  const int nbasis = 40;
  const int cb     = 0;
  RealD mass = 0.001, M5 = 1.8, mob_b = 1.5, mob_c = 0.5;

  std::string config("/Users/patrickoare/libraries/PETSc-Grid/ckpoint_lat.4000");
  if (opt("--config")) config = optS("--config");
  std::vector<int> lat_size {16, 16, 16, 32};
  if (opt("--lat")) GridCmdOptionIntVector(optS("--lat"),lat_size);
  bool tepid = opt("--tepid");

  // Default dtau = 0.025, as in the MG-HMC runs, so step s is the same field
  RealD trajL   = opt("--trajL")   ? std::stod(optS("--trajL"))   : 0.4;
  int   MDsteps = opt("--mdsteps") ? std::stoi(optS("--mdsteps")) : 16;
  RealD beta    = opt("--beta")    ? std::stod(optS("--beta"))    : 2.13;

  // Refresh setting "tol:maxsteps"; maxsteps rounds up to a GCR(10,10) restart
  std::string refineSpec = opt("--refine") ? optS("--refine") : std::string("1e-2:20");
  auto colon = refineSpec.find(':');
  assert(colon != std::string::npos);
  RealD refineTol   = std::stod(refineSpec.substr(0,colon));
  int   refineSteps = 10*((std::stoi(refineSpec.substr(colon+1))+9)/10);
  int   refineEvery = opt("--refine-every") ? std::stoi(optS("--refine-every")) : 1;

  std::vector<int> freshAt {1, 2, 4, 8, 16};
  if (opt("--fresh-at")) GridCmdOptionIntVector(optS("--fresh-at"),freshAt);   // dotted, e.g. 1.2.4.8.16

  RealD epsForce     = opt("--eps-force")       ? std::stod(optS("--eps-force"))       : 1.0e-8;
  int   forceSteps   = 1000;                                                                    // outer-step cap
  RealD coarseTol    = opt("--coarse-tol")      ? std::stod(optS("--coarse-tol"))      : 1.0e-1;
  int   coarseSteps  = opt("--coarse-maxsteps") ? std::stoi(optS("--coarse-maxsteps")) : 200;  // rounded up to a restart

  std::cout<<GridLogMessage<<"GCRUPD-PARAMS nbasis "<<nbasis<<" trajL "<<trajL<<" MDsteps "<<MDsteps<<" beta "<<beta
	   <<" refine_tol "<<refineTol<<" refine_maxsteps "<<refineSteps<<" refine_every "<<refineEvery
	   <<" eps_force "<<epsForce<<" coarse_tol "<<coarseTol<<" coarse_maxsteps "<<10*((coarseSteps+9)/10)
	   <<" fresh_at";
  for (auto s : freshAt) std::cout<<" "<<s;
  std::cout<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // Grids, RNGs, gauge field (seeds as in Example_mg_hmc.cc)
  //////////////////////////////////////////////////////////////////////////
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(lat_size, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  Coordinate clatt = lat_size;
  for(int d=0;d<clatt.size();d++) clatt[d] = clatt[d]/2;
  GridCartesian *Coarse4d = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridCartesian *Coarse5d = SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seedsS({9,10,11,12});
  std::vector<int> seedsB({13,14,15,16});   // basis-breeding noise, as in the MG-HMC runs
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers(seedsS);

  LatticeGaugeField Umu(UGrid);
  if (tepid) {
    SU<Nc>::TepidConfiguration(RNG4,Umu);
  } else {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,config);
  }
  LatticeGaugeField U0 = Umu;

  //////////////////////////////////////////////////////////////////////////
  // Operator and MG pieces (as in Example_mg_hmc.cc)
  //////////////////////////////////////////////////////////////////////////
  typedef PVdagMLinearOperator       <MobiusFermionD,LatticeFermionD> PVdagM_t;
  typedef ShiftedPVdagMLinearOperator<MobiusFermionD,LatticeFermionD> ShiftedPVdagM_t;
  typedef Aggregation<vSpinColourVector,vTComplex,nbasis>             Subspace;
  typedef DeprecatedGeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis>  LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector                           CoarseVector;
  typedef MGPreconditioner<vSpinColourVector,vTComplex,nbasis>        TwoLevelMG;
  typedef NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> CoarseOp;
  typedef PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>  CoarseGCR;
  typedef PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion> FineGCR;

  MatvecCounter counter;

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
  PVdagM_t        PVdagM(Ddwf,Dpv);
  ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv);
  CountingLinearOperator<LatticeFermion> A     (PVdagM ,counter);
  CountingLinearOperator<LatticeFermion> Ashift(ShiftedPVdagM,counter);

  NextToNearestStencilGeometry5D geom(Coarse5d);
  TrivialPrecon<LatticeFermion>  simpleFine;
  TrivialPrecon<CoarseVector>    simpleCoarse;

  Subspace Agg(Coarse5d,FGrid,cb);
  std::unique_ptr<LittleDiracOperator> LDOp;
  std::unique_ptr<CoarseOp>            LinOpCoarse;
  std::unique_ptr<CountingLinearOperator<CoarseVector>> CountedCoarse;
  std::unique_ptr<CoarseGCR>           L2PGCR;
  std::unique_ptr<FineGCR>             SmootherGCR;
  std::unique_ptr<TwoLevelMG>          MG;

  // Coarsen A(U) on the basis V; Agg.subspace becomes its block-orthonormal copy
  auto Rebuild = [&](const std::vector<LatticeFermion> &V) -> RealD {
    RealD t = -usecond();
    counter.Set("coarsen");
    Agg.subspace = V;
    MG.reset(); SmootherGCR.reset(); L2PGCR.reset(); CountedCoarse.reset(); LinOpCoarse.reset(); LDOp.reset();
    LDOp.reset(new LittleDiracOperator(geom,FGrid,Coarse5d));
    LDOp->CoarsenOperator(A,Agg);
    LinOpCoarse.reset(new CoarseOp(*LDOp));
    CountedCoarse.reset(new CountingLinearOperator<CoarseVector>(*LinOpCoarse,counter,true));
    L2PGCR.reset(new CoarseGCR(coarseTol,(coarseSteps+9)/10,*CountedCoarse,simpleCoarse,10,10));
    L2PGCR->Level(3);
    SmootherGCR.reset(new FineGCR(0.01,1,Ashift,simpleFine,16,16));
    SmootherGCR->Level(2);
    MG.reset(new TwoLevelMG(Agg,A,simpleFine,*SmootherGCR,*CountedCoarse,*L2PGCR));
    return (t+usecond())/1.0e6;
  };

  auto Breed = [&](Subspace &Into) -> RealD {
    RealD t = -usecond();
    GridParallelRNG RNGB(FGrid); RNGB.SeedFixedIntegers(seedsB);
    Into.CreateSubspaceGCR(RNGB,A,nbasis);
    return (t+usecond())/1.0e6;
  };

  //////////////////////////////////////////////////////////////////////////
  // S_0 at U_0; V is the live basis, refreshed in place at every step
  //////////////////////////////////////////////////////////////////////////
  counter.Set("setup");
  Subspace AggS0(Coarse5d,FGrid,cb);
  RealD t_breed0 = Breed(AggS0);
  std::vector<LatticeFermion> S0 = AggS0.subspace;
  std::vector<LatticeFermion> V  = S0;
  Subspace Agg0(Coarse5d,FGrid,cb);   // block span of S_0, for drift
  Agg0.subspace = S0;
  Agg0.Orthogonalise();
  std::cout<<GridLogMessage<<"GCRUPD-SETUP breed_s "<<t_breed0<<" setup_matvecs "<<counter.Get("setup")<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // One MG force solve on the current Agg/MG
  //////////////////////////////////////////////////////////////////////////
  LatticeFermion eta(FGrid), x(FGrid), r(FGrid);

  struct SolveStats { int iters; uint64_t matvecs, coarse_matvecs; RealD seconds, t_smooth, t_coarse, relres; };
  auto MGSolve = [&]() -> SolveStats {
    SolveStats st;
    counter.Set("force");
    uint64_t c0 = counter.Get("force"), cc0 = counter.GetCoarse("force");
    RealD sm0 = MG->tsmooth, co0 = MG->tcoarse;
    x = Zero();
    RealD t = -usecond();
    FineGCR GCR(epsForce,(forceSteps+15)/16,A,*MG,16,16);
    GCR.Level(1);
    GCR(eta,x);
    st.seconds  = (t+usecond())/1.0e6;
    st.iters    = GCR.Steps();
    st.matvecs  = counter.Get("force") - c0;
    st.coarse_matvecs = counter.GetCoarse("force") - cc0;
    st.t_smooth = (MG->tsmooth-sm0)/1.0e6;
    st.t_coarse = (MG->tcoarse-co0)/1.0e6;
    counter.Set("measure");
    A.Op(x,r); r = eta - r;
    st.relres = std::sqrt(norm2(r)/norm2(eta));
    return st;
  };

  //////////////////////////////////////////////////////////////////////////
  // Quenched Iwasaki MD along trajectory 1 of Example_mg_hmc.cc
  //////////////////////////////////////////////////////////////////////////
  typedef PeriodicGimplR Gimpl;
  typedef NoSmearing<Gimpl> SmearingPolicy;
  typedef MinimumNorm2<Gimpl,SmearingPolicy> IntegratorType;
  IwasakiGaugeActionR GaugeAction(beta);
  ActionLevel<LatticeGaugeField> Level1(1);
  Level1.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField,NoHirep> FullSet;
  FullSet.push_back(Level1);
  IntegratorParameters MDpar(MDsteps,trajL);
  SmearingPolicy Smearing;
  IntegratorType MD(UGrid,MDpar,FullSet,Smearing);

  MD.refresh(Umu,sRNG,RNG4);
  RealD t_total = 0.0;

  IntegrateWithHook(MD,Umu,[&](int step) {
    Ddwf.ImportGauge(Umu);
    Dpv.ImportGauge(Umu);
    GridParallelRNG RNGeta(FGrid); RNGeta.SeedFixedIntegers(std::vector<int>({21,22,23,step}));
    gaussian(RNGeta,eta);
    std::string tag = "step "+std::to_string(step);

    // Refresh the live basis in place
    counter.Set("measure");
    RealD res_before = InvarianceResidual(V,A);
    RealD t_refresh = 0.0;
    uint64_t mv_refresh = 0;
    std::vector<int> steps;
    if (step % refineEvery == 0) {
      counter.Set("refresh");
      uint64_t m0 = counter.Get("refresh");
      Subspace AggR(Coarse5d,FGrid,cb);
      AggR.subspace = V;
      t_refresh = -usecond();
      steps = RefineSubspaceGCR(AggR,A,nbasis,refineTol,refineSteps/10);
      t_refresh = (t_refresh+usecond())/1.0e6;
      mv_refresh = counter.Get("refresh") - m0;
      V = AggR.subspace;
    }
    counter.Set("measure");
    RealD res_after = InvarianceResidual(V,A);
    RealD gcond     = SpanProjector<LatticeFermion>(V).GramCondition();
    RealD eta_V     = ReportNulliness("gcrupd "+tag+" live",V,A,nbasis);

    // Recoarsen at U on the live basis, then the force solve
    RealD t_coarsen = Rebuild(V);
    SolveStats st = MGSolve();

    RealD smean = 0.0; int smax = 0;
    for (auto s : steps) { smean += (RealD)s/steps.size(); smax = std::max(smax,s); }
    std::cout<<GridLogMessage<<"GCRUPD-STEP "<<step<<" "<<LinkDistance(U0,Umu)
	     <<" "<<t_refresh<<" "<<mv_refresh<<" "<<(steps.empty() ? NAN : smean)<<" "<<smax
	     <<" "<<res_before<<" "<<res_after<<" "<<gcond<<" "<<eta_V<<" "<<t_coarsen
	     <<" "<<st.iters<<" "<<st.matvecs<<" "<<st.coarse_matvecs<<" "<<st.seconds
	     <<" "<<st.t_smooth<<" "<<st.t_coarse<<" "<<st.relres<<std::endl;
    if (!steps.empty()) {
      std::cout<<GridLogMessage<<"GCRUPD-STEPS "<<step;
      for (auto s : steps) std::cout<<" "<<s;
      std::cout<<std::endl;
    }
    t_total += t_refresh + t_coarsen + st.seconds;   // the algorithm's cost, diagnostics excluded

    // Ground truth at selected steps: a fresh basis at U
    if (std::find(freshAt.begin(),freshAt.end(),step) != freshAt.end()) {
      counter.Set("measure");
      Subspace AggF(Coarse5d,FGrid,cb);
      RealD t_fresh = Breed(AggF);
      std::vector<LatticeFermion> Sfresh = AggF.subspace;
      RealD cap_live  = ReportCapture("fresh by live "+tag,Agg,Sfresh,nbasis);   // Agg: block-orthonormal live basis
      RealD cap_S0    = ReportCapture("live by S0 "+tag,Agg0,V,nbasis);
      RealD dist2     = 0.0;
      RealD thmax     = ReportAngles("live vs fresh "+tag,V,Sfresh,nbasis,dist2);
      RealD res_fresh = InvarianceResidual(Sfresh,A);
      RealD eta_fresh = ReportNulliness("gcrupd "+tag+" fresh",Sfresh,A,nbasis);
      RealD t_cf      = Rebuild(Sfresh);
      SolveStats sf   = MGSolve();
      std::cout<<GridLogMessage<<"GCRUPD-FRESH "<<step<<" "<<t_fresh<<" "<<cap_live<<" "<<cap_S0
	       <<" "<<thmax<<" "<<dist2<<" "<<res_fresh<<" "<<eta_fresh<<" "<<t_cf
	       <<" "<<sf.iters<<" "<<sf.matvecs<<" "<<sf.coarse_matvecs<<" "<<sf.seconds<<" "<<sf.relres<<std::endl;
    }
  });

  std::cout<<GridLogMessage<<"GCRUPD-TOTAL seconds "<<t_total<<std::endl;
  counter.Report("GCRUPD-CUM");
  Grid_finalize();
  return 0;
}
