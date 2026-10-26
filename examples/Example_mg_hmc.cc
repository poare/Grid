/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_hmc.cc

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
// Example_mg_hmc.cc
//
// Can multigrid pay for its setup inside HMC? Quenched Iwasaki HMC from a
// configuration; at every MD step an MG-preconditioned solve of
// A = PV^dag M (U) stands in for the fermion force solve, and every k MD steps
// the near-null basis is corrected to first order,
//     v -> v - w ,   (Q A Q) w = Q (A(U) - A(U_last)) v ,
// with the correction solves themselves MG-preconditioned, followed by a
// recoarsening. Every fine matvec is counted by category (setup, coarsen,
// force, correct, measure) so the cumulative cost can be set against plain
// GCR on the identical trajectory and sources (--solver gcr).
//
// Design: research hub, projects/multigrid/notes/deflation-basis-hmc.md.
//
#include "Example_mg_common.h"
#include "Example_mg_hmc.h"

using namespace std;
using namespace Grid;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);
  RealD t_start = usecond();

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

  int   Ntraj   = opt("--ntraj")   ? std::stoi(optS("--ntraj"))   : 3;
  RealD trajL   = opt("--trajL")   ? std::stod(optS("--trajL"))   : 1.0;
  int   MDsteps = opt("--mdsteps") ? std::stoi(optS("--mdsteps")) : 80;
  RealD beta    = opt("--beta")    ? std::stod(optS("--beta"))    : 2.13;

  int  updateEvery   = opt("--update-every")   ? std::stoi(optS("--update-every"))   : 4;   // 0 = never
  bool useGCR        = opt("--solver") && optS("--solver") == "gcr";
  int  baselineEvery = opt("--baseline-every") ? std::stoi(optS("--baseline-every")) : 1;
  CorrectionForm form = (opt("--correction") && optS("--correction") == "plain") ? CorrectionForm::Plain
                                                                                 : CorrectionForm::Projected;
  RealD epsForce     = opt("--eps-force")  ? std::stod(optS("--eps-force"))  : 1.0e-8;
  RealD epsCorr      = opt("--eps-corr")   ? std::stod(optS("--eps-corr"))   : 1.0e-2;
  int   forceSteps   = 1000;                   // outer-step cap, MG force solve
  int   corrSteps    = 50;                     // outer-step cap, correction solve
  // Coarse solve inside every MG application: GCR(10,10). Defaults are those of
  // Example_pvdagm_svd job 341271 on the same configuration.
  RealD coarseTol    = opt("--coarse-tol")      ? std::stod(optS("--coarse-tol"))      : 3.0e-2;
  int   coarseSteps  = opt("--coarse-maxsteps") ? std::stoi(optS("--coarse-maxsteps")) : 1000; // rounded up to a restart
  int   recoarsenEvery = opt("--recoarsen-every") ? std::stoi(optS("--recoarsen-every")) : 0;  // 0 = only after an update
  int   gcrSteps     = opt("--gcr-maxsteps") ? std::stoi(optS("--gcr-maxsteps")) : 20000;
  int   freshEvery   = opt("--fresh-every")  ? std::stoi(optS("--fresh-every"))  : 0;   // 0 = end of each trajectory
  int   rebreedIters = opt("--rebreed-iters") ? std::stoi(optS("--rebreed-iters")) : 0; // 0 = off
  RealD budgetHours  = opt("--budget-hours") ? std::stod(optS("--budget-hours")) : 19.0;
  bool  rejectAll    = opt("--reject-all");

  std::cout<<GridLogMessage<<"MGHMC-PARAMS nbasis "<<nbasis<<" Ntraj "<<Ntraj<<" trajL "<<trajL<<" MDsteps "<<MDsteps
	   <<" beta "<<beta<<" solver "<<(useGCR ? "gcr" : "mg")<<" update_every "<<updateEvery
	   <<" baseline_every "<<baselineEvery<<" correction "<<(form==CorrectionForm::Projected ? "projected" : "plain")
	   <<" eps_force "<<epsForce<<" eps_corr "<<epsCorr<<" fresh_every "<<freshEvery
	   <<" rebreed_iters "<<rebreedIters<<" reject_all "<<rejectAll
	   <<" coarse_tol "<<coarseTol<<" coarse_maxsteps "<<10*((coarseSteps+9)/10)
	   <<" recoarsen_every "<<recoarsenEvery<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // Grids, RNGs, gauge field (same seeds as Example_mg_evolution.cc, so the
  // trajectory is independent of the solver settings)
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
  std::vector<int> seedsB({13,14,15,16});   // basis-breeding noise, as in the evolution sweep
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers(seedsS);

  LatticeGaugeField Umu(UGrid);
  if (tepid) {
    SU<Nc>::TepidConfiguration(RNG4,Umu);
  } else {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,config);
  }
  LatticeGaugeField U0 = Umu;      // for d_U from the start
  LatticeGaugeField Ulast = Umu;   // field at the last basis update

  //////////////////////////////////////////////////////////////////////////
  // Operators. A and the smoother's shifted A are counted; the "last"
  // operators hold A(U_last) for the update's dA.
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

  MobiusFermionD Ddwf (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
  MobiusFermionD Dpv  (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
  MobiusFermionD DdwfL(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
  MobiusFermionD DpvL (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
  PVdagM_t        PVdagM (Ddwf ,Dpv );
  PVdagM_t        PVdagML(DdwfL,DpvL);
  ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv);
  CountingLinearOperator<LatticeFermion> A     (PVdagM ,counter);
  CountingLinearOperator<LatticeFermion> Alast (PVdagML,counter);
  CountingLinearOperator<LatticeFermion> Ashift(ShiftedPVdagM,counter);

  NextToNearestStencilGeometry5D geom(Coarse5d);
  TrivialPrecon<LatticeFermionD> simpleFine;
  TrivialPrecon<CoarseVector>    simpleCoarse;

  //////////////////////////////////////////////////////////////////////////
  // The MG hierarchy, rebuilt whenever the basis changes. GeneralCoarsenedMatrix
  // is single-use, so every rebuild constructs a fresh little Dirac operator.
  // Vraw is the basis as bred or updated; coarsening block-orthonormalises
  // the copy inside Agg, never Vraw.
  //////////////////////////////////////////////////////////////////////////
  Subspace Agg(Coarse5d,FGrid,cb);
  std::vector<LatticeFermion> Vraw;
  std::unique_ptr<LittleDiracOperator> LDOp;
  std::unique_ptr<CoarseOp>            LinOpCoarse;
  std::unique_ptr<CountingLinearOperator<CoarseVector>> CountedCoarse;
  std::unique_ptr<CoarseGCR>           L2PGCR;
  std::unique_ptr<FineGCR>             SmootherGCR;
  std::unique_ptr<TwoLevelMG>          MG;

  auto Rebuild = [&]() -> RealD {
    RealD t = -usecond();
    counter.Set("coarsen");
    Agg.subspace = Vraw;
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

  auto Breed = [&](Subspace &Into, std::vector<int> &seeds) -> RealD {
    RealD t = -usecond();
    GridParallelRNG RNGB(FGrid); RNGB.SeedFixedIntegers(seeds);
    Into.CreateSubspaceGCR(RNGB,A,nbasis);
    return (t+usecond())/1.0e6;
  };

  //////////////////////////////////////////////////////////////////////////
  // Setup at U_0 (MG path only)
  //////////////////////////////////////////////////////////////////////////
  Subspace Agg0(Coarse5d,FGrid,cb);   // block span of S_0, for drift
  if (!useGCR) {
    counter.Set("setup");
    RealD t_breed = Breed(Agg,seedsB);
    Vraw = Agg.subspace;
    Agg0.subspace = Vraw;
    Agg0.Orthogonalise();
    RealD t_coarsen = Rebuild();
    std::cout<<GridLogMessage<<"MGHMC-SETUP breed_s "<<t_breed<<" coarsen_s "<<t_coarsen
	     <<" setup_matvecs "<<counter.Get("setup")<<" coarsen_matvecs "<<counter.Get("coarsen")<<std::endl;
  }

  //////////////////////////////////////////////////////////////////////////
  // Quenched Iwasaki HMC, stepped by hand so work can happen at every MD step
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

  LatticeFermion eta(FGrid), x(FGrid), r(FGrid);
  int globalStep = 0;     // counts every MD step of every trajectory, accepted or not
  int nUpdates = 0;

  // One stand-in force solve at the current field; returns outer steps
  RealD t_recoarsen = 0.0;  // seconds spent recoarsening before this step's solve
  auto ForceSolve = [&](int traj, int step) {
    GridParallelRNG RNGeta(FGrid); RNGeta.SeedFixedIntegers(std::vector<int>({21,22,23,globalStep}));
    gaussian(RNGeta,eta);
    counter.Set("force");
    uint64_t c0 = counter.Get("force");
    uint64_t cc0 = counter.GetCoarse("force");
    int calls0 = useGCR ? 0 : MG->calls;
    RealD t = -usecond();
    int iters;
    RealD tsm = 0, tco = 0;
    x = Zero();
    if (useGCR) {
      FineGCR GCR(epsForce,(gcrSteps+15)/16,A,simpleFine,16,16);
      GCR(eta,x);
      iters = GCR.Steps();
    } else {
      RealD sm0 = MG->tsmooth, co0 = MG->tcoarse;
      FineGCR GCR(epsForce,(forceSteps+15)/16,A,*MG,16,16);
      GCR.Level(1);
      GCR(eta,x);
      iters = GCR.Steps();
      tsm = (MG->tsmooth-sm0)/1.0e6; tco = (MG->tcoarse-co0)/1.0e6;
    }
    t = (t+usecond())/1.0e6;
    uint64_t mv = counter.Get("force") - c0;
    uint64_t cmv = counter.GetCoarse("force") - cc0;
    int ccalls = useGCR ? 0 : MG->calls - calls0;
    counter.Set("measure");
    A.Op(x,r); r = eta - r;
    RealD relres = std::sqrt(norm2(r)/norm2(eta));
    std::cout<<GridLogMessage<<"MGHMC-STEP "<<traj<<" "<<step<<" "<<LinkDistance(U0,Umu)<<" "<<LinkDistance(Ulast,Umu)
	     <<" "<<WilsonLoops<Gimpl>::avgPlaquette(Umu)<<" "<<iters<<" "<<mv<<" "<<t<<" "<<tsm<<" "<<tco<<" "<<relres
	     <<" "<<cmv<<" "<<ccalls<<" "<<t_recoarsen<<std::endl;
    return iters;
  };

  // Ground truth: a fresh basis at the current field, from the sweep's noise
  auto FreshCheck = [&](int traj, int step) {
    counter.Set("measure");
    Subspace AggF(Coarse5d,FGrid,cb);
    Breed(AggF,seedsB);
    std::vector<LatticeFermion> Sfresh = AggF.subspace;
    Subspace AggV(Coarse5d,FGrid,cb);
    AggV.subspace = Vraw;
    AggV.Orthogonalise();
    RealD cap_cur = ReportCapture("fresh by current",AggV,Sfresh,nbasis);
    RealD cap_S0  = ReportCapture("fresh by S0",Agg0,Sfresh,nbasis);
    std::cout<<GridLogMessage<<"MGHMC-FRESH "<<traj<<" "<<step<<" "<<cap_cur<<" "<<cap_S0<<std::endl;
  };

  RealD t_last_traj = 0.0;
  for (int traj = 1; traj <= Ntraj; traj++) {

    RealD elapsed = (usecond()-t_start)/1.0e6;
    if ( traj > 1 && elapsed + t_last_traj > budgetHours*3600.0 ) {
      std::cout<<GridLogMessage<<"MGHMC stopping before traj "<<traj<<": elapsed "<<elapsed
	       <<" s + last trajectory "<<t_last_traj<<" s exceeds budget "<<budgetHours<<" h"<<std::endl;
      break;
    }
    RealD t_traj = -usecond();

    // Snapshot, restored if the trajectory is rejected
    LatticeGaugeField Usnap = Umu, UlastSnap = Ulast;
    std::vector<LatticeFermion> Vsnap = Vraw;

    MD.refresh(Umu,sRNG,RNG4);
    RealD H0 = MD.S(Umu);

    IntegrateWithHook(MD,Umu,[&](int step) {
      globalStep++;
      Ddwf.ImportGauge(Umu);
      Dpv.ImportGauge(Umu);

      bool solveNow = !useGCR || (globalStep % baselineEvery == 0);
      if (!solveNow) return;

      // Galerkin coarse operator of A(U) from the unchanged basis; matvecs count as coarsen
      t_recoarsen = 0.0;
      if (!useGCR && recoarsenEvery > 0 && globalStep % recoarsenEvery == 0) t_recoarsen = Rebuild();
      int iters = ForceSolve(traj,step);

      if (useGCR) return;

      if (rebreedIters > 0 && iters > rebreedIters) {
        counter.Set("setup");
        Breed(Agg,seedsB);
        Vraw = Agg.subspace;
        Rebuild();
        Ulast = Umu; DdwfL.ImportGauge(Umu); DpvL.ImportGauge(Umu);
        std::cout<<GridLogMessage<<"MGHMC-REBREED "<<traj<<" "<<step<<std::endl;
        return;
      }

      if (updateEvery > 0 && step % updateEvery == 0) {
        std::vector<LatticeFermion> Vprev = Vraw;
        uint64_t cc0 = counter.GetCoarse("correct");
        int calls0 = MG->calls;
        UpdateStats st = PerturbativeUpdate(Vraw,A,Alast,*MG,epsCorr,corrSteps,form,counter);
        uint64_t cmv = counter.GetCoarse("correct") - cc0;
        int ccalls = MG->calls - calls0;    // before Rebuild() replaces MG
        RealD t_coarsen = 0.0, dist2 = 0.0, gcond = NAN, cap0 = NAN;
        if (!st.skipped) {
          counter.Set("measure");
          ReportNulliness("updated traj "+std::to_string(traj)+" step "+std::to_string(step),Vraw,A,nbasis);
          ReportAngles("update traj "+std::to_string(traj)+" step "+std::to_string(step),Vprev,Vraw,nbasis,dist2);
          gcond = SpanProjector<LatticeFermion>(Vraw).GramCondition();
          cap0  = ReportCapture("updated by S0",Agg0,Vraw,nbasis);
          t_coarsen = Rebuild();
          Ulast = Umu; DdwfL.ImportGauge(Umu); DpvL.ImportGauge(Umu);
        }
        nUpdates++;
        std::cout<<GridLogMessage<<"MGHMC-UPDATE "<<traj<<" "<<step<<" skipped "<<st.skipped<<" capped "<<st.capped
		 <<" matvecs "<<st.matvecs<<" seconds "<<st.seconds<<" coarsen_s "<<t_coarsen
		 <<" resid_before "<<st.resid_before<<" resid_after "<<st.resid_after
		 <<" cap_S0 "<<cap0<<" dist2_prev "<<dist2<<" gram_cond "<<gcond
		 <<" coarse_matvecs "<<cmv<<" coarse_calls "<<ccalls;
        std::cout<<" iters";  for (auto i : st.iters)  std::cout<<" "<<i;
        std::cout<<" wratio"; for (auto w : st.wratio) std::cout<<" "<<w;
        std::cout<<std::endl;
        if (freshEvery > 0 && nUpdates % freshEvery == 0) FreshCheck(traj,step);
      }
    });

    RealD H1 = MD.S(Umu);
    RealD dH = H1 - H0;
    bool accepted = rejectAll ? false : MetropolisAccept(dH,sRNG);
    bool restored = false;
    if (!accepted) {
      Umu = Usnap; Ulast = UlastSnap;
      Ddwf.ImportGauge(Umu);  Dpv.ImportGauge(Umu);
      DdwfL.ImportGauge(Ulast); DpvL.ImportGauge(Ulast);
      if (!useGCR) { Vraw = Vsnap; Rebuild(); }
      restored = true;
    }
    if (!useGCR && freshEvery == 0) FreshCheck(traj,MDsteps);

    t_traj = (t_traj+usecond())/1.0e6;
    t_last_traj = t_traj;
    std::cout<<GridLogMessage<<"MGHMC-TRAJ "<<traj<<" dH "<<dH<<" accepted "<<accepted<<" restored "<<restored
	     <<" seconds "<<t_traj;
    for (auto &kv : counter.counts) std::cout<<" cum_"<<kv.first<<" "<<kv.second;
    for (auto &kv : counter.coarse) std::cout<<" cum_coarse_"<<kv.first<<" "<<kv.second;
    std::cout<<std::endl;
  }

  counter.Report("MGHMC-CUM");
  Grid_finalize();
  return 0;
}
