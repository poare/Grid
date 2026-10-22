/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_evolution.cc

    Copyright (C) 2026

Author: Patrick Oare

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
//
// Example_mg_evolution.cc
//
// Experiment: how does a multigrid near-null (deflation) subspace S move as the
// gauge field evolves under HMC, and how cheaply can it be kept up to date?
//
// Motivation: to use multigrid inside HMC we cannot afford to regenerate the
// near-null basis of the Dirac operator at every gauge update. Since each HMC
// trajectory changes U by a smooth perturbation, the near-null space of the
// updated operator should be close to that of the old one.
//
// PRIMARY measurement -- geometry of S, on bases as bred (before coarsening,
// which block-orthonormalises in place and changes the global span):
//
//   S0          : bred at U_0.
//   S_t^fresh   : re-bred at U_t from the SAME noise as S0 (the breeding RNG
//                 is re-seeded identically), so it differs from S0 only
//                 through the operator. A re-breed at U_0 gives the floor.
//   S_t^refined : the previous refined basis after one sloppy GCR
//                 inverse-iteration pass at U_t.
//
//   per trajectory: the captured fraction |P w|^2/|w|^2 of each w in
//   S_t^fresh by the BLOCK span of S0, S_t^refined, S_{t-1}^fresh and
//   S_t^fresh itself -- the block span is the coarse space the MG uses.
//   At U_0 a basis bred from different noise sets the same-U ceiling and
//   Gaussian vectors the floor. Also logged, on the GLOBAL span: principal
//   angles and eta(S). Job 424581 (trajL = 1) found both saturated after one
//   trajectory while the MG solves survived, which is why capture leads.
//   d_U = 1 - <Re tr(U_0^dag U_t)>/Nc is the common axis across trajL.
//
// SECONDARY measurement -- outer GCR steps of a 2-level MG solve of
// PV^dag M (U_t) x = src to 1e-8, capped at outerMaxRestarts*16 steps:
//
//   (b) RECOARSEN  : S0 frozen, coarse operator rebuilt at U_t.
//   (c) REFINE     : S_t^refined, coarse operator rebuilt at U_t.
//   (d) FRESH      : S_t^fresh, i.e. setup from scratch at U_t.
//
// The earlier rung (a), the coarse operator frozen at U_0, stalled at relres
// ~ 0.09 after a single trajectory (job 424322) and is dropped. nbasis is
// small because the question is how S moves, not MG performance; solve
// counts at small nbasis compress the spread between strategies.
//
// The MG construction (PVdagM operator, 2-level preconditioner, GCR subspace
// setup, tolerances, 2^4 blocking) follows examples/Example_pvdagm_svd.cc.
//
// Gauge evolution: by default a QUENCHED Iwasaki HMC (see notes at the action
// setup below) — smooth O(dt) evolution of U is all that is required for this
// test. A dynamical pseudofermion action can be added at the marked location.
//
#include "Example_mg_common.h"

using namespace std;
using namespace Grid;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);
  RealD t_start = usecond();

  //////////////////////////////////////////////////////////////////////////
  // Parameters
  //////////////////////////////////////////////////////////////////////////
  const int Ls=16;
  std::vector<int> lat_size {16, 16, 16, 32};   // ckpoint_lat.4000 is 16^3 x 32

  RealD mass = 0.001;              // light quark mass
  RealD M5   = 1.8;                // domain wall height
  RealD mob_b= 1.5;                // Mobius kernel: b-c=1 (Shamir-like), b+c=2
  RealD mob_c= 0.5;

  std::string config("/Users/patrickoare/libraries/PETSc-Grid/ckpoint_lat.4000");
  if (GridCmdOptionExists(argv,argv+argc,"--config")) {
    config = GridCmdOptionPayload(argv,argv+argc,"--config");
  }
  std::cout<<GridLogMessage<<"Reading configuration "<<config<<std::endl;

  // Stop starting new trajectories once the next one would cross this budget
  RealD budgetHours = 19.0;
  if (GridCmdOptionExists(argv,argv+argc,"--budget-hours")) {
    budgetHours = std::stod(GridCmdOptionPayload(argv,argv+argc,"--budget-hours"));
  }

  const int nbasis = 8;            // small: the question is how S moves, not MG performance
  const int cb     = 0;

  int   Ntraj   = 10;              // HMC trajectories; enough for the trend
  RealD beta    = 2.13;            // Iwasaki coupling for the (quenched) evolution -- still a guess
  RealD trajL   = 1.0;             // trajectory length
  if (GridCmdOptionExists(argv,argv+argc,"--ntraj")) {
    Ntraj = std::stoi(GridCmdOptionPayload(argv,argv+argc,"--ntraj"));
  }
  if (GridCmdOptionExists(argv,argv+argc,"--trajL")) {
    trajL = std::stod(GridCmdOptionPayload(argv,argv+argc,"--trajL"));
  }
  // MD steps per trajectory: by default keep dt = 1/80 fixed across trajectory
  // lengths (80 steps at trajL = 1 gave dH <~ 0.3, job 424581), at least 4
  int   MDsteps = std::max(4,(int)std::lround(80.0*trajL));
  if (GridCmdOptionExists(argv,argv+argc,"--mdsteps")) {
    MDsteps = std::stoi(GridCmdOptionPayload(argv,argv+argc,"--mdsteps"));
  }
  std::cout<<GridLogMessage<<"MGEVO parameters: nbasis "<<nbasis<<" Ntraj "<<Ntraj
	   <<" trajL "<<trajL<<" MDsteps "<<MDsteps<<" beta "<<beta<<std::endl;

  RealD refineTol      = 1.0e-2;   // strategy (c): sloppy tolerance ...
  int   refineMaxOuter = 2;        // ... and restart budget for the refresh GCR(10,10)

  int   outerMaxRestarts = 13;     // outer GCR(16,16) cap: 13*16 = 208 steps, ~8x the fresh baseline

  //////////////////////////////////////////////////////////////////////////
  // Grids: fine 5d, and a coarse grid blocked 2^4 in the 4d volume with the
  // whole of Ls aggregated (coarse Ls = 1), as in Example_pvdagm_svd.cc.
  //////////////////////////////////////////////////////////////////////////
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(lat_size, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  Coordinate clatt = lat_size;
  for(int d=0;d<clatt.size();d++) clatt[d] = clatt[d]/2;

  GridCartesian *Coarse4d =  SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridCartesian *Coarse5d =  SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  std::vector<int> seedsS({9,10,11,12});
  std::vector<int> seedsB({13,14,15,16});  // basis-breeding noise, re-seeded for every fresh basis
  std::vector<int> seedsB2({17,18,19,20}); // different noise, for the same-U capture ceiling at t=0
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers(seedsS);

  //////////////////////////////////////////////////////////////////////////
  // Gauge field: start from the checkpointed configuration
  //////////////////////////////////////////////////////////////////////////
  LatticeGaugeField Umu(UGrid);
  FieldMetaData header;
  NerscIO::readConfiguration(Umu,header,config);
  LatticeGaugeField U0 = Umu;      // kept for the gauge distance d_U

  //////////////////////////////////////////////////////////////////////////
  // Fixed fermion source, reused for every measurement so that iteration
  // counts are comparable across trajectories and strategies.
  //////////////////////////////////////////////////////////////////////////
  LatticeFermion src(FGrid); gaussian(RNG5,src);
  LatticeFermion result(FGrid);

  //////////////////////////////////////////////////////////////////////////
  // MG types (as Example_pvdagm_svd.cc, with Mobius kernel)
  //////////////////////////////////////////////////////////////////////////
  typedef PVdagMLinearOperator       <MobiusFermionD,LatticeFermionD> PVdagM_t;
  typedef ShiftedPVdagMLinearOperator<MobiusFermionD,LatticeFermionD> ShiftedPVdagM_t;
  typedef Aggregation<vSpinColourVector,vTComplex,nbasis> Subspace;
  typedef GeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;
  typedef MGPreconditioner<vSpinColourVector,vTComplex,nbasis> TwoLevelMG;

  NextToNearestStencilGeometry5D geom(Coarse5d);

  TrivialPrecon<LatticeFermionD> simpleFine;
  TrivialPrecon<CoarseVector>    simpleCoarse;

  //////////////////////////////////////////////////////////////////////////
  // Breed a basis by GCR inverse iteration on noise. The noise RNG is
  // re-seeded identically every call, so every fresh basis is the same
  // filter applied to the same noise: differences between fresh bases at
  // U_s and U_t come from the operator, not from sampling.
  //////////////////////////////////////////////////////////////////////////
  auto BreedBasis = [&](Subspace &Agg, PVdagM_t &Op, std::vector<int> &seeds) -> RealD
  {
    GridParallelRNG RNGB(FGrid); RNGB.SeedFixedIntegers(seeds);
    RealD t = -usecond();
    Agg.CreateSubspaceGCR(RNGB,Op,nbasis);
    return (t+usecond())/1.0e6;
  };

  //////////////////////////////////////////////////////////////////////////
  // Coarsen Agg against FineOp into a fresh little Dirac operator (it is
  // single-use: CoarsenOperator leaves _A on the padded grid, so a second
  // call on the same object fails conformable()), then do one 2-level
  // preconditioned solve of PV^dag M (U_t) x = src to 1e-8, capped at
  // outerMaxRestarts*16 outer steps. Returns the outer step count; wall
  // times and the true relative residual come back by reference, so a
  // stalled solve is a data point rather than a hang.
  //
  // Coarsening block-orthonormalises Agg.subspace in place.
  //////////////////////////////////////////////////////////////////////////
  auto CoarsenAndSolve = [&](const std::string &tag,
			     Subspace            &Aggregates,
			     PVdagM_t            &FineOp,
			     ShiftedPVdagM_t     &SmootherOp,
			     RealD               &tcoarsen,
			     RealD               &tsolve,
			     RealD               &relres) -> int
  {
    LittleDiracOperator LittleDiracOp(geom,FGrid,Coarse5d);
    tcoarsen = -usecond();
    LittleDiracOp.CoarsenOperator(FineOp,Aggregates);
    tcoarsen = (tcoarsen+usecond())/1.0e6;

    NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> LinOpCoarse(LittleDiracOp);
    PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>   L2PGCR(3.0e-2,100,LinOpCoarse,simpleCoarse,10,10);
    L2PGCR.Level(3);

    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> SmootherGCR(0.01,1,SmootherOp,simpleFine,16,16);
    SmootherGCR.Level(2);

    TwoLevelMG TwoLevelPrecon(Aggregates,FineOp,
			      simpleFine,SmootherGCR,
			      LinOpCoarse,L2PGCR);

    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermion> L1PGCR(1.0e-8,outerMaxRestarts,FineOp,TwoLevelPrecon,16,16);
    L1PGCR.Level(1);

    result = Zero();
    tsolve = -usecond();
    L1PGCR(src,result);
    tsolve = (tsolve+usecond())/1.0e6;

    LatticeFermion r(FGrid);
    FineOp.Op(result,r);
    r = src - r;
    relres = std::sqrt(norm2(r)/norm2(src));

    std::cout<<GridLogMessage<<"MGEVO solve "<<tag<<" : outer steps = "<<L1PGCR.steps
	     <<" relres "<<relres<<" time "<<tsolve<<" s"<<std::endl;
    std::cout<<GridLogMessage<<"MGEVO-PRECTIME "<<tag<<" calls "<<TwoLevelPrecon.calls
	     <<" smooth "<<TwoLevelPrecon.tsmooth/1.0e6
	     <<" coarse "<<TwoLevelPrecon.tcoarse/1.0e6
	     <<" proj "  <<TwoLevelPrecon.tproj/1.0e6
	     <<" fineop "<<TwoLevelPrecon.tfine/1.0e6<<std::endl;
    return L1PGCR.steps;
  };

  //////////////////////////////////////////////////////////////////////////
  // Initial setup at U_0.
  //
  //  - S0          : the basis bred at U_0, kept as bred (never coarsened)
  //  - SfreshPrev  : the previous trajectory's fresh basis, as bred
  //  - AggregatesFrozen  : S0, coarsened afresh each trajectory, rung (b)
  //  - AggregatesRefined : evolves via RefineSubspaceGCR,         rung (c)
  //  - AggregatesFresh   : re-bred from the same noise at U_t,     rung (d)
  //  - AggregatesFreshPrev : previous trajectory's fresh basis, block-
  //                          orthonormalised, for the per-trajectory capture
  //////////////////////////////////////////////////////////////////////////
  MobiusFermionD Ddwf0(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
  MobiusFermionD Dpv0 (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
  PVdagM_t        PVdagM0(Ddwf0,Dpv0);
  ShiftedPVdagM_t ShiftedPVdagM0(0.01,Ddwf0,Dpv0);

  Subspace AggregatesFrozen   (Coarse5d,FGrid,cb);
  Subspace AggregatesRefined  (Coarse5d,FGrid,cb);
  Subspace AggregatesFresh    (Coarse5d,FGrid,cb);
  Subspace AggregatesFreshPrev(Coarse5d,FGrid,cb);

  RealD t0_setup = BreedBasis(AggregatesFrozen,PVdagM0,seedsB);
  std::vector<LatticeFermion> S0         = AggregatesFrozen.subspace;
  std::vector<LatticeFermion> SfreshPrev = S0;
  std::vector<LatticeFermion> SrefinedPrev = S0;   // refined basis as refined, never block-orthonormalised

  // Reproducibility check: re-breeding at U_0 from the same noise should
  // give theta ~ 0. Anything larger is the floor for every angle below.
  RealD t0_rebreed = BreedBasis(AggregatesFresh,PVdagM0,seedsB);
  RealD dist0;
  RealD th0 = ReportAngles("traj 0 rebreed vs S0",S0,AggregatesFresh.subspace,nbasis,dist0);

  RealD null0 = ReportNulliness("traj 0 S0",S0,PVdagM0,nbasis);

  // Baseline: fresh setup solving at U_0
  RealD t0_coarsen, t0_solve, relres0;
  int steps0 = CoarsenAndSolve("traj 0 (fresh setup)",AggregatesFrozen,PVdagM0,ShiftedPVdagM0,
			       t0_coarsen,t0_solve,relres0);

  //////////////////////////////////////////////////////////////////////////
  // Capture controls at U_0, all against the block span of S0 (now block-
  // orthonormal after the coarsening above):
  //   self    : S0 itself                         -> 1 by construction
  //   other   : a basis bred from DIFFERENT noise -> same-U ceiling
  //   random  : Gaussian vectors                  -> floor, ~ nbasis/(block dof)
  // Every later capture sits between floor and ceiling; staleness is the
  // drop below the ceiling, not below 1.
  //////////////////////////////////////////////////////////////////////////
  RealD t0_other = -usecond();
  Subspace AggregatesOther(Coarse5d,FGrid,cb);
  BreedBasis(AggregatesOther,PVdagM0,seedsB2);
  t0_other = (t0_other+usecond())/1.0e6;

  std::vector<LatticeFermion> Srandom(nbasis,FGrid);
  for(int b=0;b<nbasis;b++) gaussian(RNG5,Srandom[b]);

  RealD cap0_self   = ReportCapture("traj 0 self   S0 by S0",   AggregatesFrozen,S0,                     nbasis);
  RealD cap0_other  = ReportCapture("traj 0 other  noise2 by S0",AggregatesFrozen,AggregatesOther.subspace,nbasis);
  RealD cap0_random = ReportCapture("traj 0 random gauss by S0", AggregatesFrozen,Srandom,                nbasis);

  AggregatesFreshPrev.subspace = AggregatesFrozen.subspace;

  // Wall times in seconds:
  //   MGEVO-TIME 0 setup rebreed other coarsen solve
  std::cout<<GridLogMessage<<"MGEVO-TIME 0 setup "<<t0_setup<<" rebreed "<<t0_rebreed<<" other "<<t0_other
	   <<" coarsen "<<t0_coarsen<<" solve "<<t0_solve<<std::endl;

  //////////////////////////////////////////////////////////////////////////
  // HMC: quenched Iwasaki evolution.
  //
  // NOTE: for this experiment only the smooth O(dt) evolution of U matters,
  // so we evolve with the pure gauge action. To make the evolution dynamical,
  // push_back a pseudofermion action (e.g. TwoFlavourEvenOddRatioPseudoFermionAction
  // wrapping the Mobius operators, on a separate integrator level) — beware
  // this makes each trajectory vastly more expensive at mass=0.001.
  //
  // We reuse Grid's HybridMonteCarlo driver one trajectory at a time
  // (Trajectories=1) so that control returns here between trajectories.
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
  IntegratorType MDynamics(UGrid,MDpar,FullSet,Smearing);

  std::vector<HmcObservable<LatticeGaugeField>*> NoObservables;

  struct TrajRecord {
    int   traj;
    RealD plaq;
    RealD th_flow, th_step, th_refined;    // theta_max: S0, S_{t-1}^fresh, S_t^refined vs S_t^fresh
    RealD d_flow,  d_step,  d_refined;     // matching dist2
    RealD eta_frozen, eta_refined, eta_fresh;
    int   steps_b, steps_c, steps_d;
    RealD relres_b, relres_c, relres_d;
    RealD dU;                              // gauge distance from U_0
    RealD cap_frozen, cap_refined, cap_step, cap_self; // mean capture of S_t^fresh by block spans
  };
  std::vector<TrajRecord> records;

  RealD t_last_traj = 0.0;
  for(int traj=1;traj<=Ntraj;traj++){

    ////////////////////////////////////////////////////////////////////////
    // Wall-clock guard: do not start a trajectory that would cross the
    // budget, so the final table is always printed.
    ////////////////////////////////////////////////////////////////////////
    RealD elapsed = (usecond()-t_start)/1.0e6;
    if ( elapsed + t_last_traj > budgetHours*3600.0 ) {
      std::cout<<GridLogMessage<<"MGEVO stopping before traj "<<traj<<": elapsed "<<elapsed
	       <<" s + last trajectory "<<t_last_traj<<" s exceeds budget "<<budgetHours<<" h"<<std::endl;
      break;
    }
    RealD t_traj = -usecond();
    std::string tt = "traj "+std::to_string(traj);

    ////////////////////////////////////////////////////////////////////////
    // One HMC trajectory: momentum refresh, MD integration, Metropolis.
    // On rejection Umu is left unchanged (a genuine HMC step).
    ////////////////////////////////////////////////////////////////////////
    HMCparameters HMCpar;
    HMCpar.StartTrajectory   = traj;
    HMCpar.Trajectories      = 1;
    HMCpar.NoMetropolisUntil = 0;
    HMCpar.MetropolisTest    = true;
    HMCpar.PerformRandomShift= false;

    RealD t_hmc = -usecond();
    HybridMonteCarlo<IntegratorType> HMC(HMCpar,MDynamics,sRNG,RNG4,NoObservables,Umu);
    HMC.evolve();
    t_hmc = (t_hmc+usecond())/1.0e6;

    RealD plaq = WilsonLoops<Gimpl>::avgPlaquette(Umu);
    std::cout<<GridLogMessage<<"MGEVO "<<tt<<" plaquette "<<plaq<<std::endl;

    ////////////////////////////////////////////////////////////////////////
    // Fresh operators on the updated gauge field. These are what we solve
    // with in ALL strategies — only the preconditioner content differs.
    ////////////////////////////////////////////////////////////////////////
    MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
    MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
    PVdagM_t        PVdagM(Ddwf,Dpv);
    ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv);

    ////////////////////////////////////////////////////////////////////////
    // Geometry (primary measurement), all on bases as bred:
    //   fresh   S_t^fresh  : re-bred at U_t from the same noise as S0
    //   refined S_t^refined: one sloppy inverse-iteration pass at U_t
    ////////////////////////////////////////////////////////////////////////
    RealD t_fresh = BreedBasis(AggregatesFresh,PVdagM,seedsB);
    std::vector<LatticeFermion> Sfresh = AggregatesFresh.subspace;

    // Seed the refresh from the previous refined basis AS REFINED. Seeding from
    // AggregatesRefined after coarsening (block-orthonormalised, so cut at
    // block boundaries) made rung (c) drift away from S_t^fresh faster than
    // the frozen basis did (job 424665: capture 0.9997 -> 0.75 over 10 traj).
    AggregatesRefined.subspace = SrefinedPrev;
    RealD t_refine = -usecond();
    RefineSubspaceGCR(AggregatesRefined,PVdagM,nbasis,refineTol,refineMaxOuter);
    t_refine = (t_refine+usecond())/1.0e6;
    std::vector<LatticeFermion> Srefined = AggregatesRefined.subspace;
    SrefinedPrev = Srefined;

    TrajRecord rec;
    rec.traj = traj;
    rec.plaq = plaq;
    rec.dU   = LinkDistance(U0,Umu);
    std::cout<<GridLogMessage<<"MGEVO "<<tt<<" d_U "<<rec.dU<<std::endl;

    RealD t_geom = -usecond();
    rec.th_flow    = ReportAngles("flow    "+tt+" S0 vs fresh",           S0,        Sfresh,nbasis,rec.d_flow);
    rec.th_step    = ReportAngles("step    "+tt+" prev fresh vs fresh",   SfreshPrev,Sfresh,nbasis,rec.d_step);
    rec.th_refined = ReportAngles("refined "+tt+" refined vs fresh",      Srefined,  Sfresh,nbasis,rec.d_refined);
    rec.eta_frozen  = ReportNulliness("frozen  "+tt,S0,      PVdagM,nbasis);
    rec.eta_refined = ReportNulliness("refined "+tt,Srefined,PVdagM,nbasis);
    rec.eta_fresh   = ReportNulliness("fresh   "+tt,Sfresh,  PVdagM,nbasis);
    t_geom = (t_geom+usecond())/1.0e6;

    SfreshPrev = Sfresh;

    ////////////////////////////////////////////////////////////////////////
    // MG solves (secondary measurement), each coarsened against U_t:
    //   (b) RECOARSEN : S0 frozen, coarse operator rebuilt at U_t
    //   (c) REFINE    : S_t^refined
    //   (d) FRESH     : S_t^fresh, the setup-from-scratch baseline at U_t
    // (a), the coarse operator frozen at U_0, is dropped: it stalled at
    // relres ~ 0.09 after one trajectory in job 424322.
    ////////////////////////////////////////////////////////////////////////
    RealD t_coarsenB, t_solveB, t_coarsenC, t_solveC, t_coarsenD, t_solveD;
    rec.steps_b = CoarsenAndSolve("(b) recoarsen "+tt,AggregatesFrozen, PVdagM,ShiftedPVdagM,t_coarsenB,t_solveB,rec.relres_b);
    rec.steps_c = CoarsenAndSolve("(c) refine    "+tt,AggregatesRefined,PVdagM,ShiftedPVdagM,t_coarsenC,t_solveC,rec.relres_c);
    rec.steps_d = CoarsenAndSolve("(d) fresh     "+tt,AggregatesFresh,  PVdagM,ShiftedPVdagM,t_coarsenD,t_solveD,rec.relres_d);

    ////////////////////////////////////////////////////////////////////////
    // Block-span capture of S_t^fresh (as bred). Each Aggregation is now
    // block-orthonormal, having just been coarsened above.
    ////////////////////////////////////////////////////////////////////////
    RealD t_cap = -usecond();
    rec.cap_frozen  = ReportCapture("frozen  "+tt+" fresh by S0",        AggregatesFrozen,   Sfresh,nbasis);
    rec.cap_refined = ReportCapture("refined "+tt+" fresh by refined",   AggregatesRefined,  Sfresh,nbasis);
    rec.cap_step    = ReportCapture("step    "+tt+" fresh by prev fresh",AggregatesFreshPrev,Sfresh,nbasis);
    rec.cap_self    = ReportCapture("self    "+tt+" fresh by fresh",     AggregatesFresh,    Sfresh,nbasis);
    t_cap = (t_cap+usecond())/1.0e6;

    AggregatesFreshPrev.subspace = AggregatesFresh.subspace;

    records.push_back(rec);

    t_traj = (t_traj+usecond())/1.0e6;
    t_last_traj = t_traj;

    // Wall times in seconds, one grep-able line per trajectory:
    //   MGEVO-TIME traj hmc fresh refine geom capture coarsen_b solve_b coarsen_c solve_c coarsen_d solve_d total
    std::cout<<GridLogMessage<<"MGEVO-TIME "<<traj<<" hmc "<<t_hmc<<" fresh "<<t_fresh
	     <<" refine "<<t_refine<<" geom "<<t_geom<<" capture "<<t_cap
	     <<" coarsen_b "<<t_coarsenB<<" solve_b "<<t_solveB
	     <<" coarsen_c "<<t_coarsenC<<" solve_c "<<t_solveC
	     <<" coarsen_d "<<t_coarsenD<<" solve_d "<<t_solveD
	     <<" total "<<t_traj<<std::endl;

    // One grep-able line per trajectory:
    //   MGEVO-SUMMARY traj plaq d_U cap_frozen cap_refined cap_step cap_self
    //                 th_flow th_step th_refined d_flow d_step d_refined
    //                 eta_frozen eta_refined eta_fresh steps_b steps_c steps_d relres_b relres_c relres_d
    std::cout<<GridLogMessage<<"MGEVO-SUMMARY "<<traj<<" "<<plaq<<" "<<rec.dU
	     <<" "<<rec.cap_frozen<<" "<<rec.cap_refined<<" "<<rec.cap_step<<" "<<rec.cap_self
	     <<" "<<rec.th_flow<<" "<<rec.th_step<<" "<<rec.th_refined
	     <<" "<<rec.d_flow<<" "<<rec.d_step<<" "<<rec.d_refined
	     <<" "<<rec.eta_frozen<<" "<<rec.eta_refined<<" "<<rec.eta_fresh
	     <<" "<<rec.steps_b<<" "<<rec.steps_c<<" "<<rec.steps_d
	     <<" "<<rec.relres_b<<" "<<rec.relres_c<<" "<<rec.relres_d<<std::endl;
  }

  //////////////////////////////////////////////////////////////////////////
  // Final table
  //////////////////////////////////////////////////////////////////////////
  std::cout<<GridLogMessage<<"=================================================================="<<std::endl;
  std::cout<<GridLogMessage<<" MG subspace evolution summary (nbasis "<<nbasis<<", trajL "<<trajL
	   <<", MDsteps "<<MDsteps<<", eta(S0) = "<<null0
	   <<", rebreed floor theta_max = "<<th0<<" deg)"<<std::endl;
  std::cout<<GridLogMessage<<" capture of S_t^fresh by block spans; at U_0: self "<<cap0_self
	   <<", other-noise ceiling "<<cap0_other<<", random floor "<<cap0_random<<std::endl;
  std::cout<<GridLogMessage<<" relres after at most "<<outerMaxRestarts*16<<" outer steps; fresh setup at U_0 = "
	   <<relres0<<" ("<<steps0<<" steps)"<<std::endl;
  std::cout<<GridLogMessage<<" traj  plaquette  d_U  cap(frozen)  cap(refined)  cap(step)  cap(self)  relres(b)  relres(c)  relres(d)"<<std::endl;
  for(auto &rec : records){
    std::cout<<GridLogMessage<<" "<<rec.traj<<"  "<<rec.plaq<<"  "<<rec.dU
	     <<"  "<<rec.cap_frozen<<"  "<<rec.cap_refined<<"  "<<rec.cap_step<<"  "<<rec.cap_self
	     <<"  "<<rec.relres_b<<"  "<<rec.relres_c<<"  "<<rec.relres_d<<std::endl;
  }
  std::cout<<GridLogMessage<<"=================================================================="<<std::endl;

  Grid_finalize();
  return 0;
}
