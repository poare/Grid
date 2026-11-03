/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_hmc_multigrid_quenched_frontier.cc

    Copyright (C) 2026

Author: Patrick Oare

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution
    directory
*************************************************************************************/
/*  END LEGAL */

//
// Quenched HMC with the three-level mrhs PVdagM multigrid attached
// (Grid/algorithms/multigrid/PVdagMMultiGrid.h).  This is diagnostic test 1
// of the deflation-basis-in-HMC plan (research hub, projects/multigrid,
// writeups/deflation_basis_evolution):
//
//   1. generate the near-null basis on the starting configuration (timed, fine
//      matvecs counted) and save it, or load a saved one;
//   2. coarsen, build the dense bottom and certify, as Example_pvdagm_multigrid;
//   3. one mrhs and one single-RHS solve on that basis;
//   4. quenched trajectories: Iwasaki gauge action only, MinimumNorm2,
//      momentum refresh, MD, Metropolis.  No solve is attached to the MD yet;
//      every step logs MD time, plaquette and wall time.  TrajL and MDsteps are
//      paired lists, with Ntraj trajectories per pair.  With Chain 0 each one
//      starts from the starting configuration, with fresh momenta; with Chain 1
//      each continues from the end of the last one;
//   5. with Mode set, MG solves at the gauge field of the MD (the passenger):
//      every SolveEvery steps of the trajectories in SolveTraj (empty: all).
//      The solve does not feed back, so with fixed seeds every Mode sees the
//      same chain.  Sources are Gaussian, refreshed per trajectory.
//
//   Mode none        no solves in the MD
//   Mode frozen      the fine operators follow U; basis and coarse operators
//                    stay at the starting configuration
//   Mode recoarsen   the same, and the coarse operators are rebuilt at U on
//                    the frozen basis (needs Setup.RetainSubspace)
//   Mode refresh     the live basis gets RefreshPasses sloppy inverse-iteration
//                    passes per vector at U (RefreshTol, RefreshMaxOuter restarts
//                    of GCR(10,10)), then the same rebuild.  A rejected
//                    trajectory restores the basis of its start.
//                    RefreshOrthonormalise 1 orthonormalises the basis after
//                    each pass (subspace iteration).  RefreshKick > 0 adds
//                    Gaussian noise of that size relative to |v| to every
//                    vector before each pass, from its own RNG so the MD
//                    sources are unchanged.
//   Mode perturbative  the first-order update v -> v - w, (Q A Q) w = Q dA v,
//                    with dA = A(U) - A(U at the last update) and Q the
//                    projector off the basis: mrhs GCR in batches of CorrNrhs,
//                    preconditioned by the V-cycle from the last update, to
//                    CorrTol, at most CorrMaxRestarts restarts.  Q is rebuilt
//                    from the current basis for each batch (no copy of the
//                    old one), an O(eps^2) change.  Then the same rebuild and
//                    the same restore on rejection.
//   Mode control     no MG (RunMultiGrid 0): the same system PV^dag M x = b
//                    at U by Krylov.  PV^dag y = b by CG on PV^dag PV (y = PV z),
//                    then M x = y by red-black CG to Outer.Tol on the Schur
//                    system.  The true residual of PV^dag M is logged after.
//
// RunMultiGrid 0 skips 1-3, for an HMC-only scan of the integrator.
//
// Every parameter is in ONE serialisable struct, read from XML with
// --hmc-params, as in Example_pvdagm_multigrid.  The basis:
//
//   MultiGrid.Setup.SubspaceFile set    the file is LOADED, and must exist:
//                                       given a missing file, GetSubspace
//                                       would generate and save over that path.
//   MultiGrid.Setup.SubspaceFile empty  the basis is GENERATED, then saved to
//                                       SaveSubspace if that is set.  An
//                                       existing SaveSubspace stops the run
//                                       before generating.
//
#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/multigrid/PVdagMMultiGrid.h>
#include "Example_mg_hmc.h"
#include "Example_mg_refine.h"

using namespace std;
using namespace Grid;

// Q M for an mrhs preconditioner M.  GCR hands it residuals already in the
// range of Q, so only the output needs projecting.
template<class Field>
class MrhsProjectedPreconditioner : public MrhsLinearFunction<Field> {
  SpanProjector<Field>      &_Q;
  MrhsLinearFunction<Field> &_M;
public:
  MrhsProjectedPreconditioner(SpanProjector<Field> &Q, MrhsLinearFunction<Field> &M) : _Q(Q), _M(M) {};
  void operator()(std::vector<Field> &in, std::vector<Field> &out) {
    _M(in,out);
    for(auto &f : out) _Q.Complement(f,f);
  }
};

#ifndef NBASIS
#define NBASIS 60
#endif

#ifdef COARSE_SINGLE
typedef sTComplexF CoarseScalar_t;
#else
typedef sTComplexD CoarseScalar_t;
#endif

struct QuenchedHMCDriverParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(QuenchedHMCDriverParams,
                                  int,         Ls,
                                  RealD,       Mass,
                                  RealD,       M5,
                                  RealD,       MobiusB,
                                  RealD,       MobiusC,
                                  std::string, Config,          // empty: hot start
                                  std::string, SaveSubspace,    // a generated basis is saved here; empty: not saved
                                  int,         Nrhs,
                                  int,         SolveSingleRHS,  // also run Nrhs=1 through the same objects
                                  RealD,       Beta,            // Iwasaki gauge action
                                  int,         RunMultiGrid,    // 0: skip sections 1-3, HMC only
                                  std::vector<RealD>, TrajL,    // one (TrajL,MDsteps) pair per scan point
                                  std::vector<int>,   MDsteps,
                                  int,         Ntraj,           // trajectories per pair
                                  int,         Chain,           // 1: each trajectory continues from the last
                                  std::string, Mode,            // none, frozen, recoarsen, refresh, perturbative or control
                                  int,         SolveEvery,      // MD steps between solves
                                  std::vector<int>, SolveTraj,  // trajectories that get solves; empty: all
                                  RealD,       RefreshTol,      // Mode refresh: GCR tolerance per vector
                                  int,         RefreshMaxOuter, // Mode refresh: restarts of GCR(10,10) per vector
                                  int,         RefreshPasses,   // Mode refresh: inverse-iteration passes per update
                                  int,         RefreshOrthonormalise, // Mode refresh: 1 orthonormalises after each pass
                                  RealD,       RefreshKick,     // Mode refresh: noise before each pass, relative to |v|; 0: none
                                  RealD,       CorrTol,         // Mode perturbative: batch GCR tolerance
                                  int,         CorrMaxRestarts, // Mode perturbative: restarts of the Outer cycle per batch
                                  int,         CorrNrhs,        // Mode perturbative: batch size, divides the basis size
                                  PVdagMMultiGridParams, MultiGrid);
  QuenchedHMCDriverParams()
    : Ls(24), Mass(0.00078), M5(1.8), MobiusB(1.5), MobiusC(0.5),
      Config(""), SaveSubspace(""), Nrhs(12), SolveSingleRHS(1),
      Beta(2.13), RunMultiGrid(1), TrajL({1.0}), MDsteps({20}), Ntraj(1), Chain(0),
      Mode("none"), SolveEvery(5), RefreshTol(0.1), RefreshMaxOuter(1), RefreshPasses(1), RefreshOrthonormalise(0), RefreshKick(0.0),
      CorrTol(1.0e-2), CorrMaxRestarts(4), CorrNrhs(12) {};
};

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  QuenchedHMCDriverParams P;
  {
    std::string pfile("");
    if( GridCmdOptionExists(argv,argv+argc,"--hmc-params") )
      pfile = GridCmdOptionPayload(argv,argv+argc,"--hmc-params");
    if ( pfile.length() ) {
      bool good; { std::ifstream f(pfile); good = f.good(); }
      if ( !good ) {
        if ( GlobalSharedMemory::WorldRank == 0 ) {
          XmlWriter WR(pfile+".templ");
          write(WR, "QuenchedHMCDriver", P);
          std::cout << GridLogMessage << pfile << " does not exist; template written to "
                    << pfile << ".templ" << std::endl;
        }
        Grid_finalize(); return 0;
      }
      XmlReader RD(pfile);
      read(RD, "QuenchedHMCDriver", P);
    }
    CheckValidity(P.MultiGrid);
    GRID_ASSERT( P.Mode=="none" || P.Mode=="frozen" || P.Mode=="recoarsen" || P.Mode=="refresh" || P.Mode=="perturbative" || P.Mode=="control" );
    GRID_ASSERT( P.Mode=="none" || P.Mode=="control" || P.RunMultiGrid );
    GRID_ASSERT( (P.Mode!="recoarsen" && P.Mode!="refresh" && P.Mode!="perturbative") || P.MultiGrid.Setup.RetainSubspace );
    GRID_ASSERT( P.Mode!="perturbative" || NBASIS%P.CorrNrhs==0 );
    std::cout << GridLogMessage << "QuenchedHMCDriver parameters ("
              << (pfile.length() ? pfile : std::string("defaults")) << "):" << std::endl;
    std::cout << P << std::endl;
  }

  Coordinate latt = GridDefaultLatt();
  Coordinate mpi  = GridDefaultMpi();
  Coordinate fsimd= GridDefaultSimd(Nd,vComplex::Nsimd());

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(latt,fsimd,mpi);
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(P.Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(P.Ls,UGrid);

  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers({1,2,3,4});
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers({5,6,7,8});
  GridParallelRNG RNGkick(FGrid); RNGkick.SeedFixedIntegers({9,10,11,12});  // Mode refresh kicks only
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers({9,10,11,12});

  LatticeGaugeField Umu(UGrid);
  if ( P.Config.length() ) {
    std::cout << GridLogMessage << "Reading gauge field " << P.Config << std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,P.Config);
  } else {
    std::cout << GridLogMessage << "Hot start gauge field" << std::endl;
    SU<Nc>::HotConfiguration(RNG4,Umu);
  }

  //////////////////////////////////////////////////////////////////////
  // Basis files are checked before anything expensive runs
  //////////////////////////////////////////////////////////////////////
  auto FileExists = [&](const std::string &f) -> bool {
    uint64_t e=0;
    if ( FGrid->IsBoss() ){ std::ifstream s(f); e = s.good()?1:0; }
    FGrid->GlobalSum(e);
    return e!=0;
  };
  const std::string &SubspaceFile = P.MultiGrid.Setup.SubspaceFile;
  const int generate = ( SubspaceFile.length() == 0 );
  if ( P.RunMultiGrid && !generate && !FileExists(SubspaceFile) ) {
    std::cout << GridLogError << "SubspaceFile " << SubspaceFile << " does not exist" << std::endl;
    GRID_ASSERT(0);
  }
  if ( P.RunMultiGrid && generate && P.SaveSubspace.length() && FileExists(P.SaveSubspace) ) {
    std::cout << GridLogError << "SaveSubspace " << P.SaveSubspace << " exists; not overwriting" << std::endl;
    GRID_ASSERT(0);
  }

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,   P.M5,P.MobiusB,P.MobiusC);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  PVdagM_t PVdagM(Ddwf,Dpv);

  // A at the U of the last update, for dA in Mode perturbative
  MobiusFermionD DdwfLast(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionD DpvLast (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,   P.M5,P.MobiusB,P.MobiusC);
  PVdagM_t PVdagMLast(DdwfLast,DpvLast);

  MatvecCounter counter;
  CountingLinearOperator<LatticeFermionD> CountedPVdagM(PVdagM,counter);

  typedef PVdagMMultiGridCoarsening<vSpinColourVector,CoarseScalar_t,NBASIS> Coarsening_t;
  std::cout << GridLogMessage << "Coarse sector precision (compiled): "
            << (sizeof(typename GridTypeMapper<CoarseScalar_t>::scalar_type)==sizeof(ComplexF) ? "fp32" : "fp64")
            << ", nbasis " << NBASIS << std::endl;

  MGCoarseGrids CGrids(FGrid, P.MultiGrid.Setup);
  MGFineGridsF  FGridsF(FGrid);

  LatticeGaugeFieldF UmuF(FGridsF.UGridF);
  precisionChange(UmuF,Umu);
  MobiusFermionF DdwfF(UmuF,*FGridsF.FGridF,*FGridsF.FrbGridF,*FGridsF.UGridF,*FGridsF.UrbGridF,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionF DpvF (UmuF,*FGridsF.FGridF,*FGridsF.FrbGridF,*FGridsF.UGridF,*FGridsF.UrbGridF,1.0,   P.M5,P.MobiusB,P.MobiusC);

  Coarsening_t  Coarsening(CGrids, FGridsF, P.MultiGrid.Setup);

  auto RunSolve = [&](int nr)
  {
    PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,nr);
    std::vector<LatticeFermionD> src(nr,FGrid), sol(nr,FGrid);
    for(int r=0;r<nr;r++){ gaussian(RNG5,src[r]); sol[r]=Zero(); }
    Solver.Solve(src,sol);
  };

  auto Rebuild = [&]()
  {
    Coarsening.Coarsen(PVdagM);
    Coarsening.BuildDenseBottom();
  };

  if ( P.RunMultiGrid ) {
    //////////////////////////////////////////////////////////////////////
    // 1. Basis: generate (or load), then save a generated one
    //////////////////////////////////////////////////////////////////////
    counter.Set("subspace");
    RealD t_subspace = -usecond();
    Coarsening.GetSubspace(RNG5, CountedPVdagM);
    t_subspace = (t_subspace+usecond())/1.0e6;
    std::cout << GridLogMessage << "QHMC-SUBSPACE " << (generate ? "generated" : "loaded")
              << " seconds " << t_subspace << " fine_matvecs " << counter.Get("subspace") << std::endl;

    if ( generate && P.SaveSubspace.length() ) {
      RealD t_save = -usecond();
      saveSubspace(Coarsening.rawNull, P.SaveSubspace);
      t_save = (t_save+usecond())/1.0e6;
      std::cout << GridLogMessage << "QHMC-SAVE " << P.SaveSubspace << " seconds " << t_save << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    // 2. Coarsen, dense bottom, certificate
    //////////////////////////////////////////////////////////////////////
    RealD t_coarsen = -usecond();
    Coarsening.Coarsen(PVdagM);
    t_coarsen = (t_coarsen+usecond())/1.0e6;
    RealD t_dense = -usecond();
    Coarsening.BuildDenseBottom();
    t_dense = (t_dense+usecond())/1.0e6;
    Coarsening.CertifyCoarsening(PVdagM);
    std::cout << GridLogMessage << "QHMC-SETUP coarsen_seconds " << t_coarsen
              << " dense_seconds " << t_dense << std::endl;

    //////////////////////////////////////////////////////////////////////
    // 3. Solves: mrhs then (optionally) single RHS through the SAME objects
    //////////////////////////////////////////////////////////////////////
    RunSolve(P.Nrhs);
    if ( P.SolveSingleRHS && P.Nrhs != 1 ) RunSolve(1);

    // The rebuild path against the first solve: same U, same basis
    if ( P.Mode=="recoarsen" ) {
      Rebuild();
      std::cout << GridLogMessage << "QHMC-REBUILD-CHECK solve after a rebuild at the starting configuration" << std::endl;
      RunSolve(P.Nrhs);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // 4. Quenched trajectories, Ntraj per (TrajL,MDsteps) pair
  //////////////////////////////////////////////////////////////////////
  typedef PeriodicGimplR                     Gimpl;
  typedef NoSmearing<Gimpl>                  SmearingPolicy;
  typedef MinimumNorm2<Gimpl,SmearingPolicy> IntegratorType;
  IwasakiGaugeActionR GaugeAction(P.Beta);
  ActionLevel<LatticeGaugeField> Level1(1);
  Level1.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField,NoHirep> FullSet;
  FullSet.push_back(Level1);
  GRID_ASSERT( P.TrajL.size()>0 && P.TrajL.size()==P.MDsteps.size() );

  //////////////////////////////////////////////////////////////////////
  // 5. The passenger solve, at the field of an MD step
  //////////////////////////////////////////////////////////////////////
  std::vector<LatticeFermionD> mdsrc(P.Mode=="none" ? 0 : P.Nrhs,FGrid);
  std::vector<LatticeFermionD> mdsol(P.Mode=="none" ? 0 : P.Nrhs,FGrid);
  const bool updates  = ( P.Mode=="refresh" || P.Mode=="perturbative" );
  const bool rebuilds = ( P.Mode=="recoarsen" || updates );

  // Free device memory in MB on the emptiest rank, so growth shows in the log
  // before an allocation fails.  For the QHMC-MEM line of each solve.
  auto FreeDeviceMB = [&]() -> RealD
  {
    size_t free_t=0,total_t=0;
#if defined(GRID_HIP)
    auto discard = hipMemGetInfo(&free_t,&total_t);
#elif defined(GRID_CUDA)
    cudaMemGetInfo(&free_t,&total_t);
#endif
    RealD neg = -(RealD)(free_t>>20);
    FGrid->GlobalMax(neg);
    return -neg;
  };

  // One first-order update of the basis, at the operators already imported;
  // the coarse operators are still those of the last update.  Returns the mean
  // outer steps per batch.
  auto PerturbativeBasisUpdate = [&](int traj,int step) -> RealD
  {
    std::vector<LatticeFermionD> &V = Coarsening.rawNull;
    PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,P.CorrNrhs);
    RealD mem_solver = FreeDeviceMB();

    // No copy of the old basis (60 fine fields): Q is rebuilt from the current
    // V for each batch, so later batches project off a span in which earlier
    // vectors have already moved by O(eps).  w is O(eps), so this changes it at
    // O(eps^2), inside the first-order update.
    std::vector<LatticeFermionD> rhs(P.CorrNrhs,FGrid), w(P.CorrNrhs,FGrid), vold(P.CorrNrhs,FGrid);
    LatticeFermionD Aold(FGrid);
    RealD steps = 0.0;
    for(int b0=0;b0<NBASIS;b0+=P.CorrNrhs){
      SpanProjector<LatticeFermionD> Q(V);
      GRID_ASSERT( Q.Ok() );
      ProjectedLinearOperator<LatticeFermionD> QAQ(Q,CountedPVdagM);
      MrhsProjectedPreconditioner<LatticeFermionD> QM(Q,Solver.ThreeLevelPrecon);
      MrhsPGCRNonHermitian<LatticeFermionD> GCR(P.CorrTol,P.CorrMaxRestarts,QAQ,QM,P.MultiGrid.Outer.Mmax,P.MultiGrid.Outer.Nstep);
      GCR.Name("Fcorrect");
      GCR.SetZeroGuess(1);
      for(int r=0;r<P.CorrNrhs;r++){
        vold[r] = V[b0+r];
        CountedPVdagM.Op(vold[r],rhs[r]);
        PVdagMLast.Op(vold[r],Aold);
        rhs[r] = rhs[r] - Aold;
        Q.Complement(rhs[r],rhs[r]);
      }
      RealD t_batch = -usecond();
      GCR(rhs,w);
      t_batch = (t_batch+usecond())/1.0e6;
      steps += (RealD)GCR.steps*P.CorrNrhs/NBASIS;
      RealD wratio = 0.0;
      for(int r=0;r<P.CorrNrhs;r++){
        wratio += std::sqrt(norm2(w[r])/norm2(vold[r]))/P.CorrNrhs;
        V[b0+r] = vold[r] - w[r];
        V[b0+r] = V[b0+r]*std::pow(norm2(V[b0+r]),-0.5);
      }
      std::cout << GridLogMessage << "QHMC-CORRECT traj " << traj << " step " << step << " first " << b0
                << " steps " << GCR.steps << " seconds " << t_batch << " wratio " << wratio
                << " free_solver_MB " << mem_solver << " free_batch_MB " << FreeDeviceMB() << std::endl;
    }
    return steps;
  };
  auto SolveAtU = [&](int traj,int step,RealD t_MD)
  {
    RealD t_import = -usecond();
    Ddwf.ImportGauge(Umu);
    Dpv.ImportGauge(Umu);
    precisionChange(UmuF,Umu);
    DdwfF.ImportGauge(UmuF);
    DpvF.ImportGauge(UmuF);
    t_import = (t_import+usecond())/1.0e6;

    RealD mem_start = FreeDeviceMB();
    RealD null_before = InvarianceResidual(Coarsening.rawNull,PVdagM);
    RealD t_update = 0.0;
    uint64_t mv_update = 0;
    RealD update_steps = 0.0;
    if ( updates ) {
      counter.Set("update");
      uint64_t m0 = counter.Get("update");
      t_update = -usecond();
      if ( P.Mode=="refresh" ) {
        std::vector<int> steps;
        for(int p=0;p<P.RefreshPasses;p++){
          if ( P.RefreshKick>0.0 ) KickBasis(Coarsening.rawNull,RNGkick,P.RefreshKick);
          steps = RefineBasisGCR(Coarsening.rawNull,CountedPVdagM,NBASIS,P.RefreshTol,P.RefreshMaxOuter);
          if ( P.RefreshOrthonormalise ) OrthonormaliseBasis(Coarsening.rawNull);
        }
        for(auto s : steps) update_steps += (RealD)s/steps.size();
      } else {
        update_steps = PerturbativeBasisUpdate(traj,step);
      }
      t_update = (t_update+usecond())/1.0e6;
      mv_update = counter.Get("update") - m0;
    }
    RealD mem_update = FreeDeviceMB();

    RealD t_rebuild = -usecond();
    if ( rebuilds ) Rebuild();
    t_rebuild = (t_rebuild+usecond())/1.0e6;
    if ( rebuilds ) Coarsening.CertifyCoarsening(PVdagM);
    RealD mem_rebuild = FreeDeviceMB();
    RealD null_resid = InvarianceResidual(Coarsening.rawNull,PVdagM);
    RealD gram_cond  = SpanProjector<LatticeFermionD>(Coarsening.rawNull).GramCondition();

    RealD t_solve = -usecond();
    PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,P.Nrhs);
    for(int r=0;r<P.Nrhs;r++) mdsol[r]=Zero();
    Solver.Solve(mdsrc,mdsol);
    t_solve = (t_solve+usecond())/1.0e6;
    RealD mem_solve = FreeDeviceMB();
    std::cout << GridLogMessage << "QHMC-SOLVE traj " << traj << " step " << step << " t_MD " << t_MD
              << " import_seconds " << t_import << " rebuild_seconds " << t_rebuild
              << " solve_seconds " << t_solve << " null_resid " << null_resid
              << " null_before " << null_before << " update_seconds " << t_update
              << " update_matvecs " << mv_update << " update_steps " << update_steps
              << " gram_cond " << gram_cond << std::endl;
    std::cout << GridLogMessage << "QHMC-MEM traj " << traj << " step " << step
              << " free_start_MB " << mem_start << " free_update_MB " << mem_update
              << " free_rebuild_MB " << mem_rebuild << " free_solve_MB " << mem_solve
              << " grid_cache_MB " << (MemoryManager::DeviceCacheBytes()>>20) << std::endl;
    if ( P.Mode=="perturbative" ) {
      DdwfLast.ImportGauge(Umu);
      DpvLast.ImportGauge(Umu);
    }
  };
  auto ControlAtU = [&](int traj,int step,RealD t_MD)
  {
    Ddwf.ImportGauge(Umu);
    Dpv.ImportGauge(Umu);
    MdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagPV(Dpv);
    ConjugateGradient<LatticeFermionD> CGpv(1.0e-2*P.MultiGrid.Outer.Tol,100000,false);
    ConjugateGradient<LatticeFermionD> CG(P.MultiGrid.Outer.Tol,100000,false);
    SchurRedBlackDiagMooeeSolve<LatticeFermionD> RBCG(CG);
    std::vector<LatticeFermionD> sol(1,FGrid), y(1,FGrid);
    LatticeFermionD z(FGrid), r(FGrid);
    sol[0] = Zero();
    z = Zero();
    RealD t_pv = -usecond();
    CGpv(PVdagPV,mdsrc[0],z);
    Dpv.M(z,y[0]);
    t_pv = (t_pv+usecond())/1.0e6;
    RealD t_solve = -usecond();
    RBCG(Ddwf,y,sol);
    t_solve = (t_solve+usecond())/1.0e6;
    PVdagM.Op(sol[0],r);
    r = r - mdsrc[0];
    std::cout << GridLogMessage << "QHMC-CONTROL traj " << traj << " step " << step << " t_MD " << t_MD
              << " pv_seconds " << t_pv << " pv_iters " << CGpv.IterationsToComplete
              << " solve_seconds " << t_solve << " iters " << CG.IterationsToComplete
              << " true_resid " << std::sqrt(norm2(r)/norm2(mdsrc[0])) << std::endl;
  };
  auto SolvesIn = [&](int traj) -> bool {
    if ( P.Mode=="none" ) return false;
    if ( P.SolveTraj.empty() ) return true;
    return std::find(P.SolveTraj.begin(),P.SolveTraj.end(),traj)!=P.SolveTraj.end();
  };

  std::cout.precision(15);
  LatticeGaugeField U0 = Umu;
  for(int s=0;s<P.TrajL.size();s++){
    RealD tau = P.TrajL[s];
    int   N   = P.MDsteps[s];
    IntegratorParameters MDpar(N,tau);
    SmearingPolicy Smearing;
    IntegratorType MD(UGrid,MDpar,FullSet,Smearing);

    for(int t=0;t<P.Ntraj;t++){
      if ( !P.Chain ) Umu = U0;
      LatticeGaugeField Ustart = Umu;
      const bool solves = SolvesIn(t);
      std::vector<LatticeFermionD> Vstart;
      if ( updates && P.Ntraj>1 ) Vstart = Coarsening.rawNull;
      if ( P.Mode=="perturbative" ) {
        DdwfLast.ImportGauge(Umu);
        DpvLast.ImportGauge(Umu);
      }
      if ( solves ) for(int r=0;r<P.Nrhs;r++) gaussian(RNG5,mdsrc[r]);
      RealD t_traj = -usecond();
      MD.refresh(Umu,sRNG,RNG4);
      RealD H0 = MD.Sinitial(Umu);
      std::cout << GridLogMessage << "QHMC-STEP 0 t_MD 0 plaquette "
                << WilsonLoops<Gimpl>::avgPlaquette(Umu) << " seconds 0" << std::endl;

      RealD t_last = usecond();
      IntegrateWithHook(MD,Umu,[&](int step) {
        RealD t_step = (usecond()-t_last)/1.0e6;
        std::cout << GridLogMessage << "QHMC-STEP " << step << " t_MD " << step*tau/N
                  << " plaquette " << WilsonLoops<Gimpl>::avgPlaquette(Umu)
                  << " seconds " << t_step << std::endl;
        if ( solves && step%P.SolveEvery==0 ) {
          if ( P.Mode=="control" ) ControlAtU(t,step,step*tau/N);
          else                     SolveAtU(t,step,step*tau/N);
        }
        t_last = usecond();
      });

      RealD H1 = MD.S(Umu);
      RealD dH = H1 - H0;
      bool accept = MetropolisAccept(dH,sRNG);
      if ( !accept ) {
        Umu = Ustart;
        if ( !Vstart.empty() ) Coarsening.rawNull = Vstart;
      }
      t_traj = (t_traj+usecond())/1.0e6;
      std::cout << GridLogMessage << "QHMC-TRAJ tau " << tau << " N " << N << " traj " << t
                << " H0 " << H0 << " H1 " << H1 << " dH " << dH
                << " accept " << accept << " plaquette " << WilsonLoops<Gimpl>::avgPlaquette(Umu)
                << " seconds " << t_traj << std::endl;
    }
  }

  counter.Report("QHMC-MATVECS");
  Grid_finalize();
}
