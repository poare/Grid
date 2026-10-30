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
//      paired lists, with Ntraj trajectories per pair, each from the starting
//      configuration with fresh momenta.
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

using namespace std;
using namespace Grid;

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
                                  int,         Ntraj,           // trajectories per pair, each from the start config
                                  PVdagMMultiGridParams, MultiGrid);
  QuenchedHMCDriverParams()
    : Ls(24), Mass(0.00078), M5(1.8), MobiusB(1.5), MobiusC(0.5),
      Config(""), SaveSubspace(""), Nrhs(12), SolveSingleRHS(1),
      Beta(2.13), RunMultiGrid(1), TrajL({1.0}), MDsteps({20}), Ntraj(1) {};
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
    auto RunSolve = [&](int nr)
    {
      PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,nr);
      std::vector<LatticeFermionD> src(nr,FGrid), sol(nr,FGrid);
      for(int r=0;r<nr;r++){ gaussian(RNG5,src[r]); sol[r]=Zero(); }
      Solver.Solve(src,sol);
    };

    RunSolve(P.Nrhs);
    if ( P.SolveSingleRHS && P.Nrhs != 1 ) RunSolve(1);
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

  std::cout.precision(15);
  LatticeGaugeField U0 = Umu;
  for(int s=0;s<P.TrajL.size();s++){
    RealD tau = P.TrajL[s];
    int   N   = P.MDsteps[s];
    IntegratorParameters MDpar(N,tau);
    SmearingPolicy Smearing;
    IntegratorType MD(UGrid,MDpar,FullSet,Smearing);

    for(int t=0;t<P.Ntraj;t++){
      Umu = U0;
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
        t_last = usecond();
      });

      RealD H1 = MD.S(Umu);
      RealD dH = H1 - H0;
      bool accept = MetropolisAccept(dH,sRNG);
      if ( !accept ) Umu = U0;
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
