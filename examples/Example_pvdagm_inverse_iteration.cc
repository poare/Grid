/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvdagm_inverse_iteration.cc

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

// Inverse iteration on Gaussian noise at a FIXED field, with the three-level
// mrhs PVdagM multigrid built on the basis after every round (research hub,
// projects/multigrid, notes/inverse-iteration-test-design.md).
//
// The question is why deeper inverse iteration makes the MG worse: the
// over-filtering hypothesis is that past the setup's three rounds the basis
// concentrates on the lowest modes of A = PV^dag M and loses the mixture of
// low modes the coarse space needs.
//
// The 60-vector basis starts as normalised Gaussian noise (k = 0).  For each
// k = 0..KMax:
//   1. coarsen and build the dense bottom on V_k, Galerkin certificate;
//   2. solve ONE fixed source, Nrhs 1, zero guess;
//   3. nulliness ||A v||/||v|| and |<v|A v>|/|v|^2 of every vector, and the
//      invariance residual ||Q A V||/||V||;
//   4. (k in CensusK) Peter Boyle's coarse census and coherence checks;
//   5. one round of inverse iteration, v -> A^{-1} v/||A^{-1} v|| by
//      unpreconditioned GCR(10,10) (RefineBasisGCR), skipped after KMax.
// The setup makes its basis with three such rounds, so V_3 is the basis of
// every production run.
//
// Every random number comes from a fixed seed and the field never changes,
// so a rerun reproduces the scan.  One II-* log line per measurement per k;
// lists on II-*-LIST lines, comma separated.  The final GCR residuals of a
// round are not logged here: qhmc_compact.py rebuilds them from the GCR lines.
//
// Parameters: one serialisable struct, read from XML with --ii-params; a
// missing file gets a template written next to it.
//
#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/algorithms/multigrid/PVdagMMultiGrid.h>
#include "Example_mg_hmc.h"
#include "Example_mg_refine.h"
#include "Example_basis_diagnostics.h"
#include "Example_coarse_coherence.h"

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

struct InverseIterationParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(InverseIterationParams,
                                  int,              Ls,
                                  RealD,            Mass,
                                  RealD,            M5,
                                  RealD,            MobiusB,
                                  RealD,            MobiusC,
                                  std::string,      Config,       // empty: hot start
                                  int,              KMax,         // rounds k = 0..KMax
                                  RealD,            RoundTol,     // GCR target per vector in a round
                                  int,              RoundMaxOuter,// restarts of GCR(10,10) per vector in a round
                                  std::vector<int>, CensusK,      // the k that get the census and the checks
                                  int,              CensusNstop,  // Lanczos: modes wanted
                                  int,              CensusNk,
                                  int,              CensusNm,
                                  RealD,            CensusTol,
                                  int,              CensusMaxIt,  // Lanczos restart cap; unconverged Ritz vectors are used as they are
                                  RealD,            ChebyLo,      // filter low edge in sigma^2
                                  RealD,            ChebyHi,      // filter high edge; 0: 1.1 sigma_max^2 from a power method
                                  int,              ChebyOrder,
                                  int,              Nbasis2,      // CHECK A: census vectors in the blocked basis
                                  PVdagMMultiGridParams, MultiGrid);
  InverseIterationParams()
    : Ls(24), Mass(0.00078), M5(1.8), MobiusB(1.5), MobiusC(0.5), Config(""),
      KMax(20), RoundTol(5.0e-4), RoundMaxOuter(30),
      // Nstop 96: CHECK A needs modes past Nbasis2 and CHECK B past NBASIS, as
      // Peter's coherence run (NEV 96) needed from his census.  CensusK and the
      // restart cap are a node-hour budget: under-filtered, setup, over-filtered.
      CensusK({2,3,5,10,20}), CensusNstop(96), CensusNk(128), CensusNm(160),
      CensusTol(1.0e-5), CensusMaxIt(4), ChebyLo(4.0), ChebyHi(0.0), ChebyOrder(401),
      Nbasis2(48)
  {
    // Peter's P0 cell, as the p0 set of run_hmc_multigrid_quenched_frontier.job;
    // the library defaults differ in the smoother, coarse solver and outer fields
    MultiGrid.Setup.RetainSubspace  = 1;
    MultiGrid.FineSmoother.Mmax     = 4;
    MultiGrid.CoarseSmoother.Shift  = 2.0;
    MultiGrid.CoarseSolver.Mmax     = 8;
    MultiGrid.Outer.MaxIterations   = 20;   // restarts of Outer.Nstep: cap 240
    MultiGrid.Outer.Nstep           = 12;
  };
};

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  InverseIterationParams P;
  {
    std::string pfile("");
    if( GridCmdOptionExists(argv,argv+argc,"--ii-params") )
      pfile = GridCmdOptionPayload(argv,argv+argc,"--ii-params");
    if ( pfile.length() ) {
      bool good; { std::ifstream f(pfile); good = f.good(); }
      if ( !good ) {
        if ( GlobalSharedMemory::WorldRank == 0 ) {
          XmlWriter WR(pfile+".templ");
          write(WR, "InverseIteration", P);
          std::cout << GridLogMessage << pfile << " does not exist; template written to "
                    << pfile << ".templ" << std::endl;
        }
        Grid_finalize(); return 0;
      }
      XmlReader RD(pfile);
      read(RD, "InverseIteration", P);
    }
    CheckValidity(P.MultiGrid);
    // The basis is set here, never loaded, and must survive every coarsening
    GRID_ASSERT( P.MultiGrid.Setup.SubspaceFile.length() == 0 );
    GRID_ASSERT( P.MultiGrid.Setup.RetainSubspace );
    GRID_ASSERT( P.KMax >= 0 );
    std::cout << GridLogMessage << "InverseIteration parameters ("
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

  // Seeds as Example_hmc_multigrid_quenched_frontier, so k = 0..16 compare
  // with its NoiseBasis scan (job 5853868)
  GridParallelRNG RNG4(UGrid);     RNG4.SeedFixedIntegers({1,2,3,4});
  GridParallelRNG RNG5(FGrid);     RNG5.SeedFixedIntegers({5,6,7,8});        // the source
  GridParallelRNG RNGnoise(FGrid); RNGnoise.SeedFixedIntegers({13,14,15,16}); // the basis at k = 0

  LatticeGaugeField Umu(UGrid);
  if ( P.Config.length() ) {
    std::cout << GridLogMessage << "Reading gauge field " << P.Config << std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,P.Config);
  } else {
    std::cout << GridLogMessage << "Hot start gauge field" << std::endl;
    SU<Nc>::HotConfiguration(RNG4,Umu);
  }

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,P.Mass,P.M5,P.MobiusB,P.MobiusC);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0,   P.M5,P.MobiusB,P.MobiusC);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  PVdagM_t PVdagM(Ddwf,Dpv);

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

  Coarsening_t Coarsening(CGrids, FGridsF, P.MultiGrid.Setup);

  //////////////////////////////////////////////////////////////////////
  // k = 0: the basis is normalised Gaussian noise; the source is drawn once
  //////////////////////////////////////////////////////////////////////
  std::vector<LatticeFermionD> &V = Coarsening.rawNull;
  V.clear();
  V.reserve(NBASIS);
  for(int b=0;b<NBASIS;b++){
    V.push_back(LatticeFermionD(FGrid));
    gaussian(RNGnoise,V[b]);
    V[b] = V[b]*std::pow(norm2(V[b]),-0.5);
  }
  LatticeFermionD src(FGrid);
  gaussian(RNG5,src);
  std::cout << GridLogMessage << "II-START nbasis " << NBASIS << " kmax " << P.KMax << std::endl;

  for(int k=0;k<=P.KMax;k++){

    //////////////////////////////////////////////////////////////////////
    // 1. Coarsen, dense bottom, certificate
    //////////////////////////////////////////////////////////////////////
    RealD t_coarsen = -usecond();
    Coarsening.Coarsen(PVdagM);
    t_coarsen = (t_coarsen+usecond())/1.0e6;
    RealD t_dense = -usecond();
    Coarsening.BuildDenseBottom();
    t_dense = (t_dense+usecond())/1.0e6;
    RealD cert = Coarsening.CertifyCoarsening(PVdagM);
    std::cout << GridLogMessage << "II-SETUP k " << k << " coarsen_seconds " << t_coarsen
              << " dense_seconds " << t_dense << " certificate " << cert << std::endl;

    //////////////////////////////////////////////////////////////////////
    // 2. The solve: the same source at every k, zero guess, Nrhs 1
    //////////////////////////////////////////////////////////////////////
    {
      PVdagMMultiGridSolver<MobiusFermionD,MobiusFermionF,Coarsening_t> Solver(Ddwf,Dpv,DdwfF,DpvF,Coarsening,P.MultiGrid,1);
      std::vector<LatticeFermionD> b(1,src), x(1,FGrid);
      x[0] = Zero();
      RealD t_solve = -usecond();
      Solver.Solve(b,x);
      t_solve = (t_solve+usecond())/1.0e6;
      int iters = Solver.L1PGCR.steps;
      int converged = ( iters < P.MultiGrid.Outer.MaxIterations*P.MultiGrid.Outer.Nstep );
      PVdagM.SloppyComms(0);
      LatticeFermionD Ax(FGrid);
      PVdagM.Op(x[0],Ax);
      Ax = Ax - src;
      RealD true_rel = std::sqrt(norm2(Ax)/norm2(src));
      std::cout << GridLogMessage << "II-SOLVE k " << k << " iters " << iters << " converged " << converged
                << " true_rel " << true_rel << " seconds " << t_solve << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    // 3. Nulliness of the raw basis V_k
    //////////////////////////////////////////////////////////////////////
    {
      std::vector<RealD> nul, rq;
      BasisNulliness(V,PVdagM,nul,rq);
      RealD inv = InvarianceResidual(V,PVdagM);
      std::cout << GridLogMessage << "II-NUL k " << k << BasisStat("nul",nul) << BasisStat("rq",rq)
                << " inv_resid " << inv << std::endl;
      std::cout << GridLogMessage << "II-NUL-LIST k " << k << " nul " << BasisList(nul)
                << " rq " << BasisList(rq) << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    // 4. Peter's coarse census and coherence checks on A_c(V_k)
    //    (Example_pvdagm_census.cc CENSUS 1, 3, 4; Example_pvdagm_coherence.cc
    //    CHECK 0, 1, A, B), on the production coarse operator.  A_c^dag comes
    //    from coarsening A^dag on the same basis, never from Mdag (which is M).
    //////////////////////////////////////////////////////////////////////
    if ( std::find(P.CensusK.begin(),P.CensusK.end(),k) != P.CensusK.end() ) {
      typedef Coarsening_t::CoarseOperator     CoarseOp_t;
      typedef Coarsening_t::CoarseVector       CoarseVector;
      typedef Lattice<iScalar<CoarseScalar_t> > CoarseCoarseScalar;
      GridCartesian *C5  = CGrids.Coarse5d;
      GridCartesian *CC5 = CGrids.CoarseCoarse5d;   // Block2 blocks of the coarse lattice
      RealD t_census = -usecond();

      // A_c^dag.  CoarsenOperator re-imports the basis into the coarsening's
      // own projector: the same raw basis, orthonormalised the same way, so P
      // is unchanged, and no second ~nbasis-field store is allocated.
      RealD t_adj = -usecond();
      CoarseOp_t CoarseOpAdj(Coarsening.geom,C5);
      {
        PVdagMAdjointOperator<MobiusFermionD,LatticeFermionD> PVdagMAdj(PVdagM);
        std::vector<LatticeFermionD> sub(V);
        CoarseOpAdj.SetGrid(CGrids.CoarseBatch);
        PVdagMAdj.SloppyComms(P.MultiGrid.Setup.FineSloppyComms);
        CoarseOpAdj.CoarsenOperator(PVdagMAdj,sub,C5,CGrids.batch,Coarsening.MrhsProjectorL1);
        PVdagMAdj.SloppyComms(0);
        Coarsening.MrhsProjectorL1.ReleaseScratch();
      }
      Coarsening.SetNrhs(1);
      CoarseOpAdj.SetGrid(Coarsening.CMrhs);
      t_adj = (t_adj+usecond())/1.0e6;

      CoarseSliceOperator<CoarseOp_t> Ac(Coarsening.CoarseOpPV,CoarseOpAdj,C5,Coarsening.CMrhs);
      GridParallelRNG CRNG(C5); CRNG.SeedFixedIntegers({5,6,7,8});

      // CENSUS 1.  O(1) if AdjOp were not the adjoint; the threshold allows
      // for the sloppy-halo coarsening of both operators at many ranks.
      RealD mismatch = AdjointMismatch(Ac,CRNG,C5);
      std::cout << GridLogMessage << "II-ADJCHECK k " << k << " mismatch " << mismatch << std::endl;
      GRID_ASSERT( mismatch < 1.0e-6 );

      // CENSUS 3 and 4: sigma_max^2, then Chebyshev-filtered Lanczos on A_c^dag A_c
      CoarseVector x0(C5);
      random(CRNG,x0);
      PowerMethod<CoarseVector> PM;
      RealD sigmax2 = PM(Ac,x0);
      RealD cheby_hi = ( P.ChebyHi > 0.0 ) ? P.ChebyHi : 1.1*sigmax2;

      RealD t_lanczos = -usecond();
      std::vector<RealD>        eval(P.CensusNm);
      std::vector<CoarseVector> evec(P.CensusNm,C5);
      int Nconv = 0;
      {
        Chebyshev<CoarseVector>      Cheby(P.ChebyLo,cheby_hi,P.ChebyOrder);
        FunctionHermOp<CoarseVector> OpCheby(Cheby,Ac);
        PlainHermOp<CoarseVector>    Op(Ac);
        ImplicitlyRestartedLanczos<CoarseVector> IRL(OpCheby,Op,P.CensusNstop,P.CensusNk,P.CensusNm,P.CensusTol,P.CensusMaxIt);
        random(CRNG,x0);
        IRL.calc(eval,evec,x0,Nconv);   // at the restart cap it still returns Nstop Ritz vectors
      }
      t_lanczos = (t_lanczos+usecond())/1.0e6;
      const int nev = evec.size();
      GRID_ASSERT( P.Nbasis2 < nev );

      // CHECK 0 (sigma and left/right alignment) and the residuals
      std::vector<RealD> sigma2(nev), resid(nev), sigma(nev), align(nev);
      {
        CoarseVector Av(C5), Hv(C5);
        for(int j=0;j<nev;j++){
          Ac.Op(evec[j],Av);
          RealD nv  = norm2(evec[j]);
          RealD nAv = norm2(Av);
          sigma[j]  = std::sqrt(nAv/nv);
          align[j]  = abs(innerProduct(Av,evec[j]))/std::sqrt(nAv*nv);   // |<u|v>|, u = A v/||A v||
          Ac.HermOp(evec[j],Hv);
          Hv = Hv - eval[j]*evec[j];
          sigma2[j] = eval[j];
          resid[j]  = std::sqrt(norm2(Hv)/nv);
        }
      }

      // CHECK 1: overlap of each mode with the GLOBAL span of the 60 psi_c
      std::vector<RealD> overlap(nev);
      {
        std::vector<CoarseVector> Qc(Coarsening.rawPsi);
        for(int i=0;i<NBASIS;i++){
          for(int j=0;j<i;j++){
            ComplexD ip = innerProduct(Qc[j],Qc[i]);
            Qc[i] = Qc[i] - ip*Qc[j];
          }
          Qc[i] = Qc[i]*std::pow(norm2(Qc[i]),-0.5);
        }
        for(int j=0;j<nev;j++){
          RealD ov = 0.0;
          for(int i=0;i<NBASIS;i++){
            ComplexD ip = innerProduct(Qc[i],evec[j]);
            ov += real(ip*conjugate(ip));
          }
          overlap[j] = ov/norm2(evec[j]);
        }
      }

      // CHECK A: the lowest Nbasis2 modes, blocked on Block2; completeness of the rest
      std::vector<RealD> compA, gsAglob, gsAmin;
      RealD idemA, meanA = 0.0;
      {
        std::vector<CoarseVector> basisA(evec.begin(),evec.begin()+P.Nbasis2);
        blockGSCensus<CoarseVector,CoarseCoarseScalar>(basisA,CC5,"A",gsAglob,gsAmin);
        CoarseVector w(C5), w2(C5);
        blockPPdag<CoarseVector,CoarseCoarseScalar>(basisA,CC5,evec[P.Nbasis2],w);
        blockPPdag<CoarseVector,CoarseCoarseScalar>(basisA,CC5,w,w2);
        w2 = w2 - w;
        idemA = std::sqrt(norm2(w2)/norm2(w));
        for(int j=P.Nbasis2;j<nev;j++){
          blockPPdag<CoarseVector,CoarseCoarseScalar>(basisA,CC5,evec[j],w);
          compA.push_back(norm2(w)/norm2(evec[j]));
          meanA += compA.back()/(nev-P.Nbasis2);
        }
      }

      // CHECK B: the 60 psi_c, blocked on Block2; completeness of every mode
      std::vector<RealD> compB, gsBglob, gsBmin;
      RealD idemB, meanBlow = 0.0, meanBhigh = 0.0;
      {
        std::vector<CoarseVector> basisB(Coarsening.rawPsi);
        blockGSCensus<CoarseVector,CoarseCoarseScalar>(basisB,CC5,"B",gsBglob,gsBmin);
        CoarseVector w(C5), w2(C5);
        blockPPdag<CoarseVector,CoarseCoarseScalar>(basisB,CC5,evec[0],w);
        blockPPdag<CoarseVector,CoarseCoarseScalar>(basisB,CC5,w,w2);
        w2 = w2 - w;
        idemB = std::sqrt(norm2(w2)/norm2(w));
        int nlow = 0, nhigh = 0;
        for(int j=0;j<nev;j++){
          blockPPdag<CoarseVector,CoarseCoarseScalar>(basisB,CC5,evec[j],w);
          compB.push_back(norm2(w)/norm2(evec[j]));
          if ( j < NBASIS ) { meanBlow  += compB.back(); nlow++;  }
          else              { meanBhigh += compB.back(); nhigh++; }
        }
        meanBlow  = nlow  ? meanBlow/nlow   : NAN;
        meanBhigh = nhigh ? meanBhigh/nhigh : NAN;
      }

      CoarseOpAdj.ReleaseGrid();   // CMrhs belongs to the coarsening and outlives it
      t_census = (t_census+usecond())/1.0e6;

      std::cout << GridLogMessage << "II-CENSUS k " << k << " sigma2_max " << sigmax2 << " cheby_hi " << cheby_hi
                << " nconv " << Nconv << " nev " << nev << " adj_mismatch " << mismatch
                << " adj_coarsen_seconds " << t_adj << " lanczos_seconds " << t_lanczos
                << " seconds " << t_census << std::endl;
      std::cout << GridLogMessage << "II-CENSUS-LIST k " << k << " sigma2 " << BasisList(sigma2)
                << " resid " << BasisList(resid) << std::endl;
      std::cout << GridLogMessage << "II-CHECK0-LIST k " << k << " sigma " << BasisList(sigma)
                << " align " << BasisList(align) << std::endl;
      std::cout << GridLogMessage << "II-CHECK1-LIST k " << k << " overlap " << BasisList(overlap) << std::endl;
      std::cout << GridLogMessage << "II-CHECKA k " << k << " nbasis2 " << P.Nbasis2 << " mean " << meanA
                << " idempotency " << idemA << std::endl;
      std::cout << GridLogMessage << "II-CHECKA-LIST k " << k << " comp " << BasisList(compA) << std::endl;
      std::cout << GridLogMessage << "II-CHECKB k " << k << " mean_low " << meanBlow << " mean_high " << meanBhigh
                << " idempotency " << idemB << std::endl;
      std::cout << GridLogMessage << "II-CHECKB-LIST k " << k << " comp " << BasisList(compB) << std::endl;
      std::cout << GridLogMessage << "II-GS-LIST k " << k << " A_glob " << BasisList(gsAglob)
                << " A_minblock " << BasisList(gsAmin) << " B_glob " << BasisList(gsBglob)
                << " B_minblock " << BasisList(gsBmin) << std::endl;
      Coarsening.ReportDeviceFootprint("after census");
    }

    //////////////////////////////////////////////////////////////////////
    // 5. One round of inverse iteration, V_k -> V_{k+1}
    //////////////////////////////////////////////////////////////////////
    if ( k < P.KMax ) {
      RealD t_round = -usecond();
      std::vector<int> steps = RefineBasisGCR(V,PVdagM,NBASIS,P.RoundTol,P.RoundMaxOuter);
      t_round = (t_round+usecond())/1.0e6;
      std::vector<RealD> s(steps.begin(),steps.end());
      std::cout << GridLogMessage << "II-ROUND k " << k << BasisStat("steps",s)
                << " seconds " << t_round << std::endl;
      std::cout << GridLogMessage << "II-ROUND-LIST k " << k << " steps " << BasisList(s) << std::endl;
    }
  }

  std::cout << GridLogMessage << "II-DONE" << std::endl;
  Grid_finalize();
  return 0;
}
