/*************************************************************************************

    Compares the low spectrum of PV^\dag M against that of PV^{-1} M for a Mobius
    domain-wall operator M = D(m) with Pauli-Villars operator PV = D(1). Both spectra
    are computed with Krylov-Schur from the same random source.

    Usage :
      $ ./Example_pvinv_vs_pvdagm_spec <Nm> <Nk> <maxiter> <Nstop> <inFile> <outDir> <?rf>
            --grid Lx.Ly.Lz.Lt --mpi ... [--Ls 16] [--mass 0.1]

      Nm      = Maximum size of approximation subspace.
      Nk      = Size of truncation subspace.
      maxiter = Maximum number of iterations.
      Nstop   = Stop when Nstop eigenvalues have converged.
      inFile  = Gauge configuration to read in (NERSC format).
      outDir  = Directory to write output to.
      rf      = (Optional) RitzFilter, default EvalNormSmall.

    Output:
      ${outDir}/PVdagM_evals.txt = eigenvalues of PV^\dag M, one `$idx $eval $ritz` per line.
      ${outDir}/PVinvM_evals.txt = eigenvalues of PV^{-1} M, same format.
      ${outDir}/PVdagM_evecs     = the Nk eigenvectors of PV^\dag M, in the order of
                                   PVdagM_evals.txt, as one SciDAC file (saveSubspace;
                                   read back with loadSubspace). Needs LIME.
      ${outDir}/PVinvM_evecs     = the same for PV^{-1} M.

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvinv_vs_pvdagm_spec.cc

    Copyright (C) 2026

    Author: Patrick Oare <poare@bnl.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */

#include <Grid/Grid.h>
#include <Grid/algorithms/multigrid/PVdagMOperators.h>
#include <Grid/algorithms/multigrid/MultiGridIO.h>

using namespace std;
using namespace Grid;

/**
 * A = PV^{-1} M, applied as out = PV^{-1} (M in). The PV solve is delegated to a
 * Pauli-Villars inverter from PauliVillarsInverters.h, which sets the mass of the matrix
 * it is handed to 1 for the duration of the solve.
 *
 * Krylov-Schur only calls Op, so the adjoint (which would need a PV^\dag solve) is not
 * implemented. The inner solve must be converged well below the Krylov-Schur tolerance,
 * since Krylov-Schur assumes it is applying a fixed linear map.
 */
template<class Matrix,class Field,class PVinverter>
class PVinvMLinearOperator : public LinearOperatorBase<Field> {
  Matrix     &_Mat;
  Matrix     &_PV;
  PVinverter &_PVinv;
public:
  PVinvMLinearOperator(Matrix &Mat,Matrix &PV,PVinverter &PVinv): _Mat(Mat),_PV(PV),_PVinv(PVinv){};

  void OpDiag (const Field &in, Field &out)                 { GRID_ASSERT(0); }
  void OpDir  (const Field &in, Field &out,int dir,int disp) { GRID_ASSERT(0); }
  void OpDirAll  (const Field &in, std::vector<Field> &out)  { GRID_ASSERT(0); }
  void Op     (const Field &in, Field &out)
  {
    Field tmp(in.Grid());
    _Mat.M(in,tmp);
    out = Zero();
    _PVinv(_PV,tmp,out);
  }
  void AdjOp  (const Field &in, Field &out)                  { GRID_ASSERT(0); }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2) { GRID_ASSERT(0); }
  void HermOp (const Field &in, Field &out)                  { GRID_ASSERT(0); }
};

/**
 * Writes the eigenvalues and Ritz estimates of a completed Krylov-Schur run to fname,
 * one `$idx $eval $ritz` per line. Boss-only; eigenvectors are not written.
 */
template <class Field>
void writeEvals(KrylovSchur<Field> &KS, std::string fname) {
  int Nk = KS.getNk();
  Eigen::VectorXcd evals   = KS.getEvals();
  std::vector<RealD> ritz  = KS.getRitzEstimates();
  std::cout << GridLogMessage << "Writing eigenvalues to: " << fname << std::endl;
  std::ofstream f(fname);
  f << std::setprecision(16);
  for (int i = 0; i < Nk; i++) {
    f << i << " " << evals(i) << " " << ritz[i] << "\n";
  }
  f.close();
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  std::string file   = argv[5];
  std::string outDir = argv[6];
  int Nm      = std::stoi(argv[1]);
  int Nk      = std::stoi(argv[2]);
  int maxIter = std::stoi(argv[3]);
  int Nstop   = std::stoi(argv[4]);
  RitzFilter RF = (argc >= 8 && std::string(argv[7]).rfind("--",0) != 0) ? selectRitzFilter(argv[7]) : EvalNormSmall;

  //////////////////////////////////////////////////////////////////////
  // Action parameters
  //////////////////////////////////////////////////////////////////////
  int   Ls       = 16;     // override with --Ls
  RealD mass     = 0.1;    // override with --mass
  if (GridCmdOptionExists(argv,argv+argc,"--Ls")) {
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--Ls");
    GridCmdOptionInt(arg,Ls);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--mass")) {
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--mass");
    GridCmdOptionFloat(arg,mass);
  }
  RealD M5       = 1.8;
  RealD b        = 1.5;    // Mobius scale b+c=2, b-c=1, as in Example_spec_kryschur
  RealD c        = 0.5;
  RealD KStol    = 1e-8;   // Krylov-Schur tolerance
  RealD PVtol    = 1e-12;  // inner PV solve tolerance, well below KStol

  std::cout << GridLogMessage << "Sorting eigenvalues using " << rfToString(RF) << std::endl;
  std::cout << GridLogMessage << "Reading gauge field from: " << file << std::endl;
  std::cout << GridLogMessage << "Mobius Ls=" << Ls << " M5=" << M5 << " m=" << mass
            << " b=" << b << " c=" << c << std::endl;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                    GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                                    GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  LatticeFermionD src(FGrid);   random(RNG5,src);

  LatticeGaugeFieldD Umu(UGrid);
  FieldMetaData header;
  NerscIO::readConfiguration(Umu,header,file);

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,b,c);

  //////////////////////////////////////////////////////////////////////
  // Operators: PV^\dag M from PVdagMOperators.h, PV^{-1} M from above
  //////////////////////////////////////////////////////////////////////
  typedef SchurRedBlackDiagTwoSolve<LatticeFermionD>                    SchurSolver_t;
  typedef PauliVillarsSolverRBprec<LatticeFermionD,SchurSolver_t>       PVinverter_t;

  ConjugateGradient<LatticeFermionD> PVCG(PVtol,10000);
  SchurSolver_t PVSchur(PVCG);
  PVinverter_t  PVinv(PVSchur);

  PVdagMLinearOperator<MobiusFermionD,LatticeFermionD>              PVdagM(Ddwf,Dpv);
  PVinvMLinearOperator<MobiusFermionD,LatticeFermionD,PVinverter_t> PVinvM(Ddwf,Dpv,PVinv);

  //////////////////////////////////////////////////////////////////////
  // Krylov-Schur on each, same source
  //////////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << "Krylov-Schur: Nm=" << Nm << " Nk=" << Nk
            << " maxIter=" << maxIter << " Nstop=" << Nstop << std::endl;

  std::cout << GridLogMessage << "*************** PV^dag M ***************" << std::endl;
  KrylovSchur KSdag(PVdagM, FGrid, KStol, RF);
  RealD t0 = usecond();
  KSdag(src, maxIter, Nm, Nk, Nstop);
  RealD t1 = usecond();
  std::cout << GridLogMessage << "PV^dag M Krylov-Schur time: " << (t1-t0)/1.0e6 << " s" << std::endl;
  std::cout << GridLogMessage << "PV^dag M eigenvalues: " << std::endl << KSdag.getEvals() << std::endl;
  if (FGrid->IsBoss()) { writeEvals(KSdag, outDir + "/PVdagM_evals.txt"); }
  {
    std::vector<LatticeFermionD> evecs = KSdag.getEvecs();
    saveSubspace(evecs, outDir + "/PVdagM_evecs");      // collective: every rank
  }

  std::cout << GridLogMessage << "*************** PV^-1 M ****************" << std::endl;
  KrylovSchur KSinv(PVinvM, FGrid, KStol, RF);
  RealD t2 = usecond();
  KSinv(src, maxIter, Nm, Nk, Nstop);
  RealD t3 = usecond();
  std::cout << GridLogMessage << "PV^-1 M Krylov-Schur time: " << (t3-t2)/1.0e6 << " s" << std::endl;
  std::cout << GridLogMessage << "PV^-1 M eigenvalues: " << std::endl << KSinv.getEvals() << std::endl;
  if (FGrid->IsBoss()) { writeEvals(KSinv, outDir + "/PVinvM_evals.txt"); }
  {
    std::vector<LatticeFermionD> evecs = KSinv.getEvecs();
    saveSubspace(evecs, outDir + "/PVinvM_evecs");      // collective: every rank
  }

  std::cout << GridLogMessage << "Done" << std::endl;
  Grid_finalize();
  return 0;
}
