/*************************************************************************************

    Low spectrum of PV^\dag PV for a Mobius Pauli-Villars operator PV = D(1), to test
    whether the near-real band in the spectrum of PV^\dag M (Example_pvinv_vs_pvdagm_spec)
    consists of bulk low modes of PV^\dag PV. On a vector v with PV^{-1} M v ~ v,
    PV^\dag M v = PV^\dag PV (PV^{-1} M v) ~ PV^\dag PV v. No PV inversions are needed.

    For each eigenvector v of PV^\dag PV (unit norm) it records the Rayleigh quotient
    rho = <v, PV^\dag M v> and the residual |PV^\dag M v - rho v|, which say how close
    v is to an eigenvector of PV^\dag M, and the profile |v_s|^2 across the fifth
    dimension, which separates bulk modes from modes bound to the walls.

    Usage :
      $ ./Example_pvdagpv_spec <Nm> <Nk> <maxiter> <Nstop> <inFile> <outDir>
            --grid Lx.Ly.Lz.Lt --mpi ... [--Ls 16] [--mass 0.1] [--M5 1.0,1.4,...]

      Nm      = Maximum size of approximation subspace.
      Nk      = Size of truncation subspace.
      maxiter = Maximum number of iterations.
      Nstop   = Stop when Nstop eigenvalues have converged.
      inFile  = Gauge configuration to read in (NERSC format).
      outDir  = Directory to write output to.
      --M5    = (Optional) comma-separated list of domain-wall heights, run one after another
                on the same gauge field and source. Without it M5 = 1.8 and the files below go
                in ${outDir} itself; with it they go in ${outDir}/M5_<M5>/, one per value.

    Output:
      ${outDir}/PVdagPV_evals.txt   = `$k $eval $ritz $rho $resid` per line, where $eval is
                                      real, $rho = (re,im), and $resid = |PV^\dag M v - rho v|.
      ${outDir}/PVdagPV_profile.txt = Nk rows of Ls numbers, row k = |v_s|^2 for s = 0..Ls-1.

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_pvdagpv_spec.cc

    Copyright (C) 2026

    Author: Patrick Oare <poare@bnl.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */

#include <filesystem>
#include <Grid/Grid.h>
#include <Grid/algorithms/multigrid/PVdagMOperators.h>

using namespace std;
using namespace Grid;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  std::string file   = argv[5];
  std::string outDir = argv[6];
  int Nm      = std::stoi(argv[1]);
  int Nk      = std::stoi(argv[2]);
  int maxIter = std::stoi(argv[3]);
  int Nstop   = std::stoi(argv[4]);

  //////////////////////////////////////////////////////////////////////
  // Action parameters, as in Example_pvinv_vs_pvdagm_spec
  //////////////////////////////////////////////////////////////////////
  int   Ls       = 16;     // override with --Ls
  RealD mass     = 0.1;    // override with --mass; enters only through PV^\dag M
  if (GridCmdOptionExists(argv,argv+argc,"--Ls")) {
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--Ls");
    GridCmdOptionInt(arg,Ls);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--mass")) {
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--mass");
    GridCmdOptionFloat(arg,mass);
  }
  std::vector<RealD> M5s = {1.8};   // override with --M5 1.0,1.4,...
  bool M5subdirs = GridCmdOptionExists(argv,argv+argc,"--M5");
  if (M5subdirs) {
    std::vector<std::string> s;
    GridCmdOptionCSL(GridCmdOptionPayload(argv,argv+argc,"--M5"),s);
    M5s.clear();
    for (auto &x : s) M5s.push_back(std::stod(x));
  }
  RealD b        = 1.5;
  RealD c        = 0.5;
  RealD KStol    = 1e-8;   // Krylov-Schur tolerance

  std::cout << GridLogMessage << "Reading gauge field from: " << file << std::endl;
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

  for (RealD M5 : M5s) {
    // Output directory for this M5: outDir itself without --M5, outDir/M5_<M5> with it
    std::string dir = outDir;
    if (M5subdirs) {
      std::ostringstream os; os << std::fixed << std::setprecision(2) << M5;
      dir = outDir + "/M5_" + os.str();
      if (FGrid->IsBoss()) std::filesystem::create_directories(dir);
      FGrid->Barrier();
    }
    std::cout << GridLogMessage << "=============== Mobius Ls=" << Ls << " M5=" << M5 << " m=" << mass
              << " b=" << b << " c=" << c << " -> " << dir << " ===============" << std::endl;

    MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
    MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,b,c);

    //////////////////////////////////////////////////////////////////////
    // PV^\dag PV: MdagMLinearOperator's HermOp, presented as Op by HermOpAdaptor
    //////////////////////////////////////////////////////////////////////
    MdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVop(Dpv);
    HermOpAdaptor<LatticeFermionD>                      PVdagPV(PVop);
    PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM(Ddwf,Dpv);

    std::cout << GridLogMessage << "Krylov-Schur on PV^dag PV: Nm=" << Nm << " Nk=" << Nk
              << " maxIter=" << maxIter << " Nstop=" << Nstop << std::endl;
    KrylovSchur KS(PVdagPV, FGrid, KStol, EvalNormSmall);
    RealD t0 = usecond();
    KS(src, maxIter, Nm, Nk, Nstop);
    RealD t1 = usecond();
    std::cout << GridLogMessage << "PV^dag PV Krylov-Schur time: " << (t1-t0)/1.0e6 << " s" << std::endl;

    Eigen::VectorXcd evals             = KS.getEvals();
    std::vector<RealD> ritz            = KS.getRitzEstimates();
    std::vector<LatticeFermionD> evecs = KS.getEvecs();

    //////////////////////////////////////////////////////////////////////
    // Per eigenvector: Rayleigh quotient of PV^\dag M, its residual, and the s-profile
    //////////////////////////////////////////////////////////////////////
    std::vector<ComplexD> rho(Nk);
    std::vector<RealD>    resid(Nk);
    std::vector<std::vector<RealD>> prof(Nk);
    LatticeFermionD v(FGrid), Av(FGrid);
    for (int k = 0; k < Nk; k++) {
      v = evecs[k] * (1.0 / std::sqrt(norm2(evecs[k])));
      PVdagM.Op(v, Av);
      rho[k]   = innerProduct(v, Av);
      Av       = Av - rho[k] * v;
      resid[k] = std::sqrt(norm2(Av));
      sliceNorm(prof[k], v, 0);                          // dimension 0 of a 5D field is s
      std::cout << GridLogMessage << "k=" << k << " lambda=" << evals(k) << " rho=" << rho[k]
                << " resid=" << resid[k] << " wall weight=" << prof[k][0] + prof[k][Ls-1] << std::endl;
    }

    if (FGrid->IsBoss()) {
      std::ofstream f(dir + "/PVdagPV_evals.txt");
      f << std::setprecision(16);
      for (int k = 0; k < Nk; k++) {
        f << k << " " << real(evals(k)) << " " << ritz[k] << " " << rho[k] << " " << resid[k] << "\n";
      }
      f.close();
      std::ofstream p(dir + "/PVdagPV_profile.txt");
      p << std::setprecision(10);
      for (int k = 0; k < Nk; k++) {
        for (int s = 0; s < Ls; s++) p << prof[k][s] << (s + 1 < Ls ? " " : "\n");
      }
      p.close();
    }
  }

  std::cout << GridLogMessage << "Done" << std::endl;
  Grid_finalize();
  return 0;
}
