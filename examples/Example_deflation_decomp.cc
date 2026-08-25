/*************************************************************************************

    Decomposes one or more fermion fields against a basis of vectors read from disk --
    a deflation basis, a set of Ritz vectors, or any other collection -- and reports how
    the norm is distributed across the basis.

    WHY.  Two sources built from the same three field-of-values vectors converge ~550
    iterations apart (notes/field-of-values.md section 9.6), and that gap is NOT the GCR
    breakdown: GMRES pays it too.  The remaining candidate is that the two have different
    expansions in F's eigenbasis -- the three-vector span is not F-invariant, so "same
    subspace" was never "same spectral content".  This driver measures that directly.

    THE BASIS NEED NOT BE ORTHONORMAL, and for a non-normal operator it generally is not:
    Ritz vectors of F are not orthogonal even though the Schur vectors are.  So the
    coefficients are NOT <e_i|v>.  This solves the least squares problem

        c = argmin || v - E c ||,      i.e.    G c = E^dag v,   G_ij = <e_i|e_j>,

    which reduces to c_i = <e_i|v> when G = I.  Using the naive inner products on a
    non-orthonormal basis silently reports the wrong profile, which is why G is formed and
    its deviation from the identity is printed.

    MEMORY.  All Nbasis vectors are held at once to form G.  One LatticeFermionD at
    16^3x32 with Ls = 16 is ~0.4 GB globally, so Nbasis = 50 is ~20 GB spread over ranks.

    No gauge field and no operator are needed -- this is pure linear algebra on fields.

    Usage :
      $ ./Example_deflation_decomp --basis <dir> --nbasis N --vecs <f1>[,<f2>...] \
            [--prefix evec] [--ls N] [--evals <file>] [--out <file>] [Grid options]

      --basis <dir>   = Directory holding the basis, named ${prefix}0 .. ${prefix}N-1 in
                        SCIDAC format. REQUIRED.
      --nbasis N      = How many basis vectors to read. REQUIRED.
      --vecs <list>   = Comma separated PATHS of the fields to decompose, e.g. the
                        stagnate_r0 and stagnate_control written by
                        Example_pvdagm_stagnate. REQUIRED.
      --prefix P      = Basis file prefix. Default "evec". Use "fov" for the field of
                        values vectors.
      --ls N          = Fifth dimension. Default 16. Must match the files.
      --evals <file>  = Optional evals.txt from Example_spec_kryschur, used to label each
                        row with its Ritz value so the profile can be read against the
                        spectrum. Lines starting with # are skipped; the last two numbers
                        on a line are taken as (Re, Im).
      --out <file>    = Optional text dump, one row per (vector, mode). Default none.

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_deflation_decomp.cc

    Copyright (C) 2026

    Author: Patrick Oare <poare@bnl.gov>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */

#include <fstream>
#include <iomanip>

#include <Grid/Grid.h>

#include <Grid/parallelIO/IldgIOtypes.h>
#include <Grid/parallelIO/IldgIO.h>

using namespace Grid;

template <class T> void readFile(T& out, std::string const fname){
  #ifdef HAVE_LIME
    std::cout << Grid::GridLogMessage << "Reads from: " << fname << std::endl;
    Grid::emptyUserRecord record;
    Grid::ScidacReader SR;
    SR.open(fname);
    SR.readScidacFieldRecord(out, record);
    SR.close();
  #else
    std::cout << Grid::GridLogError
              << "Grid was built without LIME; cannot read " << fname << std::endl;
  #endif
}

/** Ritz values from an evals.txt, for labelling only. Empty if the file is absent.
 *  Example_spec_kryschur writes `idx (re,im) ritz`, and older runs wrote `idx (re,im)`
 *  with no Ritz estimate, so the pair is taken from inside the parentheses rather than
 *  from the tail of the line -- otherwise the optional third column is read as the
 *  imaginary part. A bare `idx re im [ritz]` is accepted as well. */
std::vector<ComplexD> readEvals(std::string fname)
{
  std::vector<ComplexD> ev;
  std::ifstream f(fname);
  if (!f) return ev;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    size_t lp = line.find('(');
    size_t rp = line.find(')', lp == std::string::npos ? 0 : lp);
    std::string body;
    if (lp != std::string::npos && rp != std::string::npos) {
      body = line.substr(lp + 1, rp - lp - 1);
      for (auto &ch : body) if (ch == ',') ch = ' ';
      std::istringstream is(body);
      double re, im;
      if (is >> re >> im) ev.push_back(ComplexD(re, im));
    } else {
      std::istringstream is(line);
      std::vector<double> tok; double x;
      while (is >> x) tok.push_back(x);
      if (tok.size() >= 3) ev.push_back(ComplexD(tok[1], tok[2]));
    }
  }
  return ev;
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  std::string basisDir, vecList, prefix = "evec", evalFile, outFile;
  int nbasis = 0, Ls = 16;

  if (!GridCmdOptionExists(argv,argv+argc,"--basis") ||
      !GridCmdOptionExists(argv,argv+argc,"--nbasis") ||
      !GridCmdOptionExists(argv,argv+argc,"--vecs")) {
    std::cout << GridLogError << "Usage: " << argv[0]
              << " --basis <dir> --nbasis N --vecs <f1>[,<f2>...] "
              << "[--prefix evec] [--ls N] [--evals <file>] [--out <file>]" << std::endl;
    Grid_finalize();
    return 1;
  }
  basisDir = GridCmdOptionPayload(argv,argv+argc,"--basis");
  vecList  = GridCmdOptionPayload(argv,argv+argc,"--vecs");
  { std::string s = GridCmdOptionPayload(argv,argv+argc,"--nbasis");
    GridCmdOptionInt(s, nbasis); }
  if (GridCmdOptionExists(argv,argv+argc,"--prefix"))
    prefix = GridCmdOptionPayload(argv,argv+argc,"--prefix");
  if (GridCmdOptionExists(argv,argv+argc,"--ls")) {
    std::string s = GridCmdOptionPayload(argv,argv+argc,"--ls");
    GridCmdOptionInt(s, Ls);
  }
  if (GridCmdOptionExists(argv,argv+argc,"--evals"))
    evalFile = GridCmdOptionPayload(argv,argv+argc,"--evals");
  if (GridCmdOptionExists(argv,argv+argc,"--out"))
    outFile = GridCmdOptionPayload(argv,argv+argc,"--out");

  // GridCmdOptionIntVector only handles ints, and there is no string equivalent.
  std::vector<std::string> targets;
  {
    std::stringstream ss(vecList);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) targets.push_back(item);
  }
  assert(nbasis > 0 && "need at least one basis vector");
  assert(targets.size() > 0 && "--vecs parsed to nothing");

  Coordinate lat_size   = GridDefaultLatt();
  Coordinate simd_layout= GridDefaultSimd(Nd,vComplexD::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian * UGrid = SpaceTimeGrid::makeFourDimGrid(lat_size,simd_layout,mpi_layout);
  GridCartesian * FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);

  std::cout << GridLogMessage << "Basis: " << nbasis << " x " << basisDir << "/" << prefix
            << "{0.." << nbasis-1 << "}, Ls = " << Ls << std::endl;

  // ---- Read the basis and form the Gram matrix ---------------------------------------
  std::vector<LatticeFermionD> E(nbasis, FGrid);
  for (int i = 0; i < nbasis; i++) {
    readFile(E[i], basisDir + "/" + prefix + std::to_string(i));
  }

  Eigen::MatrixXcd G(nbasis,nbasis);
  for (int i = 0; i < nbasis; i++) {
    for (int j = 0; j < nbasis; j++) G(i,j) = innerProduct(E[i], E[j]);
  }
  RealD offDiag = 0.0, diagDev = 0.0;
  for (int i = 0; i < nbasis; i++) {
    diagDev = std::max(diagDev, std::abs(G(i,i) - ComplexD(1.0,0.0)));
    for (int j = 0; j < nbasis; j++)
      if (i != j) offDiag = std::max(offDiag, std::abs(G(i,j)));
  }
  std::cout << GridLogMessage << "Gram deviation from identity: max |G_ii - 1| = "
            << diagDev << ", max_{i!=j} |G_ij| = " << offDiag << std::endl;
  std::cout << GridLogMessage
            << "(large off-diagonal is expected for Ritz vectors of a non-normal operator;"
            << " the least squares solve below handles it)" << std::endl;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> solver(G);
  std::vector<ComplexD> evals = readEvals(evalFile);

  std::ofstream fout;
  if (!outFile.empty() && FGrid->IsBoss()) {
    fout.open(outFile);
    fout << "# vector  mode  Re_lambda  Im_lambda  |c_i|  frac  cumfrac\n";
    fout << std::scientific << std::setprecision(10);
  }

  // ---- Decompose each target ---------------------------------------------------------
  LatticeFermionD v(FGrid), rec(FGrid);
  for (auto &tname : targets) {
    readFile(v, tname);
    RealD nv = norm2(v);

    Eigen::VectorXcd rhs(nbasis);
    for (int i = 0; i < nbasis; i++) rhs(i) = innerProduct(E[i], v);
    Eigen::VectorXcd c = solver.solve(rhs);

    rec = Zero();
    for (int i = 0; i < nbasis; i++) rec = rec + c(i)*E[i];
    RealD captured = norm2(rec)/nv;
    RealD resid    = std::sqrt(axpy_norm(rec,-1.0,v,rec)/nv);   // rec is clobbered here

    std::cout << GridLogMessage << "=== " << tname << " ===" << std::endl;
    std::cout << GridLogMessage << "||v||^2 = " << nv
              << ", captured fraction ||Ec||^2/||v||^2 = " << captured
              << ", unrepresented ||v - Ec||/||v|| = " << resid << std::endl;

    RealD cum = 0.0;
    for (int i = 0; i < nbasis; i++) {
      RealD frac = std::norm(c(i))*real(G(i,i))/nv;   // per-mode share, exact iff G = I
      cum += frac;
      std::cout << GridLogMessage << "  mode " << std::setw(4) << i;
      if (i < (int)evals.size()) std::cout << "  lambda " << evals[i];
      std::cout << "  |c| " << std::abs(c(i)) << "  frac " << frac
                << "  cum " << cum << std::endl;
      if (fout.is_open()) {
        ComplexD lam = (i < (int)evals.size()) ? evals[i] : ComplexD(0.0,0.0);
        fout << tname << " " << i << " " << real(lam) << " " << imag(lam) << " "
             << std::abs(c(i)) << " " << frac << " " << cum << "\n";
      }
    }
  }
  if (fout.is_open()) fout.close();

  std::cout << GridLogMessage << "Done" << std::endl;
  Grid_finalize();
  return 0;
}
