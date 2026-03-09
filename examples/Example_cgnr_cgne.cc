/*
 * Warning: This code illustrative only: not well tested, and not meant for production use
 * without regression / tests being applied
 */

#include <Grid/Grid.h>

#include <cstdio>

using namespace std;
using namespace Grid;

template<class Gimpl,class Field> class CovariantLaplacianCshift : public SparseMatrixBase<Field>
{
public:
  INHERIT_GIMPL_TYPES(Gimpl);

  GridBase *grid;
  GaugeField U;
  
  CovariantLaplacianCshift(GaugeField &_U)    :
    grid(_U.Grid()),
    U(_U) {  };

  virtual GridBase *Grid(void) { return grid; };

  virtual void  M    (const Field &in, Field &out)
  {
    out=Zero();
    for(int mu=0;mu<Nd-1;mu++) {
      GaugeLinkField Umu = PeekIndex<LorentzIndex>(U, mu); // NB: Inefficent
      out = out - Gimpl::CovShiftForward(Umu,mu,in);    
      out = out - Gimpl::CovShiftBackward(Umu,mu,in);    
      out = out + 2.0*in;
    }
  };
  virtual void  Mdag (const Field &in, Field &out) { M(in,out);}; // Laplacian is hermitian
  virtual  void Mdiag    (const Field &in, Field &out)                  {assert(0);}; // Unimplemented need only for multigrid
  virtual  void Mdir     (const Field &in, Field &out,int dir, int disp){assert(0);}; // Unimplemented need only for multigrid
  virtual  void MdirAll  (const Field &in, std::vector<Field> &out)     {assert(0);}; // Unimplemented need only for multigrid
};

void MakePhase(Coordinate mom,LatticeComplex &phase)
{
  GridBase *grid = phase.Grid();
  auto latt_size = grid->GlobalDimensions();
  ComplexD ci(0.0,1.0);
  phase=Zero();

  LatticeComplex coor(phase.Grid());
  for(int mu=0;mu<Nd;mu++){
    RealD TwoPiL =  M_PI * 2.0/ latt_size[mu];
    LatticeCoordinate(coor,mu);
    phase = phase + (TwoPiL * mom[mu]) * coor;
  }
  phase = exp(phase*ci);
}

void PointSource(Coordinate &coor,LatticePropagator &source)
{
  //  Coordinate coor({0,0,0,0});
  source=Zero();
  SpinColourMatrix kronecker; kronecker=1.0;
  pokeSite(kronecker,source,coor);
}

void Z2WallSource(GridParallelRNG &RNG,int tslice,LatticePropagator &source)
{
  GridBase *grid = source.Grid();
  LatticeComplex noise(grid);
  LatticeComplex zz(grid); zz=Zero();
  LatticeInteger t(grid);

  RealD nrm=1.0/sqrt(2);
  bernoulli(RNG, noise); // 0,1 50:50

  noise = (2.*noise - Complex(1,1))*nrm;

  LatticeCoordinate(t, Tdir);
  noise = where(t==Integer(tslice), noise, zz);

  // std::cout << "noise" << noise << std::endl;

  source = 1.0;
  source = source*noise;
  std::cout << " Z2 wall " << norm2(source) << std::endl;
}

template<class Field>
void GaussianSmear(LatticeGaugeField &U,Field &unsmeared,Field &smeared)
{
  typedef CovariantLaplacianCshift <PeriodicGimplR,Field> Laplacian_t;
  Laplacian_t Laplacian(U);

  Integer Iterations = 40;
  Real width = 2.0;
  Real coeff = (width*width) / Real(4*Iterations);

  Field tmp(U.Grid());
  smeared=unsmeared;
  //  chi = (1-p^2/2N)^N kronecker
  for(int n = 0; n < Iterations; ++n) {
    Laplacian.M(smeared,tmp);
    smeared = smeared - coeff*tmp;
    std::cout << " smear iter " << n<<" " <<norm2(smeared)<<std::endl;
  }
}

void GaussianSource(Coordinate &site,LatticeGaugeField &U,LatticePropagator &source)
{
  LatticePropagator tmp(source.Grid());
  PointSource(site,source);
  std::cout << " GaussianSource Kronecker "<< norm2(source)<<std::endl;
  tmp = source;
  GaussianSmear(U,tmp,source);
  std::cout << " GaussianSource Smeared "<< norm2(source)<<std::endl;
}

void GaussianWallSource(GridParallelRNG &RNG,int tslice,LatticeGaugeField &U,LatticePropagator &source)
{
  Z2WallSource(RNG,tslice,source);
  auto tmp = source;
  GaussianSmear(U,tmp,source);
}

void SequentialSource(int tslice,Coordinate &mom,LatticePropagator &spectator,LatticePropagator &source)
{
  assert(mom.size()==Nd);
  assert(mom[Tdir] == 0);

  GridBase * grid = spectator.Grid();


  LatticeInteger ts(grid);
  LatticeCoordinate(ts,Tdir);
  source = Zero();
  source = where(ts==Integer(tslice),spectator,source); // Stick in a slice of the spectator, zero everywhere else

  LatticeComplex phase(grid);
  MakePhase(mom,phase);

  source = source *phase;
}

class MesonFile: Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFile, std::vector<std::vector<Complex> >, data);
};

// class OutputFile: Serializable {
// public:
//   GRID_SERIALIZABLE_CLASS_MEMBERS(OutputFile, std::vector< std::vector< Complex > >, data);
// };
class OutputFile: Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(OutputFile, std::vector< std::vector< Real > >, data);
};

/**
 * Computes the two-norm of a lattice field across each time slice and pushes the resulting vector into an XML file.
 * 
 * Parameters
 * ----------
 * field    (Field&)
 *    Field u(x, t) to reduce. 
 * outFile  (OutputFile&)
 *    Output file to write \sum_x |u(x, t)|^2 to.
 * tReduce  (int, default = Tdir + 1)
 *    Tensor direction to reduce over (for 4D fields, Tdir = 3 reduces over the time direction. for 5d, use Tdir+1 = 4).
 */
template <class Field>
void twoNormSlicePushBack(const Field &field, OutputFile &outFile, int tReduce = Tdir + 1) {

  LatticeRealD fieldsq = toReal( localNorm2(field) );
  std::vector<TReal> fieldsqt;               // Norm of source not summed over time slices, ssq(t) = \sum_x |b(x, t)|^2
  sliceSum(fieldsq, fieldsqt, tReduce);          // ssqt = \sum_x |b(x, t)|^2. Note Tdir = 3 is for 4d fields, Tdir + 1 = 4 is for 5d (DWF) fields
  // std::cout << GridLogIterative << "sum_x |u(x, t)|^2 = " << fieldsqt << std::endl;

  int nt = fieldsqt.size();
  std::vector<Real> fieldsqt_unwrap(nt);         // Unwrap the additional stuff around the vector to return.
  for(int t = 0; t < nt; t++){
    fieldsqt_unwrap[t] = TensorRemove(fieldsqt[t]);
  }
  outFile.data.push_back(fieldsqt_unwrap);
}

template<class Field> class NormalEquationsTimeslice : public LinearFunction<Field>{
// private:
public:
  SparseMatrixBase<Field> & _Matrix;
  OperatorFunction<Field> & _HermitianSolver;
  LinearFunction<Field>   & _Guess;
public:

  /////////////////////////////////////////////////////
  // Wrap the usual normal equations trick
  /////////////////////////////////////////////////////
 NormalEquationsTimeslice(SparseMatrixBase<Field> &Matrix, OperatorFunction<Field> &HermitianSolver,
		 LinearFunction<Field> &Guess) 
   :  _Matrix(Matrix), _HermitianSolver(HermitianSolver), _Guess(Guess) {}; 

  void operator() (const Field &in, Field &out){
 
    Field src(in.Grid());
    Field tmp(in.Grid());

    MdagMLinearOperator<SparseMatrixBase<Field>,Field> MdagMOp(_Matrix);
    _Matrix.Mdag(in,src);                   // set b' = A^\dagger b
    _Guess(src,out);
    _HermitianSolver(MdagMOp, src, out);    // solve A^\dagger A psi = b' = A^\dagger b

  }     
};

/**
 * This one is solving M M^dag psi' = b M^dag, psi = M^dag psi'
 */
template<class Field> class NormalResidualTimeslice : public LinearFunction<Field>{
// private:
public:
  SparseMatrixBase<Field> & _Matrix;
  OperatorFunction<Field> & _HermitianSolver;
  LinearFunction<Field>   & _Guess;

public:

  /////////////////////////////////////////////////////
  // Wrap the usual normal equations trick
  /////////////////////////////////////////////////////
 NormalResidualTimeslice(SparseMatrixBase<Field> &Matrix, OperatorFunction<Field> &HermitianSolver,
		 LinearFunction<Field> &Guess) 
   :  _Matrix(Matrix), _HermitianSolver(HermitianSolver), _Guess(Guess) {}; 

  void operator() (const Field &in, Field &out){
 
    Field res(in.Grid());
    Field tmp(in.Grid());

    // std::vector<Field> iterates;

    MMdagLinearOperator<SparseMatrixBase<Field>,Field> MMdagOp(_Matrix);
    _Guess(in,res);
    _HermitianSolver(MMdagOp, in, res);   // Solve M Mdag res = in ;
    _Matrix.Mdag(res,out);                // Set out = Mdag res, and return out as the solution

    // std::vector<Field> iterates;
    // iterates = _HermitianSolver(MMdagOp, in, res);  // M Mdag res = in ;);  // Mdag M out = Mdag in
    // for (int i = 0; i < iterates.size(); i++) {
    //   _Matrix.Mdag(iterates[i], iterates[i]);       // multiply each iterate by M^dag
    // }
    // return iterates;

  }     
};

/**
 * Simple modification of conjugate gradient that outputs the residual as a function 
 * of time, in order to study the large wavelength behavior of the solver. 
 */
// template <class Field, class Action>
template <class Field>
class ConjugateGradientTimeslice : public OperatorFunction<Field> {
public:

  using OperatorFunction<Field>::operator();
  
  bool ErrorOnNoConverge;  // throw an assert when the CG fails to converge.
                           // Defaults true.
  bool MDagNorm;           // whether or not to normalize the error by M^\dagger (default = false).
  RealD Tolerance;
  std::string ResidFileName;
  Integer MaxIterations;
  Integer IterationsToComplete; //Number of iterations the CG took to finish. Filled in upon completion
  RealD TrueResidual;
  SparseMatrixBase<Field>& DiracOp;
  
  ConjugateGradientTimeslice(SparseMatrixBase<Field>& D, RealD tol, std::string filename, Integer maxit, bool err_on_no_conv = true, bool mdag_norm = false)
    : Tolerance(tol),
      ResidFileName(filename),
      MaxIterations(maxit),
      DiracOp (D),
      ErrorOnNoConverge(err_on_no_conv),
      MDagNorm (mdag_norm)
  {};

  virtual void LogIteration(int k,RealD a,RealD b){
    //    std::cout << "ConjugageGradient::LogIteration() "<<std::endl;
  };

  virtual void LogBegin(void){
    std::cout << "ConjugageGradient::LogBegin() "<<std::endl;
  };

  void operator()(LinearOperatorBase<Field> &Linop, const Field &src, Field &psi) {

    this->LogBegin();

    GRID_TRACE("ConjugateGradientTimeslice");
    GridStopWatch PreambleTimer;
    GridStopWatch ConstructTimer;
    GridStopWatch NormTimer;
    GridStopWatch AssignTimer;
    PreambleTimer.Start();
    psi.Checkerboard() = src.Checkerboard();

    conformable(psi, src);

    RealD cp, c, a, d, b, ssq, qq;

    // Was doing copies
    ConstructTimer.Start();
    Field p  (src.Grid());
    Field mmp(src.Grid());
    Field r  (src.Grid());
    ConstructTimer.Stop();

    Field xprev (src.Grid());
    xprev = Zero();

    Field rUpdate (src.Grid());      // container for writing out the updated field.

    OutputFile outFile;
    std::vector<Field> iterates;

    // push back zero vectors for normalization comparison. Want to check that the 
    // zeroth component of the error is the same for CGNR / CGNE. 
    iterates.push_back(xprev);

    // Initial residual computation & set up
    NormTimer.Start();
    ssq = norm2(src);                         // Norm of source vector ||b||^2

    twoNormSlicePushBack<Field>(src, outFile);       // Norm \sum_x |b(x, t)|^2

    RealD guess = norm2(psi);         // Norm of initial guess ||psi||^2
    NormTimer.Stop();
    assert(std::isnan(guess) == 0);
    AssignTimer.Start();
    if ( guess == 0.0 ) {
      r = src;
      p = r;
      a = ssq;
    } else { 
      Linop.HermOpAndNorm(psi, mmp, d, b);        // 
      r = src - mmp;      // Initial residual r0 = b - A guess
      p = r;              // initial conj vector p0 = r0
      a = norm2(p);
    }
    cp = a;
    AssignTimer.Stop();

    // Handle trivial case of zero src
    if (ssq == 0.){
      psi = Zero();
      IterationsToComplete = 1;
      TrueResidual = 0.;
      return;
    }

    Field Adagsrc (src.Grid());
    DiracOp.Mdag(src, Adagsrc);
    std::cout << GridLogMessage << " |src|^2 : " << norm2(src) << std::endl;
    std::cout << GridLogMessage << " |Adag src|^2 : " << norm2(Adagsrc) << std::endl;

    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient: guess " << guess << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:   src " << ssq << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:    mp " << d << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:   mmp " << b << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:  cp,r " << cp << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:     p " << a << std::endl;

    RealD rsq = Tolerance * Tolerance * ssq;

    // Check if guess is really REALLY good :)
    if (cp <= rsq) {
      TrueResidual = std::sqrt(a/ssq);
      std::cout << GridLogMessage << "ConjugateGradient guess is converged already " << std::endl;
      IterationsToComplete = 0;	
      return;
    }

    std::cout << GridLogIterative << std::setprecision(8)
              << "ConjugateGradient: k=0 residual " << cp << " target " << rsq << std::endl;

    PreambleTimer.Stop();
    GridStopWatch LinalgTimer;
    GridStopWatch InnerTimer;
    GridStopWatch AxpyNormTimer;
    GridStopWatch LinearCombTimer;
    GridStopWatch MatrixTimer;
    GridStopWatch SolverTimer;

    RealD usecs = -usecond();
    SolverTimer.Start();
    int k;
    for (k = 1; k <= MaxIterations; k++) {

      GridStopWatch IterationTimer;
      IterationTimer.Start();
      c = cp;

      MatrixTimer.Start();
      Linop.HermOp(p, mmp);         // Computes mmp = Ap
      MatrixTimer.Stop();

      LinalgTimer.Start();

      InnerTimer.Start();
      ComplexD dc  = innerProduct(p,mmp);         // p^\dagger A p
      InnerTimer.Stop();
      d = dc.real();
      a = c / d;

      // updated residual r_{k+1} -> r_k - \alpha * Ap
      rUpdate = r - a * mmp;
      if (MDagNorm) {
        DiracOp.Mdag(rUpdate, rUpdate);     // Scale by Mdag if CGNR (Saad's CGNE)
      }
      twoNormSlicePushBack<Field>(rUpdate, outFile);
      // original code saving slice sum of residual
      // twoNormSlicePushBack<Field>(r - a * mmp, outFile);       // write slice sum of ||r - a A p||^2(t) to out file
      
      // update and save iterate at each step, x_{k+1} -> x_k + \alpha * p
      xprev = xprev + a * p;
      iterates.push_back(xprev);

      // Field tmp = r - a * mmp;// for debug

      // What is axpy? Some accelerator or something? Check Lattice_arith.h
      AxpyNormTimer.Start();

      // axpy_norm computes ax+by for vectors x and y compatible with the computer architecture. Here b is set to 1 (see the function in Lattice_reduction.h). 
      // The first argument passes r by reference, so it stores r --> -a * Ap + 1 * r, i.e. it performs an update on 
      // r_k --> r_{k+1} = r_k - \alpha_k A p_k. The function returns the norm squared of the first variable, i.e. ||r_{k+1}||^2.
      cp = axpy_norm(r, -a, mmp, r);
      
      AxpyNormTimer.Stop();
      b = cp / c;

      LinearCombTimer.Start();
      {
        autoView( psi_v , psi, AcceleratorWrite);
        autoView( p_v   , p,   AcceleratorWrite);
        autoView( r_v   , r,   AcceleratorWrite);
        accelerator_for(ss,p_v.size(), Field::vector_object::Nsimd(),{
            coalescedWrite(psi_v[ss], a      *  p_v(ss) + psi_v(ss));
            coalescedWrite(p_v[ss]  , b      *  p_v(ss) + r_v  (ss));
        });
      }
      LinearCombTimer.Stop();
      LinalgTimer.Stop();
      LogIteration(k,a,b);

      IterationTimer.Stop();
      if ( (k % 500) == 0 ) {
        std::cout << GridLogMessage << "ConjugateGradient: Iteration " << k
                << " residual " << sqrt(cp/ssq) << " target " << Tolerance << std::endl;
      } else { 
        std::cout << GridLogIterative << "ConjugateGradient: Iteration " << k
                << " residual " << sqrt(cp/ssq) << " target " << Tolerance << " took " << IterationTimer.Elapsed() << std::endl;
      }

      // Stopping condition
      if (cp <= rsq) {
        usecs +=usecond();
        SolverTimer.Stop();
        Linop.HermOpAndNorm(psi, mmp, d, qq);     // mmp = A psi, p = A psi - b
        p = mmp - src;
        GridBase *grid = src.Grid();
        RealD DwfFlops = (1452. )*grid->gSites()*4*k
   	               + (8+4+8+4+4)*12*grid->gSites()*k; // CG linear algebra
        RealD srcnorm = std::sqrt(norm2(src));
        RealD resnorm = std::sqrt(norm2(p));
        RealD true_residual = resnorm / srcnorm;
        std::cout << GridLogMessage << "ConjugateGradient Converged on iteration " << k 
          << "\tComputed residual " << std::sqrt(cp / ssq)
          << "\tTrue residual " << true_residual
          << "\tTarget " << Tolerance << std::endl;

        // GridLogMessage logs the message to the terminal output; GridLogPerformance probably writes to a log file?
        //	std::cout << GridLogMessage << "\tPreamble   " << PreambleTimer.Elapsed() <<std::endl;
        std::cout << GridLogMessage << "\tSolver Elapsed    " << SolverTimer.Elapsed() <<std::endl;
        std::cout << GridLogPerformance << "Time breakdown "<<std::endl;
        std::cout << GridLogPerformance << "\tMatrix     " << MatrixTimer.Elapsed() <<std::endl;
        std::cout << GridLogPerformance << "\tLinalg     " << LinalgTimer.Elapsed() <<std::endl;
        std::cout << GridLogPerformance << "\t\tInner      " << InnerTimer.Elapsed() <<std::endl;
        std::cout << GridLogPerformance << "\t\tAxpyNorm   " << AxpyNormTimer.Elapsed() <<std::endl;
        std::cout << GridLogPerformance << "\t\tLinearComb " << LinearCombTimer.Elapsed() <<std::endl;

        std::cout << GridLogDebug << "\tMobius flop rate " << DwfFlops/ usecs<< " Gflops " <<std::endl;

        if (ErrorOnNoConverge) assert(true_residual / Tolerance < 10000.0);

        IterationsToComplete = k;	
        TrueResidual = true_residual;

        // write iterates
        Field dx (src.Grid());
        Field xstar = iterates.back();              // optimal x^*

        std::cout << GridLogMessage << "x[0]: " << norm2(iterates[0]) << std::endl;
        if (MDagNorm) {
            Field Mdag_xstar (src.Grid());
            DiracOp.Mdag(xstar, Mdag_xstar);
            std::cout << GridLogMessage << "x*: " << norm2(Mdag_xstar) << std::endl;
          } else {
            std::cout << GridLogMessage << "x*: " << norm2(xstar) << std::endl;
          }

        std::cout << GridLogMessage << "x*: " << norm2(xstar) << std::endl;


        // if this is zero, then it means everything is doing what we think it should be.
        std::cout << GridLogMessage << "xback - psi = " << norm2(psi - xstar) << " (should be 0) " << std::endl;

        std::cout << "MdagNorm: " << MDagNorm << std::endl;
        for (Field xcur : iterates) {
          dx = xcur - xstar;
          // std::cout << "writing this error" << std::endl;
          if (MDagNorm) {
            // std::cout << "Normalizing by Mdag" << std::endl;
            // DiracOp.Mdag(dx, dx);     // want D^\dagger (x - x*) for CGNR in code
            Field dxOut (src.Grid());
            DiracOp.Mdag(dx, dxOut);     // want D^\dagger (x - x*) for CGNR in code
            twoNormSlicePushBack<Field>(dxOut, outFile);
          } else {
            twoNormSlicePushBack<Field>(dx, outFile);
          }
          // twoNormSlicePushBack<Field>(dx, outFile);
          
        }

        {
          XmlWriter WR(ResidFileName);
          write(WR, "OutputFile", outFile);
          std::cout << GridLogMessage << "Output written to: " << ResidFileName << std::endl;
        }

        return;
      }
    }
    // Failed. Calculate true residual before giving up                                                         
    // Linop.HermOpAndNorm(psi, mmp, d, qq);
    //    p = mmp - src;
    //TrueResidual = sqrt(norm2(p)/ssq);
    //    TrueResidual = 1;

    std::cout << GridLogMessage << "ConjugateGradient did NOT converge "<<k<<" / "<< MaxIterations
    	      <<" residual "<< std::sqrt(cp / ssq)<< std::endl;
    SolverTimer.Stop();
    std::cout << GridLogMessage << "\tPreamble   " << PreambleTimer.Elapsed() <<std::endl;
    std::cout << GridLogMessage << "\tConstruct  " << ConstructTimer.Elapsed() <<std::endl;
    std::cout << GridLogMessage << "\tNorm       " << NormTimer.Elapsed() <<std::endl;
    std::cout << GridLogMessage << "\tAssign     " << AssignTimer.Elapsed() <<std::endl;
    std::cout << GridLogMessage << "\tSolver     " << SolverTimer.Elapsed() <<std::endl;
    std::cout << GridLogMessage << "Solver breakdown "<<std::endl;
    std::cout << GridLogMessage << "\tMatrix     " << MatrixTimer.Elapsed() <<std::endl;
    std::cout << GridLogMessage<< "\tLinalg     " << LinalgTimer.Elapsed() <<std::endl;
    std::cout << GridLogPerformance << "\t\tInner      " << InnerTimer.Elapsed() <<std::endl;
    std::cout << GridLogPerformance << "\t\tAxpyNorm   " << AxpyNormTimer.Elapsed() <<std::endl;
    std::cout << GridLogPerformance << "\t\tLinearComb " << LinearCombTimer.Elapsed() <<std::endl;

    if (ErrorOnNoConverge) assert(0);
    IterationsToComplete = k;

    return;

  }
};

/**
 * Solves D\psi = src twice, first using CGNR and then using CGNE.
 * 
 * Parameters
 * ----------
 * filename (std::string) : name for output file.
 * solveAll (bool) : whether to solve the entire propagator, or just a single component (used for testing).
 */
template<class Action>
void Solve(Action &D, LatticePropagator &source, LatticePropagator &propagatorResid, LatticePropagator &propagatorEq, std::string fnameCGNR, std::string fnameCGNE, GridParallelRNG &RNG5, bool solveAll = true)
{
  GridBase *UGrid = D.GaugeGrid();
  GridBase *FGrid = D.FermionGrid();

  LatticeFermion src4  (UGrid); 
  LatticeFermion src5  (FGrid); 
  LatticeFermion result5Resid(FGrid);
  LatticeFermion result4Resid(UGrid);
  LatticeFermion result5Eq(FGrid);
  LatticeFermion result4Eq(UGrid);

  LatticeFermion diff (UGrid);  // diff between CGNR and CGNE result
  
  ZeroGuesser<LatticeFermion> ZG;

  ConjugateGradientTimeslice<LatticeFermion> CGNR(D, 1.0e-8, fnameCGNR, 100000, true, true);
  ConjugateGradientTimeslice<LatticeFermion> CGNE(D, 1.0e-8, fnameCGNE, 100000, true, false);

  NormalResidualTimeslice<LatticeFermion> normResid (D, CGNR, ZG);
  NormalEquationsTimeslice<LatticeFermion> normEq (D, CGNE, ZG);

  int maxSpin = 1;
  int maxColor = 1;
  if (solveAll) {
    maxSpin = Nd;
    maxColor = Nc;
  }

  for (int s = 0; s < maxSpin; s++){
    for (int c = 0; c < maxColor; c++){
      PropToFerm<Action>(src4, source, s, c);
      D.ImportPhysicalFermionSource(src4, src5);

      // right now: starting from \psi = 0. Maybe start with a random guess?
      result5Resid = Zero();
      result5Eq = Zero();

      // Fill these fields with random Ber(0.5) noise (NO: need to edit the guesser)
      // bernoulli(RNG5, result5Resid);
      // bernoulli(RNG5, result5Eq);

      std::cout << GridLogMessage << "=====BEGIN_CGNR=====" << std::endl;
      std::cout << GridLogMessage << "||guess||^2 = " << norm2(result5Resid) << std::endl;

      normResid(src5, result5Resid);

      std::cout << GridLogMessage
          << "CGNR solve: spin " << s << " color " << c
          << " norm2(src5d) " << norm2(src5)
          << " norm2(result5d) " << norm2(result5Resid) << std::endl;
      
      std::cout << GridLogMessage << "=====END_CGNR=====" << std::endl;

      std::cout << GridLogMessage << "=====BEGIN_CGNE=====" << std::endl;
      std::cout << GridLogMessage << "||guess||^2 = " << norm2(result5Eq) << std::endl;

      normEq(src5, result5Eq);

      std::cout << GridLogMessage
          << "CGNE solve: spin " << s << " color " << c
          << " norm2(src5d) " << norm2(src5)
          << " norm2(result5d) " << norm2(result5Eq) << std::endl;
      std::cout << GridLogMessage << "=====END_CGNE=====" << std::endl;

      D.ExportPhysicalFermionSolution(result5Resid, result4Resid);
      D.ExportPhysicalFermionSolution(result5Eq, result4Eq);

      std::cout << GridLogMessage << "|| CGNR - CGNE ||^2 = " << axpy_norm(diff, -1., result4Resid, result4Eq) << std::endl;

      FermToProp<Action>(propagatorResid, result4Resid, s, c);
      FermToProp<Action>(propagatorEq, result4Eq, s, c);
    }
  }

}

void MesonTrace(std::string file,LatticePropagator &q1,LatticePropagator &q2,LatticeComplex &phase)
{
  const int nchannel=4;
  Gamma::Algebra Gammas[nchannel][2] = {
    {Gamma::Algebra::Gamma5      ,Gamma::Algebra::Gamma5},
    {Gamma::Algebra::GammaTGamma5,Gamma::Algebra::GammaTGamma5},
    {Gamma::Algebra::GammaTGamma5,Gamma::Algebra::Gamma5},
    {Gamma::Algebra::Gamma5      ,Gamma::Algebra::GammaTGamma5}
  };

  Gamma G5(Gamma::Algebra::Gamma5);

  LatticeComplex meson_CF(q1.Grid());
  MesonFile MF;

  for(int ch=0;ch<nchannel;ch++){

    Gamma Gsrc(Gammas[ch][0]);
    Gamma Gsnk(Gammas[ch][1]);

    meson_CF = trace(G5*adj(q1)*G5*Gsnk*q2*adj(Gsrc));

    std::vector<TComplex> meson_T;
    sliceSum(meson_CF,meson_T, Tdir);

    int nt=meson_T.size();

    std::vector<Complex> corr(nt);
    for(int t=0;t<nt;t++){
      corr[t] = TensorRemove(meson_T[t]); // Yes this is ugly, not figured a work around
      // std::cout << " channel "<<ch<<" t "<<t<<" " <<corr[t]<<std::endl;
    }
    MF.data.push_back(corr);
  }

  {
    XmlWriter WR(file);
    write(WR,"MesonFile",MF);
  }
}

int main (int argc, char ** argv)
{
  const int Ls = 8;

  Grid_init(&argc, &argv);

  // freopen("output.txt", "w", stdout);    // write output to the file ./output.txt

  // Double precision grids
  // GridCartesian * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), 
	// 							            GridDefaultSimd(Nd,vComplex::Nsimd()),
	// 							            GridDefaultMpi());
  std::vector<int> lat_size {16, 16, 16, 32};
  std::cout << "Lattice size: " << lat_size << std::endl;
  GridCartesian * UGrid = SpaceTimeGrid::makeFourDimGrid(lat_size, 
								          GridDefaultSimd(Nd,vComplex::Nsimd()),
								          GridDefaultMpi());

  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  //////////////////////////////////////////////////////////////////////
  // You can manage seeds however you like.
  // Recommend SeedUniqueString.
  //////////////////////////////////////////////////////////////////////
  std::vector<int> seeds4({1, 2, 3, 4}); 
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);

  std::vector<int> seeds5({1, 2, 3, 4, 5}); 
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(seeds5);

  std::string outStrStem = "/Users/patrickoare/Dropbox (MIT)/research/multigrid/grid_out/";

  // Read command line input for configuration, or generate a cold start lattice
  LatticeGaugeField Umu(UGrid);
  std::string config;
  if( argc > 1 && argv[1][0] != '-' )
  {
    std::cout<<GridLogMessage << "Loading configuration from " << argv[1] << std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, argv[1]);
    config = argv[1];
    std::cout << "config is: " << config << std::endl;
  }
  else
  {
    // std::cout<<GridLogMessage << "Using hot configuration" <<std::endl;
    SU<Nc>::ColdConfiguration(Umu);
    //    SU<Nc>::HotConfiguration(RNG4,Umu);
    config="HotConfig";
  }

  std::vector<RealD> masses({ 0.01 });
  std::vector<std::string> mlabels ({ "0p01" });
  // std::vector<RealD> masses({ 0.00001, 0.0001, 0.001, 0.01, 0.1});
  // std::vector<std::string> mlabels ({ "0p00001", "0p0001", "0p001", "0p01", "0p1" });

  std::cout << "Masses: " << masses << std::endl;

  int nmass = masses.size();

  std::vector<MobiusFermionD *> FermActs;
  
  std::cout<<GridLogMessage <<"======================"<<std::endl;
  std::cout<<GridLogMessage <<"MobiusFermion action as Scaled Shamir kernel"<<std::endl;
  std::cout<<GridLogMessage <<"======================"<<std::endl;

  for(auto mass: masses) {

    RealD M5=1.0;
    RealD b=1.5;// Scale factor b+c=2, b-c=1
    RealD c=0.5;
    
    FermActs.push_back(new MobiusFermionD(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c));
   
  }

  bool isWall (true);       // use a random Z2 wall source, or a point source at the origin

  std::string srcType ("wall");
  LatticePropagator wall_source(UGrid);
  if (isWall) {
    Z2WallSource  (RNG4, 0, wall_source);
  } else {
    Coordinate coord({0,0,0,0});
    PointSource(coord, wall_source);
    srcType = "pt";
    outStrStem = outStrStem + "pt_src/";
  }
  std::cout << GridLogMessage << "Using a " << srcType << " source." << std::endl;;

  std::vector<LatticePropagator> Z2PropsCGNR(nmass, UGrid);
  std::vector<LatticePropagator> Z2PropsCGNE(nmass, UGrid);

  for(int m=0;m<nmass;m++) {
    std::cout << GridLogMessage << "======================" << std::endl;
    std::cout << GridLogMessage << "Solving for mass m = " << masses[m] << std::endl;

    std::stringstream outStrCGNR, outStrCGNE;
    outStrCGNR << outStrStem << "cgnr_m" << mlabels[m] << ".xml";
    outStrCGNE << outStrStem << "cgne_m" << mlabels[m] << ".xml";

    Solve(*FermActs[m], wall_source, Z2PropsCGNR[m], Z2PropsCGNE[m], outStrCGNR.str(), outStrCGNE.str(), RNG5, false);
    std::cout<<GridLogMessage << "======================" << std::endl;
  }

  LatticeComplex phase(UGrid);
  Coordinate mom({0, 0, 0, 0});
  MakePhase(mom, phase);

  // TODO:
  // See how things change when we add a random volume source
  // Compute condition number of D_dwf
  // Do some low-mode preconditioning and see how it changes. 
  // assert props are equal
  // Condition number for complex matrices? cause D_dwf has negative modes
  
  // TODO don't necessarily need this. Think about what we want to save out
  // for(int m1=0; m1 < nmass; m1++) {
  //   for(int m2 = m1; m2 < nmass; m2++) {
  //     std::stringstream ssp, ssg, ssz;
  //     ssz<<config<< "_m" << m1 << "_m"<< m2 << "_wall_meson.xml";
  //     MesonTrace(ssz.str(), Z2PropsCGNR[m1], Z2PropsCGNR[m2], phase);
  //   }
  // }

  Grid_finalize();
}



