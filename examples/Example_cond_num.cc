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

/**
 * Computes the condition number of the Dirac operator, given an action.
 * 
 * Parameters
 * ----------
 * filename (std::string) : name for output file.
 * solveAll (bool) : whether to solve the entire propagator, or just a single component (used for testing).
 */
/*
template<class Action>
void ConditionNumber(Action &D, LatticePropagator &source, LatticePropagator &propagatorResid, LatticePropagator &propagatorEq, std::string fnameCGNR, std::string fnameCGNE, bool solveAll = true)
{

  SchurDiagMooeeOperator<MobiusFermionD, LatticeFermion> DHerm(D);
  LatticeFermion    src(FrbGrid);
  random(RNG5,src);
  PowerMethod<LatticeFermionD> pm;
  pm(DHerm, );
  
  GridBase *UGrid = D.GaugeGrid();
  GridBase *FGrid = D.FermionGrid();

  LatticeFermion src4  (UGrid); 
  LatticeFermion src5  (FGrid); 
  LatticeFermion result5Resid(FGrid);
  LatticeFermion result4Resid(UGrid);
  LatticeFermion result5Eq(FGrid);
  LatticeFermion result4Eq(UGrid);

  // TODO function stub

}
*/

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
  std::vector<int> seeds4({1,2,3,4}); 
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);

  std::vector<int> seeds5({5,6,7,8});
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
    std::cout<<GridLogMessage << "Using hot configuration" <<std::endl;
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

  for(int m=0;m<nmass;m++) {
    std::cout << GridLogMessage << "======================" << std::endl;
    std::cout << GridLogMessage << "Computing max eigenvalue for mass m = " << masses[m] << std::endl;
    std::cout << GridLogMessage << "======================" << std::endl;

    // ConditionNumber(*FermActs[m], *FrbGrid);

    SchurDiagMooeeOperator<MobiusFermionD, LatticeFermion> DHerm(*FermActs[m]);
    LatticeFermion src(FrbGrid);
    random(RNG5, src);

    // Use power method to compute largest eval
    PowerMethod<LatticeFermionD> pm;
    RealD lambdaMax = pm(DHerm, src);

    // Run power method on Dherm - lambdaMax * id
    // MobiusFermionD tmp;
    // tmp = lambdaMax * FermActs[m];
    // DHerm = DHerm * lambdaMax;
    // MobiusFermionD idAction;
    // SchurDiagMooeeOperator<MobiusFermionD, LatticeFermion> shifted = DHerm - lambdaMax * idAction;
    
  }

  // TODO:
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



