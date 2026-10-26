/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./examples/Example_mg_hmc_tests.cc

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
// Example_mg_hmc_tests.cc
//
// Unit tests for the pieces in Example_mg_hmc.h, on a small lattice (8^4,
// Ls = 8, tepid gauge field) so they run on a laptop. Each test prints
//   MGHMC-TEST <name> PASS|FAIL <value>
// and the program exits non-zero if any test fails.
//
#include "Example_mg_common.h"
#include "Example_mg_hmc.h"

using namespace std;
using namespace Grid;

static int nfail = 0;

void Check(const std::string &name, bool pass, RealD value)
{
  std::cout<<GridLogMessage<<"MGHMC-TEST "<<name<<" "<<(pass ? "PASS" : "FAIL")<<" "<<value<<std::endl;
  if ( !pass ) nfail++;
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  const int Ls = 8;
  std::vector<int> lat_size {8, 8, 8, 8};
  RealD mass = 0.01, M5 = 1.8, mob_b = 1.5, mob_c = 0.5;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(lat_size, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers(seeds5);

  LatticeGaugeField Umu(UGrid);
  SU<Nc>::TepidConfiguration(RNG4,Umu);

  typedef PVdagMLinearOperator<MobiusFermionD,LatticeFermionD> PVdagM_t;
  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
  MobiusFermionD Dpv (Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
  PVdagM_t PVdagM(Ddwf,Dpv);

  LatticeFermion x(FGrid), y(FGrid), z(FGrid);

  //////////////////////////////////////////////////////////////////////////
  // counter: Op and AdjOp count 1, HermOp counts 2, attributed by category
  //////////////////////////////////////////////////////////////////////////
  {
    MatvecCounter counter;
    CountingLinearOperator<LatticeFermion> CountedOp(PVdagM,counter);
    gaussian(RNG5,x);
    counter.Set("a");
    for(int i=0;i<3;i++) CountedOp.Op(x,y);
    for(int i=0;i<2;i++) CountedOp.HermOp(x,y);
    counter.Set("b");
    CountedOp.Op(x,y);
    bool pass = counter.Get("a")==7 && counter.Get("b")==1 && counter.Total()==8;
    Check("counter",pass,(RealD)counter.Total());

    // The coarse tally is separate and never enters Total()
    CountingLinearOperator<LatticeFermion> CoarseCounted(PVdagM,counter,true);
    CoarseCounted.Op(x,y);
    CoarseCounted.HermOp(x,y);
    pass = counter.GetCoarse("b")==3 && counter.TotalCoarse()==3 && counter.Total()==8;
    Check("counter_coarse",pass,(RealD)counter.TotalCoarse());
  }

  //////////////////////////////////////////////////////////////////////////
  // projector: Q V = 0, Q^2 = Q, Q x orthogonal to V
  //////////////////////////////////////////////////////////////////////////
  std::vector<LatticeFermion> V(4,FGrid);
  for(auto &v : V) gaussian(RNG5,v);
  {
    SpanProjector<LatticeFermion> Q(V);
    RealD worst = 0.0;
    for(auto &v : V){
      Q.Complement(v,y);
      worst = std::max(worst,std::sqrt(norm2(y)/norm2(v)));
    }
    Check("projector_kills_span",Q.Ok() && worst<1.0e-12,worst);

    gaussian(RNG5,x);
    Q.Complement(x,y);
    Q.Complement(y,z);
    z = z - y;
    RealD idem = std::sqrt(norm2(z)/norm2(x));
    Check("projector_idempotent",idem<1.0e-12,idem);

    RealD overlap = 0.0;
    for(auto &v : V) overlap = std::max(overlap,(RealD)abs(innerProduct(v,y))/std::sqrt(norm2(x)*norm2(v)));
    Check("projector_orthogonal",overlap<1.0e-12,overlap);
  }

  //////////////////////////////////////////////////////////////////////////
  // projector_degenerate: nearly dependent vectors are reported, not fatal
  //////////////////////////////////////////////////////////////////////////
  {
    std::vector<LatticeFermion> W = V;
    gaussian(RNG5,x);
    W[3] = W[2] + 1.0e-15 * x;
    SpanProjector<LatticeFermion> Q(W);
    bool pass = (!Q.Ok()) || (Q.GramCondition() > 1.0e12);
    Check("projector_degenerate",pass,Q.GramCondition());
  }

  //////////////////////////////////////////////////////////////////////////
  // projected_op: range of Q A Q lies in the complement of V
  //////////////////////////////////////////////////////////////////////////
  {
    SpanProjector<LatticeFermion> Q(V);
    ProjectedLinearOperator<LatticeFermion> QAQ(Q,PVdagM);
    gaussian(RNG5,x);
    QAQ.Op(x,y);
    Q.Complement(y,z);
    z = z - y;
    RealD leak = std::sqrt(norm2(z)/norm2(y));
    Check("projected_op",leak<1.0e-12,leak);
  }

  //////////////////////////////////////////////////////////////////////////
  // integrator_equivalence: stepping with a hook reproduces integrate()
  // bit for bit, and the hook sees steps 1..MDsteps
  //////////////////////////////////////////////////////////////////////////
  {
    typedef PeriodicGimplR Gimpl;
    typedef MinimumNorm2<Gimpl,NoSmearing<Gimpl>> IntegratorType;
    IwasakiGaugeActionR GaugeAction(2.13);
    ActionLevel<LatticeGaugeField> Level1(1);
    Level1.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField,NoHirep> FullSet;
    FullSet.push_back(Level1);
    IntegratorParameters MDpar(5,0.1);
    NoSmearing<Gimpl> Smearing;
    IntegratorType MD(UGrid,MDpar,FullSet,Smearing);

    std::vector<int> seedsP({11,12,13,14});
    LatticeGaugeField U1 = Umu, U2 = Umu;
    {
      GridSerialRNG sRNG; sRNG.SeedFixedIntegers(seedsP);
      GridParallelRNG pRNG(UGrid); pRNG.SeedFixedIntegers(seedsP);
      MD.refresh(U1,sRNG,pRNG);
      MD.integrate(U1);
    }
    std::vector<int> hooked;
    {
      GridSerialRNG sRNG; sRNG.SeedFixedIntegers(seedsP);
      GridParallelRNG pRNG(UGrid); pRNG.SeedFixedIntegers(seedsP);
      MD.refresh(U2,sRNG,pRNG);
      IntegrateWithHook(MD,U2,[&](int s){ hooked.push_back(s); });
    }
    LatticeGaugeField diff = U1 - U2;
    RealD d = norm2(diff);
    bool steps_ok = hooked.size()==5;
    for(int s=0;s<(int)hooked.size();s++) steps_ok = steps_ok && hooked[s]==s+1;
    Check("integrator_equivalence",d==0.0 && steps_ok && norm2(U1-Umu)>0.0,d);

    GridSerialRNG sRNG; sRNG.SeedFixedIntegers(seedsP);
    bool m_ok = MetropolisAccept(-1.0,sRNG) && !MetropolisAccept(1.0e6,sRNG);
    Check("metropolis_extremes",m_ok,0.0);
  }

  //////////////////////////////////////////////////////////////////////////
  // gauge_rotation: under U -> U^g the exact basis is g V, so the update
  // must remove most of the frozen basis's capture loss, and its error
  // should fall as eps^2 faster than the frozen one's (ratio ~16 vs ~4 when
  // eps halves) if it is correct to first order.
  //////////////////////////////////////////////////////////////////////////
  {
    const int nb = 8;
    typedef Aggregation<vSpinColourVector,vTComplex,nb>               Subspace;
    typedef DeprecatedGeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nb>    LittleDiracOperator;
    typedef LittleDiracOperator::CoarseVector                         CoarseVector;
    typedef MGPreconditioner<vSpinColourVector,vTComplex,nb>          TwoLevelMG;
    typedef ShiftedPVdagMLinearOperator<MobiusFermionD,LatticeFermionD> ShiftedPVdagM_t;

    Coordinate clatt = lat_size;
    for(int d=0;d<clatt.size();d++) clatt[d] = clatt[d]/2;
    GridCartesian *Coarse4d = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
    GridCartesian *Coarse5d = SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);
    NextToNearestStencilGeometry5D geom(Coarse5d);

    // Basis bred at U, kept as bred
    Subspace Agg(Coarse5d,FGrid,0);
    Agg.CreateSubspaceGCR(RNG5,PVdagM,nb);
    std::vector<LatticeFermion> V = Agg.subspace;

    // Two-level MG for A(U), as in Example_mg_evolution.cc
    LittleDiracOperator LDOp(geom,FGrid,Coarse5d);
    LDOp.CoarsenOperator(PVdagM,Agg);
    ShiftedPVdagM_t ShiftedPVdagM(0.01,Ddwf,Dpv);
    TrivialPrecon<LatticeFermionD> simpleFine;
    TrivialPrecon<CoarseVector>    simpleCoarse;
    NonHermitianLinearOperator<LittleDiracOperator,CoarseVector> LinOpCoarse(LDOp);
    PrecGeneralisedConjugateResidualNonHermitian<CoarseVector>   L2PGCR(3.0e-2,100,LinOpCoarse,simpleCoarse,10,10);
    PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> SmootherGCR(0.01,1,ShiftedPVdagM,simpleFine,16,16);
    TwoLevelMG MG(Agg,PVdagM,simpleFine,SmootherGCR,LinOpCoarse,L2PGCR);

    // Capture loss of the truth W by the block span of B
    auto Loss = [&](std::vector<LatticeFermion> &B, std::vector<LatticeFermion> &W) -> RealD {
      Subspace AggB(Coarse5d,FGrid,0);
      AggB.subspace = B;
      AggB.Orthogonalise();
      return 1.0 - ReportCapture("gauge_rotation",AggB,W,nb);
    };

    std::vector<RealD> eps({0.05,0.025});
    for(auto form : {CorrectionForm::Projected, CorrectionForm::Plain}){
      std::string fname = (form==CorrectionForm::Projected) ? "projected" : "plain";
      std::vector<RealD> Lf, Lu;
      for(RealD e : eps){
        LatticeColourMatrix X(UGrid), g(UGrid);
        GridParallelRNG RNGg(UGrid); RNGg.SeedFixedIntegers(std::vector<int>({31,32,33,34}));
        SU<Nc>::GaussianFundamentalLieAlgebraMatrix(RNGg,X);
        g = expMat(X,e);

        LatticeGaugeField Ug = Umu;
        SU<Nc>::GaugeTransform<PeriodicGimplR>(Ug,g);
        MobiusFermionD Ddwfg(Ug,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
        MobiusFermionD Dpvg (Ug,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
        PVdagM_t PVdagMg(Ddwfg,Dpvg);

        std::vector<LatticeFermion> Vg = V;
        for(auto &v : Vg) GaugeTransformFermion5d(v,g);

        std::vector<LatticeFermion> Vu = V;
        MatvecCounter counter;
        UpdateStats st = PerturbativeUpdate(Vu,PVdagMg,PVdagM,MG,1.0e-8,200,form,counter);

        Lf.push_back(Loss(V,Vg));
        Lu.push_back(Loss(Vu,Vg));
        std::cout<<GridLogMessage<<"MGHMC-TEST gauge_rotation "<<fname<<" eps "<<e<<" L_frozen "<<Lf.back()
		 <<" L_updated "<<Lu.back()<<" resid "<<st.resid_before<<" -> "<<st.resid_after<<std::endl;
        Check("gauge_rotation_resid_"+fname+"_eps"+std::to_string(e),st.resid_after<st.resid_before,st.resid_after);
      }
      std::cout<<GridLogMessage<<"MGHMC-TEST gauge_rotation "<<fname<<" ratio frozen "<<Lf[0]/Lf[1]
	       <<" updated "<<Lu[0]/Lu[1]<<std::endl;
      // Only the projected form is required to remove most of the loss; the
      // plain form is a comparison (it removes ~2/3 at eps = 0.05) and is
      // reported, not checked.
      if (form==CorrectionForm::Projected) {
        Check("gauge_rotation_"+fname,Lu[0]<0.1*Lf[0],Lu[0]/Lf[0]);
      } else {
        std::cout<<GridLogMessage<<"MGHMC-TEST gauge_rotation_"<<fname<<" INFO "<<Lu[0]/Lf[0]<<std::endl;
      }
    }

    //////////////////////////////////////////////////////////////////////////
    // update_cap: a correction solve that hits its cap is still applied
    //////////////////////////////////////////////////////////////////////////
    {
      LatticeColourMatrix X(UGrid), g(UGrid);
      GridParallelRNG RNGg(UGrid); RNGg.SeedFixedIntegers(std::vector<int>({31,32,33,34}));
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(RNGg,X);
      g = expMat(X,0.05);
      LatticeGaugeField Ug = Umu;
      SU<Nc>::GaugeTransform<PeriodicGimplR>(Ug,g);
      MobiusFermionD Ddwfg(Ug,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c);
      MobiusFermionD Dpvg (Ug,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.0 ,M5,mob_b,mob_c);
      PVdagM_t PVdagMg(Ddwfg,Dpvg);

      std::vector<LatticeFermion> Vu = V;
      MatvecCounter counter;
      UpdateStats st = PerturbativeUpdate(Vu,PVdagMg,PVdagM,MG,1.0e-8,1,CorrectionForm::Projected,counter);
      RealD change = 0.0;
      for(int b=0;b<nb;b++){ LatticeFermion d = Vu[b]-V[b]*std::pow(norm2(V[b]),-0.5); change += norm2(d); }
      Check("update_cap",st.capped==nb && change>0.0,(RealD)st.capped);
    }
  }

  std::cout<<GridLogMessage<<"MGHMC-TEST failures "<<nfail<<std::endl;
  Grid_finalize();
  return nfail==0 ? 0 : 1;
}
