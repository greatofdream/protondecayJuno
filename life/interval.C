#include "TFeldmanCousins.h"
#include "TCanvas.h"
#include "TMath.h"
#include <vector>
#include "TTree.h"
using namespace std;
struct IntervalUL{
    Double_t up;
    Double_t low;
};
void interval(Int_t bkg=3, Double_t cl=0.9, Double_t maxvalue=10, Double_t step=0.001, Bool_t ifplot=true){
    // the bayes, fc, cls result in some bkg
    Int_t size = (Int_t)maxvalue/step;
    Double_t* observe = new Double_t[size];
    for(Int_t i=0;i<size;i++){
        observe[i] = step*i;
    }
    // new the vector for upper and lower limit for interval
    Double_t* fcUp = new Double_t[size];
    Double_t* fcLow = new Double_t[size];

    // caculate the FC interval
    TFeldmanCousins fc;
    for(Int_t i=0;i<size;i++){
        fcUp[i] = fc.CalculateUpperLimit(observe[i], bkg);
        fcLow[i] = fc.GetLowerLimit();
        //cout<<fcUp[i]<<" "<<fcLow[i]<<endl;
    }
    // plot part
    if(ifplot){
        cout<<"begin plot"<<endl;
        TCanvas* c1 = new TCanvas("interval", "", 800, 600);
        TGraph *fcgrUp = new TGraph(size, observe, fcUp);
        fcgrUp->SetTitle(TString::Format("fc interval bkg:%d", bkg));
        fcgrUp->Draw("AL");
        fcgrUp->GetXaxis()->SetTitle("observe");
        fcgrUp->GetYaxis()->SetTitle("estimate");
        fcgrUp->GetYaxis()->SetRangeUser(0,maxvalue+bkg);
        fcgrUp->GetXaxis()->SetRangeUser(0,maxvalue);
        fcgrUp->GetXaxis()->CenterTitle();
        fcgrUp->GetYaxis()->CenterTitle();
        c1->Update();
        c1->SaveAs(TString::Format("interval_bkg%d.png", bkg));
    }
}
void visualInterval(TString filename="interval_bkg1.root"){
    vector<Double_t>* itulUp = nullptr;
    vector<Double_t>* itulLow = nullptr;
    vector<Double_t>* observe = nullptr;
    Int_t bkg;
    Double_t maxvalue=10;
    Int_t size =10000;
    TFile* f = TFile::Open(filename);
    TTree* tfc = (TTree*)f->Get("fc");
    tfc->SetBranchAddress("up", &itulUp);
    tfc->SetBranchAddress("up", &itulUp);
    tfc->SetBranchAddress("bkg", &bkg);
    tfc->SetBranchAddress("observe", &observe);
    tfc->GetEntry(0);
    cout<<"begin plot"<<endl;
    cout<<bkg<<" "<<itulUp->size()<<" "<<observe->size()<<endl;
    TCanvas* c1 = new TCanvas("interval", "", 800, 600);
    TMultiGraph *mg = new TMultiGraph();
    TGraph *fcgrUp = new TGraph(size, &(*observe)[0], &(*itulUp)[0]);
    fcgrUp->SetTitle(TString::Format("fc interval bkg:%d", bkg));
    //fcgrUp->Draw("AL");
    fcgrUp->GetXaxis()->SetTitle("observe");
    fcgrUp->GetYaxis()->SetTitle("estimate");
    //fcgrUp->GetYaxis()->SetRangeUser(0,maxvalue+bkg);
    fcgrUp->GetXaxis()->SetRangeUser(0,maxvalue);
    fcgrUp->GetXaxis()->CenterTitle();
    fcgrUp->GetYaxis()->CenterTitle();
    // draw bayes
    // draw
    c1->Update();
    c1->SaveAs(TString::Format("interval_bkg%d.png", bkg));
}

void fcStore(TFile* f, Double_t step=0.1, Double_t bkg=1, Int_t size =30000, Double_t maxvalue=10){
    TFeldmanCousins fc;
    TTree *t1 = new TTree("fc", "fc interval");
    vector<Double_t> itulUp(size);
    vector<Double_t> itulLow(size);
    vector<Double_t> observe(size);
    Double_t sensitivity = 0;
    t1->Branch("up", &itulUp);
    t1->Branch("low", &itulLow);
    t1->Branch("observe", &observe);
    t1->Branch("bkg", &bkg);
    t1->Branch("maxvalue", &maxvalue);
    t1->Branch("sensitivity", &sensitivity);
    for(Int_t i=0;i<size;i++){
        observe[i] = step*i;
        itulUp[i] = fc.CalculateUpperLimit(observe[i], bkg);
        itulLow[i] = fc.GetLowerLimit();
        if(i%10==0){//control the observe[i] is integer
            sensitivity += TMath::Poisson(observe[i],bkg)*itulUp[i];
        }
    }
    t1->Fill();
    t1->Write();
}
Double_t bayesCacu(Double_t beta=0.05, Double_t mub=1, Double_t obs=0){
    Double_t bcacu = 0;
    Double_t left =0, right =100;
    Double_t step=1;
    Int_t times =0;
    Double_t mus = 50;
    while(abs(beta-bcacu)>0.000001 && times<100){
         bcacu = (1-ROOT::Math::chisquared_cdf(2*(mub+mus),2*(obs+1)))/(1-ROOT::Math::chisquared_cdf(2*(mub),2*(obs+1)));

        if(beta > bcacu){
            right = mus;
        }
        else{
            left = mus;
        }
        mus = (right+left)/2;
        times++;
    }
    //cout<<"bc:"<<bcacu<<"mus"<<mus<<"times"<<times<<endl;
    return mus;
}
void bayesStore(TFile* f, Double_t step=0.001, Int_t bkg=1, Int_t size =30000, Double_t maxvalue=10){
    TTree *t1 = new TTree("bayes", "bayes interval");
    vector<Double_t> itulUp(size);
    vector<Double_t> itulLow(size);
    vector<Double_t> observe(size);
    t1->Branch("up", &itulUp);
    t1->Branch("low", &itulLow);
    t1->Branch("observe", &observe);
    t1->Branch("bkg", &bkg);
    t1->Branch("maxvalue", &maxvalue);
    for(Int_t i=0;i<size;i++){
        observe[i] = step*i;
        itulUp[i] = bayesCacu(0.05, bkg, observe[i]);
        itulLow[i] = 0;
    }
    t1->Fill();
    t1->Write();
}
void clsStore(TFile* f, Double_t step=0.001, Int_t bkg=1, Int_t size =10000, Double_t maxvalue=10){
    TTree *t1 = new TTree("CLs", "CLs interval");
    vector<Double_t> itulUp(size);
    vector<Double_t> itulLow(size);
    vector<Double_t> observe(size);
    t1->Branch("up", &itulUp);
    t1->Branch("low", &itulLow);
    t1->Branch("observe", &observe);
    t1->Branch("bkg", &bkg);
    t1->Branch("maxvalue", &maxvalue);
    for(Int_t i=0;i<size;i++){
        observe[i] = step*i;
        itulUp[i] = bayesCacu(0.05, bkg, observe[i]-1);;
        itulLow[i] = 0;
    }
    t1->Fill();
    t1->Write();
}
void fcData(float bkg=0.008438, TString filename="interval_bkg0.008438.root"){
    TFile *f = new TFile(filename, "UPDATE");
    fcStore(f, 0.1, bkg, 200);
}
void storeData(TString filename="interval_bkg1.root"){
    TFile *f = new TFile(filename, "UPDATE");
    //fcStore(f);
    bayesStore(f);
    clsStore(f);
}