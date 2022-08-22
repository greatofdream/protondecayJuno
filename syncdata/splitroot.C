#include <iostream>
#include <TMath.h>
#include <TString.h>
#include <TFile.h>
#include <TTree.h>
using namespace std;
void splitroot(TString inputfile, TString outputfile, Int_t maxEventNum, Int_t offset){
    TFile * fin = new TFile(inputfile);
    TTree * Tin = (TTree*)fin->Get("evtinfo");

    Int_t entries = Tin->GetEntries();
    TFile * fout = 0;
    TTree * Tout = 0;
    cout<<"entries: "<<entries<<endl;
    TString outtmp = outputfile;
    for(Int_t i=0;(i*maxEventNum)<entries;i++){
        outtmp = outputfile;
        outtmp = outtmp.Insert(outputfile.Length()-5,TString::Format("%d",i));
        fout = new TFile(outtmp, "recreate");
        cout<<outtmp<<endl;
        Tout = Tin->CopyTree(TString::Format("(evtID<%d)&(evtID>=%d)",(i+1)*maxEventNum+offset,i*maxEventNum+offset));
        fout->cd();
        Tout->Write("evtinfo");
        fout->Close();
    }
    cout<<"end: "<<inputfile<<endl;
    return;
}