.PHONY: downbkg downsig
# make PASSWORD=XXX downsig1
export OMP_NUM_THREADS=2
spmtPos:=/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J19v1r1-Pre4/offline/Simulation/DetSimV2/DetSimOptions/data/3inch_pos.csv
bkgpath:=$(wildcard /junofs/users/junoprotondecay/zhangaq/data/analysis/BKG/ANRawData_*.root)
bkgs:=$(bkgpath:/junofs/users/junoprotondecay/zhangaq/data/analysis/BKG/ANRawData_%.root=/junofs/users/junoprotondecay/zhangaq/data/analysis/BKGsplit/ANRawData_%.root)
downbkg%:
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:~/offline3745/BKG/ana/RawData/ANRawData_$*.root BKG/
downsig%:
	#sshpass -p juno2020@ scp lxslc701.ihep.ac.cn:~/offline3745/SIG_LSPMT/ana/RawData/PDRawData_$*.root SIG_LSPMT/
	sshpass -p juno2020@ scp lxslc701.ihep.ac.cn:~/offline3745/SIG_newGen/ana/RawData/PDRawData_$*.root SIG_LSPMT/

downnewbkg:
	sshpass -p juno2020@ scp lxslc701.ihep.ac.cn:~/offline3745/Fitting/BKG/ana-result/PDCandidateFitInfor-03.root BKG/
downnewsig:
	sshpass -p juno2020@ scp lxslc701.ihep.ac.cn:~/offline3745/Fitting/SIG/ana-result/PDNewData.root SIG_LSPMT/

downnewsig1:
	mkdir -p SIG_LSPMT
	# sshpass -p juno2020@ scp lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation_result/Fitting/SIG/ana-result-2/Data-0301/PDNewData_A_0.root SIG_LSPMT/
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation_result/Fitting_latestGen/SIG/ana-result/Data-0818/PDNewData_A_0.root SIG_LSPMT/
downnewbkg1:
	mkdir -p BKGnew
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation_result/Fitting/BKG/ana-result-2/Data-0301/* BKGnew/

# split the root file to support parallel
bkgsplit: $(bkgs)
BKGsplit/ANRawData_%.root: BKG/ANRawData_%.root
	mkdir -p $(dir $@)
	root -lbq splitroot.C\(\"$<\",\"$@\",10000,$**40000\)
SIG_LSPMT/%.png: SIG_LSPMT/PDRawData_0.root
	python3 plotevt.py -i $* -p $^ -o $@ 

SIG_test.root: /junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation/K+/ptarget
	mkdir $@
	hadd 
BKG_test/ANRawData_0.root: $(wildcard /junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation/DSNB-NC/ptarget/*/summary.root)
	mkdir -p $(dir $@)
	hadd -f $@ $^
BKG_test/ANRawData_1.root: $(wildcard /junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation/DSNB-NC/ptarget/*/summary.root)
	root -lbq mergeRoot.C\(100000,200000,1000,\"$@\",\"/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation/DSNB-NC/ptarget/\%d/summary.root\"\) 
spmtGeometry.txt: $(spmtPos)
	python3 writeGeoTxt.py -g $^ -o $@
.DELETE_ON_ERROR:
.SECONDARY:
