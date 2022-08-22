.PHONY: all
export OMP_NUM_THREADS=2
pathPrefix:=../syncdata
srcPrefix:=../crosscheckNewFit
pythonPrefix:=/home/greatofdream/python/bin
pdpath:=$(wildcard $(pathPrefix)/SIG_LSPMT/PDRawData_*.root)
bkgpath:=$(wildcard $(pathPrefix)/BKGsplit/ANRawData_*.root)
pdnewpath:=$(wildcard $(pathPrefix)/SIG_LSPMT/PDRawData_A_*.root)
# t90 part
t90log:=""
## cut length 500
t90/bkgT90C500Delta0-.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/t90/T90C500Delta0_%.h5)
t90/sigT90C500Delta0-.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C500Delta0_%.h5)
sig/t90/T90C500Delta0_%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
	mkdir -p $(dir $@)
	python3 $(srcPrefix)/t90.py -o $@ -p $^ -l $(t90log) -d 0 -c 500 >$@.log 2>&1 
bkg/t90/T90C500Delta0_%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
	mkdir -p $(dir $@)
	python3 $(srcPrefix)/t90.py -o $@ -p $^ -l $(t90log) -d 0 -c 500 >$@.log 2>&1
## compare with main group
sig/t90/compare.pdf: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C500Delta0_%.h5)
	python3 t90compare.py -i $(pdpath) -c $^ -o $@ >$@.log 2>&1
compare/t90/sigbkg.png:
	python3 t90cmp.py -i $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C500Delta0_%.h5) -c $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/t90/T90C300Delta0_%.h5) -o $@
## cut length 300
t90/bkgT90C300Delta0-.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/t90/T90C300Delta0_%.h5)
t90/sigT90C300Delta0-.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C300Delta0_%.h5)
sig/t90/T90C300Delta0_%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
	mkdir -p $(dir $@)
	python3 t90.py -o $@ -p $^ -l $(t90log) -d 0 -c 300 >$@.log 2>&1 
bkg/t90/T90C300Delta0_%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
	mkdir -p $(dir $@)BKGsplit
	python3 t90.py -o $@ -p $^ -l $(t90log) -d 0 -c 300 >$@.log 2>&1
bkg/t90/compareC300.pdf: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/t90/T90C300Delta0_%.h5)
	python3 t90compare.py -i $(bkgpath) -c $^ -o $@ >$@.log 2>&1
sig/t90/compareC300.pdf: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C300Delta0_%.h5)
	python3 t90compare.py -i $(pdpath) -c $^ -o $@ >$@.log 2>&1

# michel part
michel/bkg_.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/michel/%.h5)
michel/sig_.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/michel/.h5)
sig/michel/%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
	mkdir -p $(dir $@)
	python3 getMichelInfo.py -o $@ -i $^ -b 150 -e 10000 >$@.log 2>&1
bkg/michel/%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
	mkdir -p $(dir $@)
	python3 getMichelInfo.py -o $@ -i $^ -b 150 -e 10000 >$@.log 2>&1
# shape feature part
shapefeature/bkgC300Delta0-.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/shapefeature/C300Delta0_%.h5)
shapefeature/sigC300Delta0-.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/shapefeature/C300Delta0_%.h5)
bkg/shapefeature/C300Delta0_%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
	mkdir -p $(dir $@)
	python3 shapefeature.py -o $@ -p $^ -l $(t90log) -d 0 -c 300 >$@.log 2>&1
sig/shapefeature/C300Delta0_%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
	mkdir -p $(dir $@)
	python3 shapefeature.py -o $@ -p $^ -l $(t90log) -d 0 -c 300 >$@.log 2>&1
# fit part
tplfile:=../protondecay/tplp/e-/tpl1ns.h5
# fit300/bkg1nsInit-.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/fit300/1nsInit-%.h5)
# fit300/sig1nsInit-.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/fit300/1nsInit-%.h5)
# bkg/fit300/1nsInit-%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
# 	mkdir -p $(dir $@)
# 	python3 fitEnergyInit.py -f 300 -s 1 -b 1 -t $(tplfile) -p $^ -o $@ >$@.log 2>&1
# sig/fit300/1nsInit-%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
# 	mkdir -p $(dir $@)
# 	python3 fitEnergyInit.py -f 300 -s 1 -b 1 -t $(tplfile) -p $^ -o $@ >$@.log 2>&1
# efficiency
# bkg/eff.pdf: bkg/fit300/res.h5
# 	mkdir -p $(dir $@)
# 	python3 checkEffi.py -i $^ -p $(bkgpath) -o $@ >$@.log 2>&1
# sig/eff.pdf: sig/fit300/res.h5
# 	mkdir -p $(dir $@)
# 	python3 checkEffi.py -i $^ -p $(pdpath) -o $@ >$@.log 2>&1

antpl:=../reference/newRef/anTpl.h5
# fit300Dtpl/bkg1nsInit-.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=fit300Dtpl/bkg1nsInit-%.h5)
# fit300Dtpl/sig1nsInit-.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=fit300Dtpl/sig1nsInit-%.h5)
# fit300Dtpl/bkg1nsInit-%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
# 	mkdir -p $(dir $@)
# 	python3 fitEnergyInit.py -f 300 -s 1 -b 1 -t $(tplfile) -a $(antpl) -p $^ -o $@ >$@.log 2>&1
# fit300Dtpl/sig1nsInit-%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
# 	mkdir -p $(dir $@)
# 	python3 fitEnergyInit.py -f 300 -s 1 -b 1 -t $(tplfile) -a $(antpl) -p $^ -o $@ >$@.log 2>&1
ktpl:=../reference/newRef/kTpl.h5
pitpl:=../reference/newRef/piTpl.h5
mutpl:=../reference/newRef/muTpl.h5
fit300Mtpl/bkg1nsInit-.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/fit300Mtpl/1nsInit-%.h5)
fit300Mtpl/sig1nsInit-.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/fit300Mtpl/1nsInit-%.h5)
bkg/fit300Mtpl/1nsInit-%.h5: $(pathPrefix)/BKGsplit/ANRawData_%.root
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 fitEnergy.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
sig/fit300Mtpl/1nsInit-%.h5: $(pathPrefix)/SIG_LSPMT/PDRawData_%.root
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 fitEnergy.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
# check the fit result
checkFit/bkg1nsInit-.h5: $(bkgpath)
	mkdir -p $(dir $@)
	python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 54233 66181 134450 152062 156212 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkFit/sig1nsInit-.h5: $(pdpath)
	mkdir -p $(dir $@)
	python3 plotFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 1 2 3 4 5 6 7 8 9 10 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/bkg1nsInit-.h5: $(bkgpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 54233 66181 134450 146891 152062 156212 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/bkg1nsInit1.h5: $(bkgpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 108 1033 1256 23737 68387 96295 159766 159801 159900 77714 100790 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/bkg1nsInit2.h5: $(bkgpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 24942  26949  39655  54196  79883  81230 122218 130778 139250 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/bkg1nsInit3.h5: $(bkgpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 plotFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 96295 146891 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1

checkZoom/sig1nsInit-.h5: $(pdpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 plotFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 1 2 3 4 5 6 7 8 9 10 347 5796 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/sig1nsInit1.h5: $(pdpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 308 347 1960 2600 2876  4173 4563 4942 7202 7764 7911 8184 8421 8463 8478 8658 8753 9406 9469 9878 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
	#$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 308 1960  -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/sig1nsInit2.h5: $(pdpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 1409 5074 6296 8023 9034 9753 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1
checkZoom/sig1nsInit3.h5: $(pdpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 547  785 1176 1183 1238 1258 1597 1933 -m $(mutpl) -a $(antpl) -i $^ -o $@ >$@.log 2>&1

gyhR:=/junofs/users/junoprotondecay/zhangaq/data/analysis
checkZoom/bkgguoyh1.h5: $(bkgpath)
	mkdir -p $(dir $@)
	python3 checkGuoFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 108 1033 1256 23737 68387 77714 100790 159766 159801 159900   -m $(mutpl) -a $(antpl) -i $^ -o $@ -g $(gyhR)/BKG/PDCandidateFitInfor-03.root >$@.log 2>&1
checkZoom/bkgguoyh.h5: $(bkgpath)
	mkdir -p $(dir $@)
	python3 checkGuoFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 54233 66181 134450 152062 156212 -m $(mutpl) -a $(antpl) -i $^ -o $@ -g $(gyhR)/BKG/PDCandidateFitInfor-03.root >$@.log 2>&1
checkZoom/sigguoyh-.h5: $(pdpath)
	mkdir -p $(dir $@)
	$(pythonPrefix)/python3 checkGuoFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 1 2 3 4 5 6 7 8 9 10 347 5796 -m $(mutpl) -a $(antpl) -i $^ -o $@ -g $(gyhR)/SIG_LSPMT/PDNewData.root >$@.log 2>&1
checkZoom/sigguoyh1.h5: $(pdpath)
	mkdir -p $(dir $@)
	python3 checkGuoFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 308 347 1960 2600 2876  4173 4563 4942 7202 7764 7911 8184 8421 8463 8478 8658 8753 9406 9469 9878 -m $(mutpl) -a $(antpl) -i $^ -o $@ -g $(gyhR)/SIG_LSPMT/PDNewData.root >$@.log 2>&1
checkZoom/sigguoyh2.h5: $(pdpath)
	mkdir -p $(dir $@)
	python3 checkGuoFit.py -f 300 -s 1 -b 1 -t $(ktpl) -p $(pitpl) -e 1409 5074 6296 8023 9034 9753 -m $(mutpl) -a $(antpl) -i $^ -o $@ -g $(gyhR)/SIG_LSPMT/PDNewData.root >$@.log 2>&1
# merge data part
bkg/fit300Mtpl/res.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/fit300Mtpl/1nsInit-%.h5) 
	$(pythonPrefix)/python3 mergeInfo.py -i $(bkgpath) -r $^ -o $@ -b 1 -u $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/t90/T90C300Delta0_%.h5) -s $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/shapefeature/C300Delta0_%.h5) -m $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/michel/%.h5) > $@.log
sig/fit300Mtpl/res.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/fit300Mtpl/1nsInit-%.h5)
	$(pythonPrefix)/python3 mergeInfo.py -i $(pdpath) -r $^ -o $@ -b 1 -u $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C300Delta0_%.h5) -s $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/shapefeature/C300Delta0_%.h5) -m $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/michel/%.h5) > $@.log
# merge data with Energy correction part
bkg/fit300Mtpl/fitEc_res.h5: $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/fit300Mtpl/1nsInit-%.h5) 
	$(pythonPrefix)/python3 mergeInfoEc.py -i $(bkgpath) -r $^ -o $@ -b 1 -u $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/t90/T90C300Delta0_%.h5) -s $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/shapefeature/C300Delta0_%.h5) -m $(bkgpath:$(pathPrefix)/BKGsplit/ANRawData_%.root=bkg/michel/%.h5) > $@.log
sig/fit300Mtpl/fitEc_res.h5: $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/fit300Mtpl/1nsInit-%.h5)
	$(pythonPrefix)/python3 mergeInfoEc.py -i $(pdpath) -r $^ -o $@ -b 1 -u $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/t90/T90C300Delta0_%.h5) -s $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/shapefeature/C300Delta0_%.h5) -m $(pdpath:$(pathPrefix)/SIG_LSPMT/PDRawData_%.root=sig/michel/%.h5) > $@.log
## efficiency
bkg/fit300Mtpl/res.pdf: bkg/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 checkEffi.py -i $^ -p $(bkgpath) -o $@ >$@.log 2>&1
sig/fit300Mtpl/res.pdf: sig/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 checkEffi.py -i $^ -p $(pdpath) -o $@ >$@.log 2>&1
# view result part
sig/fit300Mtpl/Fitres.pdf: sig/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 $(srcPrefix)/viewFit.py -i $^ -o $@ >$@.log 2>&1
bkg/fit300Mtpl/Fitres.pdf: bkg/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 $(srcPrefix)/viewFit.py -i $^ -o $@ >$@.log 2>&1
# compare/fit300Mtpl: 
# 	python3 $(srcPrefix)/fit300Mtpl/fitcompare.py -i fit300Mtpl/sigFitres.pdf.h5 -a fit300Mtpl/bkgFitres.pdf.h5 -o test

chiSeries:=$(shell seq 1.5 0.1 3)
fit300Mtpl/bkgchi: $(chiSeries:%=fit300Mtpl/bkgchi%.pdf)
fit300Mtpl/sigchi: $(chiSeries:%=fit300Mtpl/sigchi%.pdf)
bkg/fit300Mtpl/chi%.pdf: bkg/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 $(srcPrefix)/checkEffi.py -i $^ -c $* -p $(bkgpath) -o $@ >$@.log 2>&1
sig/fit300Mtpl/chi%.pdf: sig/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 $(srcPrefix)/checkEffi.py -i $^ -c $* -p $(pdpath) -o $@ >$@.log 2>&1
# adjust the chisquare and deltaT
bkg/fit300Mtpl/Adj%.pdf: bkg/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 paracheck.py -i $^ -c $* -b 1 -p $(bkgpath) -o $@ >$@.log 2>&1
sig/fit300Mtpl/Adj%.pdf: sig/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 paracheck.py -i $^ -c $* -p $(pdpath) -o $@ >$@.log 2>&1
bkg/fit300Mtpl/Adj2.4.pdf: bkg/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 paracheck.py -i $^ -c 1.4 -b 1 -p $(bkgpath) -o $@ >$@.log 2>&1
sig/fit300Mtpl/Adj2.4.pdf: sig/fit300Mtpl/res.h5
	mkdir -p $(dir $@)
	python3 paracheck.py -i $^ -c 1.4 -p $(pdpath) -o $@ >$@.log 2>&1
.DELETE_ON_ERROR:
.SECONDARY:
