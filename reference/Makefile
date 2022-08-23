.PHONY: downbkg downsig
# make PASSWORD=XXX downref
export OMP_NUM_THREADS=2
refpath:=$(wildcard /junofs/users/junoprotondecay/zhangaq/data/analysis/BKG/ANRawData_*.root)
tplfile:=/junofs/users/junoprotondecay/zhangaq/data/protondecay/tplp/e-/tpl1ns.h5
downref:
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:/junofs/production/public/users/guoyh/offline3393/simulation_result/SPMT/Document/reference* .
downNewref:
	mkdir -p newRef
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation_result/reference/Reference/* newRef/
check.png:
	python3 plotRef.py -p reference reference_K_100 reference_Mu_150 reference_Pi_350 -c $(tplfile) -o $@ > $@.log 2>&1
transfer: anTpl.h5 kTpl.h5 muTpl.h5 piTpl.h5
anTpl.h5: reference
	python3 transformat.py -p $^ -s 350 -o $@ >$@.log 2>&1
kTpl.h5: reference_K_100
	python3 transformat.py -p $^ -s 100 -o $@ >$@.log 2>&1
muTpl.h5: reference_Mu_150
	python3 transformat.py -p $^ -s 150 -o $@ >$@.log 2>&1
piTpl.h5: reference_Pi_350
	python3 transformat.py -p $^ -s 350 -o $@ >$@.log 2>&1
transferNew: newRef/anTpl.h5 newRef/kTpl.h5 newRef/muTpl.h5 newRef/piTpl.h5
newRef/anTpl.h5: newRef/AN_380 newRef/AN_386_boundary_15716
	python3 transformat.py -p $< -b $(word 2,$^) -s 380 -o $@ >$@.log 2>&1
newRef/kTpl.h5: newRef/kaon_105 newRef/kaon_105_boundary
	python3 transformat.py -p $< -b $(word 2,$^) -s 105 -o $@ >$@.log 2>&1
newRef/muTpl.h5: newRef/muon_152 newRef/muon_152_boundary
	python3 transformat.py -p $< -b $(word 2,$^) -s 152 -o $@ >$@.log 2>&1
newRef/piTpl.h5: newRef/pion_354_avescaled newRef/pion_354_boundary
	python3 transformat.py -p $< -b $(word 2,$^) -s 354 -o $@ >$@.log 2>&1

newRef/check.png:
	python3 plotRef.py -p newRef/AN_380 newRef/kaon_105 newRef/muon_152 newRef/pion_354 -c $(tplfile) -o $@ > $@.log 2>&1
newRef/check1.png:
	python3 newRef/plotTpl.py -p newRef/anTpl.h5 newRef/kTpl.h5 newRef/muTpl.h5 newRef/piTpl.h5 -o $@ > $@.log 2>&1

.DELETE_ON_ERROR:
.SECONDARY: