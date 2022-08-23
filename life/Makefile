.PHONY:all
all: genie3.0.2_atm_weight_PDG2020.root interval_b1.png tau.txt
genie3.0.2_atm_weight_PDG2020.root:
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/PublicDatabase/Background/GENIE/genie3.0.2_atm_weight_PDG2020.root .
nbkg.h5: genie3.0.2_atm_weight_PDG2020.root
	python3 config.py --weight $< -o $@
interval_bkg.root: nbkg.h5
	python3 interval.py -i $^ -o $@
interval_bkg.png: interval_bkg.root
	python3 visualInterval.py -i $^ -o $@ >$@.log 2>&1
tau.txt: interval_bkg.png ../crosscheckNewTplJitEc/sig/fit300Mtpl/efficiency.h5
	python3 tau.py $<.log $(word 2,$^) > $@