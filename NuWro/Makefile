.PHONY: all
export OMP_NUM_THREADS=2
downNu:
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/PublicDatabase/Background/NuWro/nuwro1.9.2_atm_weight_PDG2020.root .
find.h5: nuwro1.9.2_atm_weight_PDG2020.root
	python3 findPdg.py -i $^ -o $@ -p 13 321 2212 > $@.log
upNu:
	rsync -P --rsh='sshpass -p $(PASSWORD) ssh -l junoprotondecay' find.log junoprotondecay@lxslc701.ihep.ac.cn:/junofs/users/junoprotondecay/zhangaq/generator/NuWro/
