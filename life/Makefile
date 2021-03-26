.PHONY:all
all: interval_b1.png
interval_b%.png: interval_bkg%.root
	python3 visualInterval.py -i $^ -o $@ >$@.log 2>&1