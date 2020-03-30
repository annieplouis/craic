projectpath=../craic
datapath=$projectpath/data
outpath=$projectpath/models
runname=msign-comm50-3000000-sent

hiddensize=512
numlayers=1
opname=$runname/H$hiddensize-L$numlayers
mkdir $outpath/$opname


CUDA_VISIBLE_DEVICES=0 python generate_comment.py \
		    --data_dir=$datapath/$runname \
		    --train_dir=$outpath/$opname \
		    --from_train_data=$datapath/$runname/train.from.txt \
		    --to_train_data=$datapath/$runname/train.to.txt \
		    --from_dev_data=$datapath/$runname/valid.from.txt \
		    --to_dev_data=$datapath/$runname/valid.to.txt \
		    --encoder_size=60 \
		    --decoder-size=60 \
		    --size=$hiddensize \
		    --num_layers=$numlayers  \
		    --from_test_data=$datapath/$runname/test.from.txt \
		    --to_test_data=$datapath/$runname/test.to.txt \
		    --test_predictions=$outpath/$opname/test.pred \
		    --decode


