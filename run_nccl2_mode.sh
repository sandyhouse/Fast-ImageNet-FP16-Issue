#!/bin/bash
set +x

#enable_dgc=False
#
#while true ; do
#  case "$1" in
#    -enable_dgc) enable_dgc="$2" ; shift 2 ;;
#    *)
#       if [[ ${#1} > 0 ]]; then
#          echo "not supported arugments ${1}" ; exit 1 ;
#       else
#           break
#       fi
#       ;;
#  esac
#done
#
#case "${enable_dgc}" in
#    True) ;;
#    False) ;;
#    *) echo "not support argument -enable_dgc: ${dgc}" ; exit 1 ;;
#esac

export FLAGS_fraction_of_gpu_memory_to_use=0.34

export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
#export PADDLE_TRAINERS="10.255.129.36,10.255.130.22"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="2"
#export PADDLE_TRAINER_ID="0"

#export NCCL_SOCKET_IFNAME=eth0
unset http_proxy
unset https_proxy

mkdir -p logs

# NOTE: set NCCL_P2P_DISABLE so that can run nccl2 distribute train on one node.

# You can set vlog to see more details' log.
export FLAGS_eager_delete_tensor_gb=0.0
#export GLOG_v=10
export GLOG_logtostderr=1
#export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

export NUM_CARDS=2

PADDLE_CURRENT_ENDPOINT="127.0.0.1:7160" \
PADDLE_TRAINER_ID="0" \
CUDA_VISIBLE_DEVICES="0" \
./python/bin/python -u train.py --update_method nccl2 --data_dir=../fast_resnet_data --fp16=False &> logs/tr0.log &

PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
PADDLE_TRAINER_ID="1" \
CUDA_VISIBLE_DEVICES="1" \
./python/bin/python -u train.py --update_method nccl2 --data_dir=../fast_resnet_data --fp16=False &> logs/tr1.log &

##PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
#PADDLE_TRAINER_IDX="2" \
#CUDA_VISIBLE_DEVICES="2" \
#./python/bin/python -u train.py --update_method nccl2 --data_dir=/ssd2/lilong/fast_resnet_data &> logs/tr2.log &
#
##PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
#PADDLE_TRAINER_IDX="3" \
#CUDA_VISIBLE_DEVICES="3" \
#./python/bin/python -u train.py --update_method nccl2 --data_dir=/ssd2/lilong/fast_resnet_data &> logs/tr3.log &
#
#
##PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
#PADDLE_TRAINER_IDX="4" \
#CUDA_VISIBLE_DEVICES="4" \
#./python/bin/python -u train.py --update_method nccl2 --data_dir=/ssd2/lilong/fast_resnet_data &> logs/tr4.log &
#
##PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
#PADDLE_TRAINER_IDX="5" \
#CUDA_VISIBLE_DEVICES="5" \
#./python/bin/python -u train.py --update_method nccl2 --data_dir=/ssd2/lilong/fast_resnet_data &> logs/tr5.log &
#
##PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
#PADDLE_TRAINER_IDX="6" \
#CUDA_VISIBLE_DEVICES="6" \
#./python/bin/python -u train.py --update_method nccl2 --data_dir=/ssd2/lilong/fast_resnet_data &> logs/tr6.log &
#
##PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
#PADDLE_TRAINER_IDX="7" \
#CUDA_VISIBLE_DEVICES="7" \
#./python/bin/python -u train.py --update_method nccl2 --data_dir=/ssd2/lilong/fast_resnet_data &> logs/tr7.log &

wait
