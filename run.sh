projection=1
click_mode='multi'
feature_size=64
gen_sim=1
combine=1
eval_type='devQsplit'
pro_bef=14
pro_aft=64
lr=6e-6
epoch=2

click_pos=-1
click_num='total'

seeds=(0 57 137 361 462)
iter=0

method_name='DLA_DNN'

# params for DualIPW
f_mode='KL'
s1_nlayers=1
s1_hidden=8

loss_mode='listwise'

# params for IOBM
p_model='IOBM'
click_dim=16
s_nlayers=1
s_nhidden=16
bi=1

# params for IPW
ipw=0

# params for Drop and Gradrev
eta=0.7

for seed in ${seeds[@]}
do
    if [ $p_model = 'IOBM' ];then
        output_name=$method_name'_p_model'$p_model'_lr'$lr'_epoch'$epoch'_seed'$seed
    elif [ $ipw = 1 ];then
        output_name=$method_name'_ipw'$ipw'_lr'$lr'_epoch'$epoch'_seed'$seed
    else
        output_name=$method_name'_lr'$lr'_epoch'$epoch'_seed'$seed
    fi

    echo $output_name
    CUDA_VISIBLE_DEVICES=1 python ./ntcir_unbiased_ltr.py \
        --bi $bi \
        --click_dim $click_dim \
        --s_nlayers $s_nlayers \
        --s_nhidden $s_nhidden \
        --eta $eta \
        --p_model $p_model \
        --ipw $ipw \
        --f_mode $f_mode \
        --s1_nlayers $s1_nlayers \
        --s1_hidden $s1_hidden \
        --dropout 0.1 \
        --num_candidates 10 \
        --method_name $method_name \
        --n_gpus 1 \
        --output_name 'total_output/'$method_name'/'$output_name \
        --eval_step 500 \
        --save_step 50000 \
        --combine $combine \
        --n_queries_for_each_gpu 30 \
        --lr $lr \
        --rank_feature_size $feature_size \
        --epoch $epoch \
        --projection $projection \
        --click_mode $click_mode \
        --loss_mode $loss_mode \
        --pro_bef $pro_bef \
        --pro_aft $pro_aft \
        --iter $iter \
        --tra_feat_size 13 \
        --eval_type $eval_type \
        --click_pos $click_pos \
        --seed $seed \
        --click_num $click_num
done
