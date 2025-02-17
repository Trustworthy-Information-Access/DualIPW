n_treeses=(2000)
n_leaveses=(50)
lgb_lrs=(0.5)
seed=0
test_only=0
gen_only=1
pro_bef=14

for n_trees in ${n_treeses[@]}
    do
    for n_leaves in ${n_leaveses[@]}
        do
        for lgb_lr in ${lgb_lrs[@]}
            do
            output_name='lgb_T'$n_trees'_L'$n_leaves'_lr'$lgb_lr
            python ntcir_policy_ltr.py \
                --n_trees $n_trees \
                --n_leaves $n_leaves \
                --lgb_lr $lgb_lr \
                --output_name 'lgb_policy/'$output_name \
                --seed $seed \
                --test_only $test_only \
                --gen_only $gen_only \
                --pro_bef $pro_bef
        done
    done
done